import ast
import io
import json
import logging
import os
import threading
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("varia-vision")

app = FastAPI(title="Varia Vision API")
allowed_origins = [x.strip() for x in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

MODEL_FILENAME = "best_int8.onnx"
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_FILENAME)
HF_REPO_ID = os.getenv("HF_REPO_ID", "mahkzmi/Varia-Vision-Model")
HF_TOKEN = os.getenv("HF_TOKEN") or None
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "300"))
FREE_LIMIT = int(os.getenv("FREE_LIMIT_PER_DAY", "10"))

usage_tracker = {}
usage_lock = threading.Lock()
session = None
input_name = None
input_width = input_height = 640
class_names = {}


def parse_names(raw):
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if isinstance(value, list):
        return {i: str(name) for i, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {int(k): str(v) for k, v in value.items()}
        except (TypeError, ValueError):
            return None
    return None


def load_onnx_model(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.enable_mem_pattern = False
    opts.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    if len(shape) != 4 or inp.type not in ("tensor(float)", "tensor(float16)"):
        raise RuntimeError(f"Unsupported ONNX input: type={inp.type}, shape={shape}")
    height = shape[2] if isinstance(shape[2], int) else 640
    width = shape[3] if isinstance(shape[3], int) else 640
    names = parse_names((sess.get_modelmeta().custom_metadata_map or {}).get("names"))
    if not names:
        names = {0: "weed", 1: "crop"}
        logger.warning("ONNX metadata has no class names; fallback labels are in use")
    logger.info("ONNX input=%s shape=%s classes=%s providers=%s", inp.name, shape, names, sess.get_providers())
    return sess, inp.name, width, height, names


@app.on_event("startup")
def load_model():
    global session, input_name, input_width, input_height, class_names
    path = MODEL_PATH
    try:
        if not os.path.isfile(path):
            logger.info("Downloading model %s from Hugging Face", MODEL_FILENAME)
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
        session, input_name, input_width, input_height, class_names = load_onnx_model(path)
        logger.info("Varia Vision ONNX Runtime model loaded from %s", path)
    except Exception as exc:
        session = None
        logger.exception("Varia Vision model initialization failed")
        raise RuntimeError("Varia Vision model could not be initialized") from exc


def check_limit(client_ip):
    today = datetime.now(timezone.utc).date().isoformat()
    with usage_lock:
        current = usage_tracker.get(client_ip)
        if current is None or current["date"] != today:
            current = {"date": today, "count": 0}
            usage_tracker[client_ip] = current
        if current["count"] >= FREE_LIMIT:
            raise HTTPException(status_code=429, detail=f"Free limit of {FREE_LIMIT} requests per day exceeded.")
        current["count"] += 1


def preprocess(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    scale = min(input_width / original_width, input_height / original_height)
    rw, rh = max(1, round(original_width * scale)), max(1, round(original_height * scale))
    resized = image.resize((rw, rh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_width, input_height), (114, 114, 114))
    pad_x, pad_y = (input_width - rw) // 2, (input_height - rh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor, original_width, original_height, scale, pad_x, pad_y


def iou_one_to_many(box, boxes):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-7)


def nms(boxes, scores, class_ids):
    keep = []
    for cid in np.unique(class_ids):
        indices = np.where(class_ids == cid)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]
        while indices.size:
            chosen = int(indices[0])
            keep.append(chosen)
            if len(keep) >= MAX_DETECTIONS:
                return keep
            if indices.size == 1:
                break
            rest = indices[1:]
            indices = rest[iou_one_to_many(boxes[chosen], boxes[rest]) <= IOU_THRESHOLD]
    return sorted(keep, key=lambda i: float(scores[i]), reverse=True)[:MAX_DETECTIONS]


def run_inference(image):
    tensor, ow, oh, scale, pad_x, pad_y = preprocess(image)
    outputs = session.run(None, {input_name: tensor})
    if not outputs:
        raise RuntimeError("ONNX model returned no outputs")
    output = np.squeeze(np.asarray(outputs[0]))
    if output.ndim != 2:
        raise RuntimeError(f"Unexpected ONNX output shape: {outputs[0].shape}")

    # End-to-end YOLO export: rows are x1,y1,x2,y2,confidence,class_id.
    end_to_end = output.shape[0] <= 1000 and output.shape[1] == 6
    if end_to_end:
        boxes = output[:, :4].astype(np.float32)
        scores = output[:, 4].astype(np.float32)
        class_ids = output[:, 5].astype(np.int32)
    else:
        features = 4 + len(class_names)
        if output.shape[0] == features:
            rows = output.T
        elif output.shape[1] == features:
            rows = output
        elif output.shape[0] < output.shape[1]:
            rows = output.T
        else:
            rows = output
        if rows.shape[1] < 5:
            raise RuntimeError(f"Unexpected YOLO output shape: {outputs[0].shape}")
        class_scores = rows[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(len(class_ids)), class_ids].astype(np.float32)
        cx, cy, width, height = (rows[:, i] for i in range(4))
        boxes = np.stack((cx-width/2, cy-height/2, cx+width/2, cy+height/2), axis=1).astype(np.float32)

    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= CONF_THRESHOLD)
    valid &= (class_ids >= 0) & (class_ids < len(class_names))
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    if not len(scores):
        return []
    if not end_to_end:
        indices = nms(boxes, scores, class_ids)
        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]

    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / scale, 0, ow)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / scale, 0, oh)
    return [
        {
            "class": class_names.get(int(cid), f"class_{int(cid)}"),
            "confidence": round(float(score), 3),
            "bbox": [round(float(v), 2) for v in box],
        }
        for box, score, cid in zip(boxes, scores, class_ids)
    ]


@app.get("/")
def read_root():
    return {"message": "Varia Vision API is running.", "model_loaded": session is not None,
            "free_limit": f"{FREE_LIMIT} requests/day per running instance"}


@app.get("/health")
def health():
    if session is None:
        raise HTTPException(status_code=503, detail="Detection model is unavailable.")
    return {"status": "ok", "model_loaded": True, "runtime": "onnxruntime"}


@app.post("/detect")
async def detect_weed(request: Request, file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Detection model is unavailable.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.verify()
        with Image.open(io.BytesIO(image_bytes)) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid supported image.")

    client_ip = request.client.host if request.client else "unknown"
    check_limit(client_ip)
    logger.info("Detection request received: filename=%s bytes=%s client_ip=%s", file.filename, len(image_bytes), client_ip)
    try:
        detections = run_inference(image)
        return JSONResponse(content={"status": "success", "filename": file.filename,
                                     "detections_count": len(detections), "detections": detections})
    except Exception:
        logger.exception("Image inference failed")
        raise HTTPException(status_code=500, detail="Image inference failed. Please retry later.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
