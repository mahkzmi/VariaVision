import ast
import io
import json
import logging
import os
import threading
import time
import warnings
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

app = FastAPI(title="Varia Vision API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILENAME = "best_int8.onnx"
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_FILENAME)
HF_REPO_ID = os.getenv("HF_REPO_ID", "mahkzmi/Varia-Vision-Model")
HF_TOKEN = os.getenv("HF_TOKEN") or None
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "30000000"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "300"))
FREE_LIMIT = int(os.getenv("FREE_LIMIT_PER_DAY", "10"))

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
warnings.simplefilter("error", Image.DecompressionBombWarning)

usage_tracker = {}
usage_lock = threading.Lock()
inference_lock = threading.Lock()
session = None
input_name = None
input_dtype = np.float32
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
        names = {0: "weed"}
        logger.warning("ONNX metadata has no class names; using the known single-class label 'weed'")
    logger.info("ONNX input=%s shape=%s classes=%s providers=%s", inp.name, shape, names, sess.get_providers())
    return sess, inp.name, width, height, names, (np.float16 if inp.type == "tensor(float16)" else np.float32)


@app.on_event("startup")
def load_model():
    global session, input_name, input_width, input_height, class_names, input_dtype
    path = MODEL_PATH
    try:
        if not os.path.isfile(path):
            logger.info("Downloading model %s from Hugging Face", MODEL_FILENAME)
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
        session, input_name, input_width, input_height, class_names, input_dtype = load_onnx_model(path)
        logger.info("Varia Vision ONNX Runtime model loaded from %s", path)
    except Exception:
        session = None
        logger.exception("Varia Vision model initialization failed")
        raise RuntimeError("Varia Vision model could not be initialized")


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
    image = ImageOps.exif_transpose(image)
    original_width, original_height = image.size

    # Downscale large photos before RGB conversion to keep memory bounded on the free instance.
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    work_width, work_height = image.size
    scale = min(input_width / work_width, input_height / work_height)
    resized_width = max(1, round(work_width * scale))
    resized_height = max(1, round(work_height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR).convert("RGB")

    canvas = Image.new("RGB", (input_width, input_height), (114, 114, 114))
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    tensor = np.asarray(canvas, dtype=input_dtype) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    return tensor, original_width, original_height, scale_x, scale_y, pad_x, pad_y


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
    tensor, original_width, original_height, scale_x, scale_y, pad_x, pad_y = preprocess(image)
    with inference_lock:
        outputs = session.run(None, {input_name: tensor})
    del tensor
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
        boxes = np.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), axis=1).astype(np.float32)

    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= CONF_THRESHOLD)
    valid &= (class_ids >= 0) & (class_ids < len(class_names))
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    if not len(scores):
        return []
    if not end_to_end:
        indices = nms(boxes, scores, class_ids)
        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]

    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / scale_x, 0, original_width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / scale_y, 0, original_height)
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
    return {
        "message": "Varia Vision API is running.",
        "model_loaded": session is not None,
        "free_limit": f"{FREE_LIMIT} requests/day per running instance",
    }


@app.get("/health")
def health():
    if session is None:
        raise HTTPException(status_code=503, detail="Detection model is unavailable.")
    return {"status": "ok", "model_loaded": True, "runtime": "onnxruntime"}


@app.post("/detect")
async def detect_weed(request: Request, file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="Detection model is unavailable.")

    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as probe:
            image_format = probe.format
            width, height = probe.size
            if width < 2 or height < 2:
                raise HTTPException(status_code=400, detail="Image dimensions are too small.")
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image has {width * height:,} pixels; the safe limit is {MAX_IMAGE_PIXELS:,}.",
                )
            probe.verify()

        with Image.open(io.BytesIO(image_bytes)) as source:
            source.seek(0)  # First frame for animated images.
            source.load()
            image = source.copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(
            status_code=400,
            detail="This file is not a supported or valid image. Try JPG, PNG, WEBP, BMP, GIF, or TIFF.",
        )

    client_ip = request.client.host if request.client else "unknown"
    check_limit(client_ip)
    started = time.perf_counter()
    logger.info(
        "Detection started: filename=%s bytes=%d format=%s dimensions=%dx%d client=%s",
        file.filename, len(image_bytes), image_format, width, height, client_ip,
    )
    try:
        detections = run_inference(image)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "Detection completed: filename=%s detections=%d elapsed_ms=%s",
            file.filename, len(detections), elapsed_ms,
        )
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "image_width": width,
            "image_height": height,
            "detections_count": len(detections),
            "detections": detections,
            "processing_time_ms": elapsed_ms,
        })
    except Exception:
        logger.exception("Image inference failed")
        raise HTTPException(status_code=500, detail="Image inference failed. Please retry with another image.")
    finally:
        image.close()
        del image_bytes


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
