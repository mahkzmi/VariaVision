import io
import logging
import os
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("varia-vision")

app = FastAPI(title="Varia Vision API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
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

FREE_LIMIT = int(os.getenv("FREE_LIMIT_PER_DAY", "10"))
# This counter is process-local and resets on restart. Use shared storage before
# treating the free tier as an enforceable quota across multiple instances.
usage_tracker = {}
usage_lock = threading.Lock()
model = None


def load_yolo_model(model_path: str):
    """Import the heavy inference runtime only when the service starts."""
    from ultralytics import YOLO
    return YOLO(model_path)


@app.on_event("startup")
def load_model():
    """Load the detector before accepting traffic; fail startup if unavailable."""
    global model
    try:
        resolved_model_path = MODEL_PATH
        if not os.path.isfile(resolved_model_path):
            logger.info("Downloading model %s from Hugging Face", MODEL_FILENAME)
            resolved_model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                token=HF_TOKEN,
            )
        model = load_yolo_model(resolved_model_path)
        logger.info("Varia Vision model loaded from %s", resolved_model_path)
    except Exception as exc:
        model = None
        logger.exception("Varia Vision model initialization failed")
        raise RuntimeError("Varia Vision model could not be initialized") from exc


def check_limit(client_ip: str):
    today = datetime.now(timezone.utc).date().isoformat()
    with usage_lock:
        current = usage_tracker.get(client_ip)
        if current is None or current["date"] != today:
            current = {"date": today, "count": 0}
            usage_tracker[client_ip] = current
        if current["count"] >= FREE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Free limit of {FREE_LIMIT} requests per day exceeded.",
            )
        current["count"] += 1


@app.get("/")
def read_root():
    return {
        "message": "Varia Vision API is running.",
        "model_loaded": model is not None,
        "free_limit": f"{FREE_LIMIT} requests/day per running instance",
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Detection model is unavailable.")
    return {"status": "ok", "model_loaded": True}


@app.post("/detect")
async def detect_weed(request: Request, file: UploadFile = File(...)):
    if model is None:
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
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.verify()
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid supported image.")

    client_ip = request.client.host if request.client else "unknown"
    check_limit(client_ip)
    logger.info(
        "Detection request received: filename=%s client_ip=%s",
        file.filename,
        client_ip,
    )

    try:
        results = model(image, conf=0.25, verbose=False)
        detections = []
        for box in results[0].boxes:
            detections.append(
                {
                    "class": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(value, 2) for value in box.xyxy[0].tolist()],
                }
            )
        return JSONResponse(
            content={
                "status": "success",
                "filename": file.filename,
                "detections_count": len(detections),
                "detections": detections,
            }
        )
    except Exception:
        logger.exception("Image inference failed")
        raise HTTPException(status_code=500, detail="Image inference failed. Please retry later.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
