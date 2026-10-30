import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Varia Vision API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best.pt"
HF_REPO_ID = os.getenv("HF_REPO_ID", "mahkzmi/Varia-Vision-Model")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not os.path.exists(MODEL_PATH):
    try:
        print("Downloading model securely from Hugging Face...")
        MODEL_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="best.pt",
            token=HF_TOKEN
        )
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")

# --- ۲. API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Varia Vision API is running. Send a POST request to /detect."}

@app.post("/detect")
async def detect_weed(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        results = model(image, conf=0.25)
        
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
            })
            
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "detections_count": len(detections),
            "detections": detections
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)