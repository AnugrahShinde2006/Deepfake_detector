import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our PyTorch ML Predictor
from ml.inference.predictor import DeepfakePredictor

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor (This loads PyTorch models into the GPU)
predictor = DeepfakePredictor()

class Details(BaseModel):
    visual_score: float
    audio_score: float
    temporal_score: float

class AnalysisResult(BaseModel):
    prediction: str
    confidence: float
    details: Details
    extracted_faces: list[str] = []

@app.post("/api/detect/", response_model=AnalysisResult)
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")
        
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"Starting ML inference on {file.filename}...")
        results = predictor.analyze_video(file_path)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return AnalysisResult(
            prediction="FAKE" if results["is_fake"] else "REAL",
            confidence=results["overall_score"],
            details=Details(
                visual_score=results["visual_score"],
                audio_score=results["audio_score"],
                temporal_score=results["temporal_score"]
            ),
            extracted_faces=results.get("extracted_faces", [])
        )
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing video")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "gpu_enabled": str(predictor.device) != 'cpu'}
