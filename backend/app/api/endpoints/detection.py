from fastapi import APIRouter, File, UploadFile
import time
import random

router = APIRouter()

@router.post("/")
async def detect_media(file: UploadFile = File(...)):
    # Mocking ML inference delay
    time.sleep(2)
    
    # Mock results
    visual_score = random.uniform(0.1, 0.99)
    audio_score = random.uniform(0.1, 0.99)
    temporal_score = random.uniform(0.1, 0.99)
    
    fusion_score = (visual_score * 0.4) + (audio_score * 0.3) + (temporal_score * 0.3)
    is_fake = fusion_score > 0.5
    
    return {
        "filename": file.filename,
        "prediction": "FAKE" if is_fake else "REAL",
        "confidence": fusion_score,
        "details": {
            "visual_score": visual_score,
            "audio_score": audio_score,
            "temporal_score": temporal_score
        }
    }
