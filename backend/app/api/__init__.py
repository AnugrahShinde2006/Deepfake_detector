from fastapi import APIRouter
from app.api.endpoints import detection

router = APIRouter()
router.include_router(detection.router, prefix="/detect", tags=["detection"])
