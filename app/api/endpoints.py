from fastapi import APIRouter
from app.services import ml_service

router = APIRouter()

@router.get("/preview-data")
async def preview():
    data = ml_service.get_data_preview(limit=200)
    return {"status": "success", "data": data}
@router.get("/features-training")
async def features_training():
    data = ml_service.get_data_features(limit=200)
    return {"status": "success", "data": data}