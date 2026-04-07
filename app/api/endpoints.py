from fastapi import APIRouter
from app.services import ml_service

router = APIRouter()

@router.get("/preview-data")
async def preview():
    data = ml_service.get_data_preview(limit=200000)
    return {"status": "success", "data": data}
# @router.get("/features-training")
# async def features_training():
#     data = ml_service.get_data_features(limit=200000)
#     return {"status": "success", "data": data}

@router.get("/dataset/{set_name}")
async def get_dataset(set_name: str):
    # Gọi hàm tổng quát bên ml_service
    result = ml_service.get_dataset_by_name(set_name=set_name, limit=200000)

    return {
        "status": "success",
        "data": result["records"],
        "counts": result["counts"]  # Trả về cả số lượng mẫu để frontend hiển thị
    }

@router.get("/clean-dataset/{set_name}")
async def get_clean_dataset(set_name: str):
    # Gọi hàm lấy dữ liệu sạch
    result = ml_service.get_clean_dataset_by_name(set_name=set_name, limit=200000)

    return {
        "status": "success",
        "data": result["records"],
        "counts": result["counts"]
    }