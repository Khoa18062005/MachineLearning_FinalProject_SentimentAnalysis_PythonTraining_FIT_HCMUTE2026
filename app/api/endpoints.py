from fastapi import APIRouter
from app.services import ml_service, evaluation_service

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


@router.get("/train-models")
async def get_training_results():
    results = []

    # Gọi hàm từ service mới
    mnb_real_data = evaluation_service.get_mnb_real_performance()

    if mnb_real_data:
        results.append(mnb_real_data)
    else:
        results.append({
            "model_name": "Multinomial NB (Chưa có Model)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0
        })

    # Dữ liệu giả định cho SVM và XGBoost
    results.extend([
        {
            "model_name": "Support Vector Machine (SVM)",
            "training_time_sec": 45.30,
            "correct_predictions": 26100,
            "incorrect_predictions": 3900,
            "accuracy": 87.00
        },
        {
            "model_name": "XGBoost",
            "training_time_sec": 120.50,
            "correct_predictions": 27000,
            "incorrect_predictions": 3000,
            "accuracy": 90.00
        }
    ])

    return {"status": "success", "data": results}