from fastapi import APIRouter
import os
from app.services import ml_service, evaluation_service
from app.core.config import settings
from app.services.ml_service import get_clean_datasets_for_training
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

    # 1. Load dữ liệu validation MỘT LẦN cho TẤT CẢ các mô hình
    _, df_val, _ = get_clean_datasets_for_training()
    X_val = df_val['text'].fillna('')
    y_val = df_val['target']

    # 2. Khai báo danh sách các models cần đánh giá
    # Đặt tên file theo đúng format khi bạn lưu ở bước huấn luyện SVM, XGBoost
    models_to_evaluate = [
        {
            "name": "Multinomial Naive Bayes",
            "model_path": settings.MODEL_PATH,
            "vectorizer_path": settings.MODEL_PATH.replace('.pkl', '_vectorizer.pkl')
        },
        {
            "name": "Support Vector Machine (SVM)",
            "model_path": os.path.join(settings.BASE_DIR, "app", "models", "svm_model.pkl"),
            "vectorizer_path": os.path.join(settings.BASE_DIR, "app", "models", "svm_vectorizer.pkl")
        },
        {
            "name": "XGBoost",
            "model_path": os.path.join(settings.BASE_DIR, "app", "models", "xgboost_model.pkl"),
            "vectorizer_path": os.path.join(settings.BASE_DIR, "app", "models", "xgboost_vectorizer.pkl")
        }
    ]

    # 3. Lặp qua cấu hình và dùng hàm đánh giá chung
    for model_info in models_to_evaluate:
        perf_data = evaluation_service.get_general_model_performance(
            model_path=model_info["model_path"],
            vectorizer_path=model_info["vectorizer_path"],
            model_name=model_info["name"],
            X_val=X_val,
            y_val=y_val
        )

        if perf_data:
            results.append(perf_data)
        else:
            # Xử lý fallback nếu mô hình chưa được train/chưa có file pkl
            results.append({
                "model_name": f"{model_info['name']} (Chưa huấn luyện)",
                "training_time_sec": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0
            })

    return {"status": "success", "data": results}