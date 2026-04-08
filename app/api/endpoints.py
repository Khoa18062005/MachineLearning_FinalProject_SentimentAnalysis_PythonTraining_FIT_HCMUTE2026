from fastapi import APIRouter
import os
from app.models.Multinomial_Custom import predict_MNB_Custom, load_model
from app.services import ml_service, evaluation_service
from app.core.config import settings
from app.services.evaluation_service import calculate_performance_metrics
from app.services.ml_service import get_clean_datasets_for_training
router = APIRouter()

@router.get("/preview-data")
async def preview():
    data = ml_service.get_data_preview(limit=200000)
    return {"status": "success", "data": data}

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

    # Lấy dữ liệu trên tập Test
    _, _, df_test = get_clean_datasets_for_training()
    X_test = df_test['text'].fillna('')
    y_test = df_test['target']

    # Đánh giá mô hình Multinomial Naive Bayes (Custom)
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")

    if os.path.exists(model_path):
        # A. Tải mô hình và thời gian huấn luyện
        priors, w_probs, vocab, totals, train_time = load_model(model_path)

        # B. Tiến hành dự đoán trên toàn bộ tập Test
        y_pred = []
        for text in X_test:
            label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
            y_pred.append(label)

        # C. Đưa vào hàm đánh giá để lấy 3 thông số
        mnb_metrics = calculate_performance_metrics(
            model_name="Multinomial Naive Bayes (Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time
        )
        results.append(mnb_metrics)
    else:
        # Trả về 0 nếu chưa huấn luyện
        results.append({
            "model_name": "Multinomial Naive Bayes (Chưa huấn luyện)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0
        })
    return {"status": "success", "data": results}