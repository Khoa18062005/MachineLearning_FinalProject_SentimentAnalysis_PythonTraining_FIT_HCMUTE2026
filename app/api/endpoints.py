from fastapi import APIRouter
import os
from app.models.Multinomial_Custom import predict_MNB_Custom, load_model
from app.services import ml_service, evaluation_service
from app.core.config import settings
from app.services.evaluation_service import calculate_performance_metrics
from app.services.ml_service import get_clean_datasets_for_training
router = APIRouter()
PREDICTION_CACHE = {}
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
    _, _, df_test = get_clean_datasets_for_training()
    X_test = df_test['text'].fillna('')
    y_test = df_test['target']

    # --- ĐÁNH GIÁ MÔ HÌNH MNB CUSTOM ---
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
    model_key = "MNB_Custom" # Đặt một cái tên khóa cho Cache

    if os.path.exists(model_path):
        priors, w_probs, vocab, totals, train_time = load_model(model_path)

        # KIỂM TRA CACHE: Nếu đã có thì lấy ra, nếu chưa thì mới chạy dự đoán
        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            y_pred = []
            for text in X_test:
                label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
                y_pred.append(label)
            # Tính xong thì LƯU VÀO CACHE cho lần sau dùng
            PREDICTION_CACHE[model_key] = y_pred

        mnb_metrics = calculate_performance_metrics(
            model_name="Multinomial Naive Bayes (Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time
        )
        results.append(mnb_metrics)
    else:
        results.append({
            "model_name": "Multinomial Naive Bayes (Chưa huấn luyện)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0
        })

    return {"status": "success", "data": results}


@router.get("/model-errors")
async def get_model_errors(model_name: str):
    _, _, df_test = get_clean_datasets_for_training()
    X_test = df_test['text'].fillna('')

    y_pred = []

    # Xử lý mô hình Multinomial Naive Bayes (Custom)
    if model_name == "Multinomial Naive Bayes (Custom)":
        model_key = "MNB_Custom"

        # ĐỌC TỪ CACHE: Cực kỳ nhanh!
        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            # Fallback (Phòng hờ): Lỡ server bị restart mất cache thì mới phải tính lại
            model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
            priors, w_probs, vocab, totals, _ = load_model(model_path)
            for text in X_test:
                label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
                y_pred.append(label)
            PREDICTION_CACHE[model_key] = y_pred

    #=========================================
    # ... (Mấy ông viết tiếp phần xử lý mô hình tương tự ở đây nhé) ...

    #=========================================

    df_result = df_test.copy()
    df_result['predicted'] = y_pred
    df_errors = df_result[df_result['target'] != df_result['predicted']]
    result_data = df_errors[['target', 'predicted', 'text']].to_dict(orient="records")

    return {"status": "success", "data": result_data}