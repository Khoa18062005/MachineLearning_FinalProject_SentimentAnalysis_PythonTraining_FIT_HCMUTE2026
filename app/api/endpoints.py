from fastapi import APIRouter
import os

from starlette.responses import FileResponse

from app.models.Multinomial_Custom import predict_MNB_Custom, load_model
from app.models.Multinomial_Library import load_library_model
from app.services import ml_service, evaluation_service
from app.core.config import settings
from app.services.evaluation_service import calculate_performance_metrics
from app.services.ml_service import get_clean_datasets_for_training, clean_text

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
        priors, w_probs, vocab, totals, counts, train_time = load_model(model_path)

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

    lib_pipeline, lib_time = load_library_model()
    lib_key = "MNB_Library"

    if lib_pipeline:
        if lib_key in PREDICTION_CACHE:
            y_pred_lib = PREDICTION_CACHE[lib_key]
        else:
            y_pred_lib = lib_pipeline.predict(X_test)
            PREDICTION_CACHE[lib_key] = y_pred_lib

        lib_metrics = calculate_performance_metrics(
            model_name="Multinomial Naive Bayes (Library)",
            y_true=y_test,
            y_pred=y_pred_lib,
            training_time_sec=lib_time
        )
        results.append(lib_metrics)
    else:
        # THÊM NHÁNH NÀY ĐỂ FRONTEND KHÔNG BỊ LỖI KHI CHƯA TRAIN
        results.append({
            "model_name": "Multinomial NB Library (Chưa huấn luyện)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0
        })

    return {"status": "success", "data": results}


# @router.get("/model-errors")
# async def get_model_errors(model_name: str):
#     _, _, df_test = get_clean_datasets_for_training()
#     X_test = df_test['text'].fillna('')
#
#     y_pred = []
#
#     # Xử lý mô hình Multinomial Naive Bayes (Custom)
#     if model_name == "Multinomial Naive Bayes (Custom)":
#         model_key = "MNB_Custom"
#
#         # ĐỌC TỪ CACHE: Cực kỳ nhanh!
#         if model_key in PREDICTION_CACHE:
#             y_pred = PREDICTION_CACHE[model_key]
#         else:
#             # Fallback (Phòng hờ): Lỡ server bị restart mất cache thì mới phải tính lại
#             model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
#             priors, w_probs, vocab, totals, counts, _ = load_model(model_path)
#             for text in X_test:
#                 label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
#                 y_pred.append(label)
#             PREDICTION_CACHE[model_key] = y_pred
#
#     #=========================================
#     # ... (Mấy ông viết tiếp phần xử lý mô hình tương tự ở đây nhé) ...
#
#     #=========================================
#
#     df_result = df_test.copy()
#     df_result['predicted'] = y_pred
#     df_errors = df_result[df_result['target'] != df_result['predicted']]
#     result_data = df_errors[['target', 'predicted', 'text']].to_dict(orient="records")
#
#     return {"status": "success", "data": result_data}

@router.get("/model-errors")
async def get_model_errors(model_name: str):
    _, _, df_test = get_clean_datasets_for_training()
    X_test = df_test['text'].fillna('')

    y_pred = []

    # --- 1. Xử lý mô hình Multinomial Naive Bayes (Custom) ---
    if model_name == "Multinomial Naive Bayes (Custom)":
        model_key = "MNB_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
            priors, w_probs, vocab, totals, counts, _ = load_model(model_path)
            for text in X_test:
                label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
                y_pred.append(label)
            PREDICTION_CACHE[model_key] = y_pred

    # --- 2. Xử lý mô hình Multinomial Naive Bayes (Library) [MỚI BỔ SUNG] ---
    elif model_name == "Multinomial Naive Bayes (Library)":
        model_key = "MNB_Library"

        # Lấy từ Cache cho nhanh (nếu trước đó đã bấm "Kết quả Huấn luyện")
        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            # Fallback: Tính lại nếu server vừa khởi động lại mà chưa có cache
            from app.models.Multinomial_Library import load_library_model
            lib_pipeline, _ = load_library_model()
            if lib_pipeline:
                y_pred = lib_pipeline.predict(X_test)
                PREDICTION_CACHE[model_key] = y_pred

    #=========================================
    # Để sẵn chỗ sau này làm cho SVM và XGBoost
    #=========================================

    # Tránh lỗi nếu tên mô hình gửi lên không khớp với bất kỳ if/elif nào
    if len(y_pred) == 0:
        return {"status": "success", "data": []}

    # Gom dữ liệu và lọc ra các câu dự đoán sai
    df_result = df_test.copy()
    df_result['predicted'] = y_pred
    df_errors = df_result[df_result['target'] != df_result['predicted']]
    result_data = df_errors[['target', 'predicted', 'text']].to_dict(orient="records")

    return {"status": "success", "data": result_data}

@router.get("/model-details")
async def get_model_prediction_details(text: str):
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
    if not os.path.exists(model_path):
        return {"status": "error", "message": "Model not found"}

    priors, w_probs, vocab, totals, counts, _ = load_model(model_path)

    # Import hàm vừa viết ở bước 1
    from app.models.Multinomial_Custom import get_prediction_details
    details = get_prediction_details(text, priors, w_probs, vocab, totals, counts)

    return {"status": "success", "data": details}


@router.get("/charts/{model_type}")
async def get_model_charts(model_type: str):
    # Cập nhật danh sách map ảnh cho khớp với Frontend
    charts_map = {
        "mnb": "laplace_smoothing_study.png",     # Dành cho Tab Trực quan tham số
        "mnb_custom_cm": "cm_mnb_custom.png",       # CM của model Custom
        "mnb_library_cm": "cm_mnb_library.png",     # CM của model Library
        "svm_cm": "cm_svm.png",
        "xgb_cm": "cm_xgb.png"
    }

    if model_type not in charts_map:
        return {"status": "error", "message": "Loại biểu đồ không hợp lệ!"}

    file_name = charts_map[model_type]
    chart_path = os.path.join(settings.BASE_DIR, file_name)

    if os.path.exists(chart_path):
        return FileResponse(chart_path)

    return {"status": "error", "message": f"Biểu đồ {model_type} chưa được tạo!"}

@router.get("/predict-text")
async def predict_new_text(text: str):
    # 1. Tiền xử lý văn bản (yêu cầu của bạn)
    cleaned_text = clean_text(text)

    # Khởi tạo dictionary chứa kết quả. (-1 nghĩa là chưa có mô hình)
    custom_preds = {"mnb": -1, "svm": -1, "xgb": -1}
    lib_preds = {"mnb": -1, "svm": -1, "xgb": -1}

    # --- 2A. Nhóm Custom ---
    try:
        model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
        if os.path.exists(model_path):
            priors, w_probs, vocab, totals, counts, _ = load_model(model_path)
            label, _ = predict_MNB_Custom(cleaned_text, priors, w_probs, vocab, totals)
            custom_preds["mnb"] = label
    except Exception as e:
        pass

    # --- 2B. Nhóm Library ---
    try:
        from app.models.Multinomial_Library import load_library_model
        lib_pipeline, _ = load_library_model()
        if lib_pipeline:
            # predict() của sklearn yêu cầu đầu vào là mảng 1 chiều
            label = int(lib_pipeline.predict([cleaned_text])[0])
            lib_preds["mnb"] = label
    except Exception as e:
        pass

    # (Sau này bạn code SVM và XGBoost xong thì bổ sung logic gọi mô hình vào đây)

    # --- 3. Cơ chế BỎ PHIẾU (Majority Vote) ---
    def get_vote(preds_dict):
        # Chỉ lấy các phiếu hợp lệ (khác -1)
        valid_preds = [v for v in preds_dict.values() if v != -1]
        if not valid_preds: return -1 # Chưa có mô hình nào chạy
        # Lấy nhãn xuất hiện nhiều nhất
        return max(set(valid_preds), key=valid_preds.count)

    custom_vote = get_vote(custom_preds)
    lib_vote = get_vote(lib_preds)

    # Kết quả chung cuộc: Tạm thời lấy ý kiến của nhóm Library làm mỏ neo
    final_vote = lib_vote if lib_vote != -1 else custom_vote

    return {
        "status": "success",
        "data": {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "custom": custom_preds,
            "library": lib_preds,
            "votes": {
                "custom": custom_vote,
                "library": lib_vote,
                "final": final_vote
            }
        }
    }