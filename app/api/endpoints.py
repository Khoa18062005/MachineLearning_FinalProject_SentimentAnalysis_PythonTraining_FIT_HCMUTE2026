from fastapi import APIRouter
import os

from app.models.Multinomial_Custom import predict_MNB_Custom, load_model
from app.models.XGBoost_Custom import (
    predict_XGB_Custom,
    load_model as load_xgb_model,
    load_vectorizer
)

from app.models.XGBoost_Library import (
    predict_XGB_Library,
    load_model as load_xgb_library_model,
    load_vectorizer as load_xgb_library_vectorizer
)

from app.services import ml_service, evaluation_service
from app.core.config import settings
from app.services.evaluation_service import calculate_performance_metrics
from app.services.ml_service import get_clean_datasets_for_training

from starlette.responses import FileResponse

from app.models.Multinomial_Custom import predict_MNB_Custom, load_model
from app.models.Multinomial_Library import load_library_model
from app.services import ml_service, evaluation_service
from app.core.config import settings
from app.services.evaluation_service import calculate_performance_metrics
from app.services.ml_service import get_clean_datasets_for_training, clean_text

from app.models.SVM_OneSample_Custom import load_model as load_svm_one_model, encode_target_for_svm
from app.models.SVM_FullSample_Custom import load_model as load_svm_full_model
import joblib
import numpy as np
router = APIRouter()
PREDICTION_CACHE = {}


@router.get("/preview-data")
async def preview():
    data = ml_service.get_data_preview(limit=200000)
    return {"status": "success", "data": data}


@router.get("/dataset/{set_name}")
async def get_dataset(set_name: str):
    result = ml_service.get_dataset_by_name(set_name=set_name, limit=200000)

    return {
        "status": "success",
        "data": result["records"],
        "counts": result["counts"]
    }


@router.get("/clean-dataset/{set_name}")
async def get_clean_dataset(set_name: str):
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
    X_test = df_test["text"].fillna("")
    y_test = df_test["target"]

    # =========================
    # 1) MNB CUSTOM
    # =========================
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
    model_key = "MNB_Custom"

    if os.path.exists(model_path):
        priors, w_probs, vocab, totals, counts, train_time = load_model(model_path)

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            y_pred = []
            for text in X_test:
                label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
                y_pred.append(label)
            PREDICTION_CACHE[model_key] = y_pred

        mnb_metrics = calculate_performance_metrics(
            model_name="Multinomial Naive Bayes (Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time
        )
        results.append(mnb_metrics)

    # =========================
    # 2) SVM ONE-SAMPLE CUSTOM
    # =========================
    svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")
    svm_one_key = "SVM_OneSample_Custom"

    if os.path.exists(svm_one_path):
        svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)

        if svm_one_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[svm_one_key]
        else:
            X_test_tfidf = svm_one_vectorizer.transform(X_test).toarray()
            y_pred_svm = svm_one_model.predict(X_test_tfidf)
            y_pred = np.where(y_pred_svm == 1, 4, 0).tolist()
            PREDICTION_CACHE[svm_one_key] = y_pred

        svm_one_metrics = calculate_performance_metrics(
            model_name="Linear SVM (One-Sample Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=svm_one_model.training_time_sec
        )
        results.append(svm_one_metrics)

    # =========================
    # 3) SVM FULL-SAMPLE CUSTOM
    # =========================
    svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
    svm_full_key = "SVM_FullSample_Custom"

    if os.path.exists(svm_full_path):
        svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)

        if svm_full_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[svm_full_key]
        else:
            X_test_tfidf = svm_full_vectorizer.transform(X_test).toarray()
            y_pred_svm = svm_full_model.predict(X_test_tfidf)
            y_pred = np.where(y_pred_svm == 1, 4, 0).tolist()
            PREDICTION_CACHE[svm_full_key] = y_pred

        svm_full_metrics = calculate_performance_metrics(
            model_name="Linear SVM (Full-Sample Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=svm_full_model.training_time_sec
        )
        results.append(svm_full_metrics)

    # =========================
    # 4) SVM LIBRARY
    # =========================
    svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
    svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")
    svm_lib_key = "SVM_Library"

    if os.path.exists(svm_lib_model_path) and os.path.exists(svm_lib_vectorizer_path):
        svm_lib_model = joblib.load(svm_lib_model_path)
        svm_lib_vectorizer = joblib.load(svm_lib_vectorizer_path)

        if svm_lib_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[svm_lib_key]
        else:
            X_test_tfidf = svm_lib_vectorizer.transform(X_test)
            y_pred = svm_lib_model.predict(X_test_tfidf).tolist()
            PREDICTION_CACHE[svm_lib_key] = y_pred

        svm_lib_metrics = calculate_performance_metrics(
            model_name="Linear SVM (Library)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=getattr(svm_lib_model, "training_time_sec", 0.0)
        )
        results.append(svm_lib_metrics)

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

    # =========================================================
    # 2) ĐÁNH GIÁ MÔ HÌNH XGBOOST CUSTOM
    # =========================================================
    xgb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_custom.pkl")
    xgb_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer.pkl")
    xgb_model_key = "XGB_Custom"

    if os.path.exists(xgb_model_path) and os.path.exists(xgb_vectorizer_path):
        (
            base_score_logit,
            trees,
            learning_rate,
            feature_names,
            classes,
            threshold,
            train_time,
            params
        ) = load_xgb_model(xgb_model_path)

        vectorizer = load_vectorizer(xgb_vectorizer_path)

        if xgb_model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[xgb_model_key]
        else:
            y_pred = []
            for text in X_test:
                label, _ = predict_XGB_Custom(
                    text_input=text,
                    base_score_logit=base_score_logit,
                    trees=trees,
                    learning_rate=learning_rate,
                    feature_names=feature_names,
                    classes=classes,
                    threshold=threshold,
                    vectorizer=vectorizer
                )
                y_pred.append(label)

            PREDICTION_CACHE[xgb_model_key] = y_pred

        xgb_metrics = calculate_performance_metrics(
            model_name="XGBoost (Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time
        )
        results.append(xgb_metrics)
    else:
        results.append({
            "model_name": "XGBoost (Custom - Chưa huấn luyện)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0
        })

    xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
    xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")
    xgb_lib_model_key = "XGB_Library"

    if os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path):
        (
            model,
            feature_names,
            classes,
            threshold,
            train_time,
            params
        ) = load_xgb_library_model(xgb_lib_model_path)

        vectorizer = load_xgb_library_vectorizer(xgb_lib_vectorizer_path)

        if xgb_lib_model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[xgb_lib_model_key]
        else:
            y_pred = []
            for text in X_test:
                label, _ = predict_XGB_Library(
                    text_input=text,
                    model=model,
                    classes=classes,
                    threshold=threshold,
                    vectorizer=vectorizer
                )
                y_pred.append(label)

            PREDICTION_CACHE[xgb_lib_model_key] = y_pred

        xgb_lib_metrics = calculate_performance_metrics(
            model_name="XGBoost (Library)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time
        )
        results.append(xgb_lib_metrics)
    else:
        results.append({
            "model_name": "XGBoost (Library - Chưa huấn luyện)",
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
    X_test = df_test["text"].fillna("")

    y_pred = []

    # =========================================================
    # 1) XỬ LÝ MÔ HÌNH MNB CUSTOM
    # =========================================================
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

    # =========================================================
    # 2) XỬ LÝ MÔ HÌNH XGBOOST CUSTOM
    # =========================================================
    elif model_name == "XGBoost (Custom)":
        model_key = "XGB_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            xgb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_custom.pkl")
            xgb_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer.pkl")

            (
                base_score_logit,
                trees,
                learning_rate,
                feature_names,
                classes,
                threshold,
                _,
                params
            ) = load_xgb_model(xgb_model_path)
    # --- 2. Linear SVM (One-Sample Custom) ---
    elif model_name == "Linear SVM (One-Sample Custom)":
        model_key = "SVM_OneSample_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")
            svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
            X_test_tfidf = svm_one_vectorizer.transform(X_test).toarray()
            y_pred_svm = svm_one_model.predict(X_test_tfidf)
            y_pred = np.where(y_pred_svm == 1, 4, 0).tolist()
            PREDICTION_CACHE[model_key] = y_pred

    # --- 3. Linear SVM (Full-Sample Custom) ---
    elif model_name == "Linear SVM (Full-Sample Custom)":
        model_key = "SVM_FullSample_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
            svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
            X_test_tfidf = svm_full_vectorizer.transform(X_test).toarray()
            y_pred_svm = svm_full_model.predict(X_test_tfidf)
            y_pred = np.where(y_pred_svm == 1, 4, 0).tolist()
            PREDICTION_CACHE[model_key] = y_pred

    # --- 4. Linear SVM (Library) ---
    elif model_name == "Linear SVM (Library)":
        model_key = "SVM_Library"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
            svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")
            svm_lib_model = joblib.load(svm_lib_model_path)
            svm_lib_vectorizer = joblib.load(svm_lib_vectorizer_path)

            X_test_tfidf = svm_lib_vectorizer.transform(X_test)
            y_pred = svm_lib_model.predict(X_test_tfidf).tolist()
            PREDICTION_CACHE[model_key] = y_pred

    # --- 5. Multinomial Naive Bayes (Library) ---
    elif model_name == "Multinomial Naive Bayes (Library)":
        model_key = "MNB_Library"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            lib_pipeline, _ = load_library_model()
            if lib_pipeline:
                y_pred = lib_pipeline.predict(X_test)
                PREDICTION_CACHE[model_key] = y_pred

    # Nếu model_name không khớp thì trả mảng rỗng để tránh lỗi frontend
    if len(y_pred) == 0:
        return {"status": "success", "data": []}

    # Gom dữ liệu và lọc ra các câu dự đoán sai
    df_result = df_test.copy()
    df_result['predicted'] = y_pred
    df_errors = df_result[df_result['target'] != df_result['predicted']]
    result_data = df_errors[['target', 'predicted', 'text']].to_dict(orient="records")

            vectorizer = load_vectorizer(xgb_vectorizer_path)

            for text in X_test:
                label, _ = predict_XGB_Custom(
                    text_input=text,
                    base_score_logit=base_score_logit,
                    trees=trees,
                    learning_rate=learning_rate,
                    feature_names=feature_names,
                    classes=classes,
                    threshold=threshold,
                    vectorizer=vectorizer
                )
                y_pred.append(label)

            PREDICTION_CACHE[model_key] = y_pred

        PREDICTION_CACHE[model_key] = y_pred

    else:
        return {
            "status": "error",
            "message": f"Model '{model_name}' không được hỗ trợ."
        }

    df_result = df_test.copy()
    df_result["predicted"] = y_pred
    df_errors = df_result[df_result["target"] != df_result["predicted"]]
    result_data = df_errors[["target", "predicted", "text"]].to_dict(orient="records")



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
    # 1. Tiền xử lý văn bản
    cleaned_text = clean_text(text)

    # Khởi tạo dictionary chứa kết quả. (-1 nghĩa là chưa có mô hình)
    custom_preds = {"mnb": -1, "svm": -1, "xgb": -1}
    lib_preds = {"mnb": -1, "svm": -1, "xgb": -1}

    # ==========================================
    # 2A. NHÓM CUSTOM
    # ==========================================

    # --- 1. MNB Custom ---
    try:
        mnb_custom_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
        if os.path.exists(mnb_custom_path):
            priors, w_probs, vocab, totals, counts, _ = load_model(mnb_custom_path)
            label, _ = predict_MNB_Custom(cleaned_text, priors, w_probs, vocab, totals)
            custom_preds["mnb"] = label
    except Exception as e:
        print(f"Lỗi MNB Custom: {e}")

    # --- 2. SVM Full-Sample Custom ---
    try:
        svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
        if os.path.exists(svm_full_path):
            svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
            # Vector hóa văn bản đầu vào (.toarray() vì custom build dùng dense matrix)
            X_input_tfidf = svm_full_vectorizer.transform([cleaned_text]).toarray()
            # Dự đoán (trả về 1 hoặc -1)
            y_pred_svm = svm_full_model.predict(X_input_tfidf)
            # Map lại nhãn: 1 -> 4 (Tích cực), -1 -> 0 (Tiêu cực)
            label = int(np.where(y_pred_svm == 1, 4, 0)[0])
            custom_preds["svm"] = label
    except Exception as e:
        print(f"Lỗi SVM Custom: {e}")

    # ==========================================
    # 2B. NHÓM LIBRARY
    # ==========================================

    # --- 1. MNB Library ---
    try:
        from app.models.Multinomial_Library import load_library_model
        lib_pipeline, _ = load_library_model()
        if lib_pipeline:
            label = int(lib_pipeline.predict([cleaned_text])[0])
            lib_preds["mnb"] = label
    except Exception as e:
        print(f"Lỗi MNB Library: {e}")

    # --- 2. SVM Library ---
    try:
        svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
        svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")

        if os.path.exists(svm_lib_model_path) and os.path.exists(svm_lib_vectorizer_path):
            svm_lib_model = joblib.load(svm_lib_model_path)
            svm_lib_vectorizer = joblib.load(svm_lib_vectorizer_path)

            # Transform không cần toarray() vì sklearn LinearSVC tối ưu tốt cho ma trận thưa (sparse)
            X_input_tfidf = svm_lib_vectorizer.transform([cleaned_text])
            label = int(svm_lib_model.predict(X_input_tfidf)[0])
            lib_preds["svm"] = label
    except Exception as e:
        print(f"Lỗi SVM Library: {e}")

    # ==========================================
    # 3. CƠ CHẾ BỎ PHIẾU (Majority Vote)
    # ==========================================
    def get_vote(preds_dict):
        # Chỉ lấy các phiếu hợp lệ (khác -1)
        valid_preds = [v for v in preds_dict.values() if v != -1]
        if not valid_preds: return -1  # Chưa có mô hình nào chạy
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
