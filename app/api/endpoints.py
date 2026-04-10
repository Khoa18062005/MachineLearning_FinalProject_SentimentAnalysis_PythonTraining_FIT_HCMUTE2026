from fastapi import APIRouter
import os
import joblib
import numpy as np

from typing import Any
from starlette.responses import FileResponse

from app.core.config import settings
from app.services import ml_service
from app.services.evaluation_service import calculate_performance_metrics
from app.services.ml_service import get_clean_datasets_for_training, clean_text

from app.models.Multinomial_Custom import predict_MNB_Custom, load_model as load_mnb_custom_model
from app.models.Multinomial_Library import load_library_model

from app.models.SVM_OneSample_Custom import load_model as load_svm_one_model
from app.models.SVM_FullSample_Custom import load_model as load_svm_full_model

from app.models.XGBoost_Custom import (
    predict_XGB_Custom,
    load_model as load_xgb_model,
    load_vectorizer as load_xgb_vectorizer,
)

from app.models.XGBoost_Library import (
    predict_XGB_Library,
    load_model as load_xgb_library_model,
    load_vectorizer as load_xgb_library_vectorizer,
)

router = APIRouter()
PREDICTION_CACHE = {}


# =========================================================
# HELPER
# =========================================================
def _get_test_data():
    _, _, df_test = get_clean_datasets_for_training()
    X_test = df_test["text"].fillna("")
    y_test = df_test["target"]
    return df_test, X_test, y_test


def make_json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]

    return obj


def _get_majority_vote(preds_dict):
    valid_preds = [int(v) for v in preds_dict.values() if int(v) != -1]
    if not valid_preds:
        return -1
    return max(set(valid_preds), key=valid_preds.count)


def _ensure_int_labels(y_pred):
    return [int(x) for x in y_pred]


def _binary01_to_04(y_pred_binary):
    return [4 if int(v) == 1 else 0 for v in y_pred_binary]


def _predict_mnb_custom_batch(X_texts):
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
    if not os.path.exists(model_path):
        return []

    priors, w_probs, vocab, totals, counts, _ = load_mnb_custom_model(model_path)
    y_pred = []
    for text in X_texts:
        label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
        y_pred.append(int(label))
    return y_pred


def _predict_svm_custom_dense_in_batches(model, vectorizer, X_texts, batch_size=256):
    y_pred_all = []

    total = len(X_texts)
    for start in range(0, total, batch_size):
        batch_texts = X_texts.iloc[start:start + batch_size]
        X_batch = vectorizer.transform(batch_texts).astype(np.float32).toarray()
        y_batch = model.predict(X_batch)
        y_pred_all.extend(_binary01_to_04(y_batch))

    return y_pred_all


def _predict_svm_one_custom_batch(X_texts):
    svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")
    if not os.path.exists(svm_one_path):
        return []

    svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
    return _predict_svm_custom_dense_in_batches(
        model=svm_one_model,
        vectorizer=svm_one_vectorizer,
        X_texts=X_texts,
        batch_size=256,
    )


def _predict_svm_full_custom_batch(X_texts):
    svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
    if not os.path.exists(svm_full_path):
        return []

    svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
    return _predict_svm_custom_dense_in_batches(
        model=svm_full_model,
        vectorizer=svm_full_vectorizer,
        X_texts=X_texts,
        batch_size=128,
    )


def _predict_svm_library_batch(X_texts):
    svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
    svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")

    if not (os.path.exists(svm_lib_model_path) and os.path.exists(svm_lib_vectorizer_path)):
        return []

    svm_lib_model = joblib.load(svm_lib_model_path)
    svm_lib_vectorizer = joblib.load(svm_lib_vectorizer_path)
    X_test_tfidf = svm_lib_vectorizer.transform(X_texts)
    y_pred = svm_lib_model.predict(X_test_tfidf).tolist()
    return _ensure_int_labels(y_pred)


def _predict_mnb_library_batch(X_texts):
    lib_pipeline, _ = load_library_model()
    if not lib_pipeline:
        return []

    y_pred = lib_pipeline.predict(X_texts).tolist()
    return _ensure_int_labels(y_pred)


def _predict_xgb_custom_batch(X_texts):
    xgb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_custom.pkl")
    xgb_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer.pkl")

    if not (os.path.exists(xgb_model_path) and os.path.exists(xgb_vectorizer_path)):
        return []

    (
        base_score_logit,
        trees,
        learning_rate,
        feature_names,
        classes,
        threshold,
        _train_time,
        _params,
    ) = load_xgb_model(xgb_model_path)

    vectorizer = load_xgb_vectorizer(xgb_vectorizer_path)

    y_pred = []
    for text in X_texts:
        label, _ = predict_XGB_Custom(
            text_input=text,
            base_score_logit=base_score_logit,
            trees=trees,
            learning_rate=learning_rate,
            feature_names=feature_names,
            classes=classes,
            threshold=threshold,
            vectorizer=vectorizer,
        )
        y_pred.append(int(label))

    return y_pred


def _predict_xgb_library_batch(X_texts):
    xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
    xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")

    if not (os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path)):
        return []

    (
        model,
        _feature_names,
        classes,
        threshold,
        _train_time,
        _params,
    ) = load_xgb_library_model(xgb_lib_model_path)

    vectorizer = load_xgb_library_vectorizer(xgb_lib_vectorizer_path)

    y_pred = []
    for text in X_texts:
        label, _ = predict_XGB_Library(
            text_input=text,
            model=model,
            classes=classes,
            threshold=threshold,
            vectorizer=vectorizer,
        )
        y_pred.append(int(label))

    return y_pred


def _single_predict_svm_custom(cleaned_text: str):
    svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
    svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")

    if os.path.exists(svm_full_path):
        svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
        X_input_tfidf = svm_full_vectorizer.transform([cleaned_text]).astype(np.float32).toarray()
        y_pred_svm = svm_full_model.predict(X_input_tfidf)
        return int(4 if int(y_pred_svm[0]) == 1 else 0)

    if os.path.exists(svm_one_path):
        svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
        X_input_tfidf = svm_one_vectorizer.transform([cleaned_text]).astype(np.float32).toarray()
        y_pred_svm = svm_one_model.predict(X_input_tfidf)
        return int(4 if int(y_pred_svm[0]) == 1 else 0)

    return -1


def _append_placeholder(results, model_name):
    results.append({
        "model_name": model_name,
        "training_time_sec": 0,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "accuracy": 0,
    })


def _safe_metrics(model_name, y_true, y_pred, training_time_sec):
    metrics = calculate_performance_metrics(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        training_time_sec=training_time_sec,
    )
    return make_json_safe(metrics)


def _get_chart_candidates(model_type: str):
    charts_map = {
        "mnb": ["laplace_smoothing_study.png"],
        "mnb_custom_cm": ["cm_mnb_custom.png"],
        "mnb_library_cm": ["cm_mnb_library.png"],
        "svm_cm": ["cm_svm.png"],
        "svm_custom_cm": ["cm_svm.png"],
        "xgb_cm": ["cm_xgb.png", "cm_xgb_custom.png"],
        "xgb_custom_cm": ["cm_xgb_custom.png", "cm_xgb.png"],
        "xgb_library_cm": ["cm_xgb_library.png", "cm_xgb.png"],
        "svm": ["svm_study.png"],
        "xgb": ["xgb_study.png"],
    }
    return charts_map.get(model_type, [])


# =========================================================
# BASIC DATA ROUTES
# =========================================================
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
        "counts": result["counts"],
    }


@router.get("/clean-dataset/{set_name}")
async def get_clean_dataset(set_name: str):
    result = ml_service.get_clean_dataset_by_name(set_name=set_name, limit=200000)
    return {
        "status": "success",
        "data": result["records"],
        "counts": result["counts"],
    }


# =========================================================
# TRAINING RESULTS
# =========================================================
@router.get("/train-models")
async def get_training_results():
    results = []
    _, X_test, y_test = _get_test_data()

    # 1) MNB CUSTOM
    try:
        model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
        if os.path.exists(model_path):
            priors, w_probs, vocab, totals, counts, train_time = load_mnb_custom_model(model_path)
            model_key = "MNB_Custom"

            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = []
                for text in X_test:
                    label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
                    y_pred.append(int(label))
                PREDICTION_CACHE[model_key] = y_pred

            results.append(_safe_metrics(
                model_name="Multinomial Naive Bayes (Custom)",
                y_true=y_test,
                y_pred=y_pred,
                training_time_sec=train_time,
            ))
        else:
            _append_placeholder(results, "Multinomial Naive Bayes (Custom - Chưa huấn luyện)")
    except Exception as e:
        print(f"[train-models] Lỗi MNB Custom: {e}")
        _append_placeholder(results, "Multinomial Naive Bayes (Custom - Lỗi tải mô hình)")

    # 2) SVM ONE-SAMPLE CUSTOM
    try:
        svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")
        if os.path.exists(svm_one_path):
            svm_one_model, _svm_one_vectorizer = load_svm_one_model(svm_one_path)
            model_key = "SVM_OneSample_Custom"

            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_one_custom_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

            results.append(_safe_metrics(
                model_name="Linear SVM (One-Sample Custom)",
                y_true=y_test,
                y_pred=y_pred,
                training_time_sec=getattr(svm_one_model, "training_time_sec", 0.0),
            ))
        else:
            _append_placeholder(results, "Linear SVM (One-Sample Custom - Chưa huấn luyện)")
    except Exception as e:
        print(f"[train-models] Lỗi SVM One-Sample Custom: {e}")
        _append_placeholder(results, "Linear SVM (One-Sample Custom - Lỗi tải mô hình)")

    # 3) SVM FULL-SAMPLE CUSTOM
    try:
        svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
        if os.path.exists(svm_full_path):
            svm_full_model, _svm_full_vectorizer = load_svm_full_model(svm_full_path)
            model_key = "SVM_FullSample_Custom"

            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_full_custom_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

            results.append(_safe_metrics(
                model_name="Linear SVM (Full-Sample Custom)",
                y_true=y_test,
                y_pred=y_pred,
                training_time_sec=getattr(svm_full_model, "training_time_sec", 0.0),
            ))
        else:
            _append_placeholder(results, "Linear SVM (Full-Sample Custom - Chưa huấn luyện)")
    except Exception as e:
        print(f"[train-models] Lỗi SVM Full-Sample Custom: {e}")
        _append_placeholder(results, "Linear SVM (Full-Sample Custom - Lỗi tải mô hình)")

    # 4) SVM LIBRARY
    try:
        svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
        svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")
        if os.path.exists(svm_lib_model_path) and os.path.exists(svm_lib_vectorizer_path):
            svm_lib_model = joblib.load(svm_lib_model_path)
            model_key = "SVM_Library"

            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_library_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

            results.append(_safe_metrics(
                model_name="Linear SVM (Library)",
                y_true=y_test,
                y_pred=y_pred,
                training_time_sec=getattr(svm_lib_model, "training_time_sec", 0.0),
            ))
        else:
            _append_placeholder(results, "Linear SVM (Library - Chưa huấn luyện)")
    except Exception as e:
        print(f"[train-models] Lỗi SVM Library: {e}")
        _append_placeholder(results, "Linear SVM (Library - Lỗi tải mô hình)")

    # 5) MNB LIBRARY
    try:
        lib_pipeline, lib_time = load_library_model()
        model_key = "MNB_Library"

        if lib_pipeline:
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_mnb_library_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

            results.append(_safe_metrics(
                model_name="Multinomial Naive Bayes (Library)",
                y_true=y_test,
                y_pred=y_pred,
                training_time_sec=lib_time,
            ))
        else:
            _append_placeholder(results, "Multinomial Naive Bayes (Library - Chưa huấn luyện)")
    except Exception as e:
        print(f"[train-models] Lỗi MNB Library: {e}")
        _append_placeholder(results, "Multinomial Naive Bayes (Library - Lỗi tải mô hình)")

    # 6) XGBOOST CUSTOM
    try:
        xgb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_custom.pkl")
        xgb_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer.pkl")

        if os.path.exists(xgb_model_path) and os.path.exists(xgb_vectorizer_path):
            (
                _base_score_logit,
                _trees,
                _learning_rate,
                _feature_names,
                _classes,
                _threshold,
                train_time,
                _params,
            ) = load_xgb_model(xgb_model_path)

            model_key = "XGB_Custom"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_xgb_custom_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

            results.append(_safe_metrics(
                model_name="XGBoost (Custom)",
                y_true=y_test,
                y_pred=y_pred,
                training_time_sec=train_time,
            ))
        else:
            _append_placeholder(results, "XGBoost (Custom - Chưa huấn luyện)")
    except Exception as e:
        print(f"[train-models] Lỗi XGBoost Custom: {e}")
        _append_placeholder(results, "XGBoost (Custom - Lỗi tải mô hình)")

    # 7) XGBOOST LIBRARY
    try:
        xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
        xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")

        if os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path):
            (
                _model,
                _feature_names,
                _classes,
                _threshold,
                train_time,
                _params,
            ) = load_xgb_library_model(xgb_lib_model_path)

            model_key = "XGB_Library"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_xgb_library_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

            results.append(_safe_metrics(
                model_name="XGBoost (Library)",
                y_true=y_test,
                y_pred=y_pred,
                training_time_sec=train_time,
            ))
        else:
            _append_placeholder(results, "XGBoost (Library - Chưa huấn luyện)")
    except Exception as e:
        print(f"[train-models] Lỗi XGBoost Library: {e}")
        _append_placeholder(results, "XGBoost (Library - Lỗi tải mô hình)")

    return {"status": "success", "data": make_json_safe(results)}


# =========================================================
# MODEL ERRORS
# =========================================================
@router.get("/model-errors")
async def get_model_errors(model_name: str):
    df_test, X_test, _ = _get_test_data()
    y_pred = []

    try:
        if model_name == "Multinomial Naive Bayes (Custom)":
            model_key = "MNB_Custom"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_mnb_custom_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

        elif model_name == "Linear SVM (One-Sample Custom)":
            model_key = "SVM_OneSample_Custom"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_one_custom_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

        elif model_name == "Linear SVM (Full-Sample Custom)":
            model_key = "SVM_FullSample_Custom"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_full_custom_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

        elif model_name == "Linear SVM (Library)":
            model_key = "SVM_Library"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_library_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

        elif model_name == "Multinomial Naive Bayes (Library)":
            model_key = "MNB_Library"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_mnb_library_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

        elif model_name == "XGBoost (Custom)":
            model_key = "XGB_Custom"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_xgb_custom_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

        elif model_name == "XGBoost (Library)":
            model_key = "XGB_Library"
            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_xgb_library_batch(X_test)
                PREDICTION_CACHE[model_key] = y_pred

        else:
            return {
                "status": "error",
                "message": f"Model '{model_name}' không được hỗ trợ.",
            }

        if len(y_pred) == 0:
            return {"status": "success", "data": []}

        df_result = df_test.copy()
        df_result["predicted"] = y_pred
        df_errors = df_result[df_result["target"] != df_result["predicted"]]
        result_data = df_errors[["target", "predicted", "text"]].to_dict(orient="records")

        return {"status": "success", "data": make_json_safe(result_data)}

    except Exception as e:
        print(f"[model-errors] Lỗi với model '{model_name}': {e}")
        return {
            "status": "error",
            "message": f"Không thể lấy danh sách lỗi cho model '{model_name}'.",
        }


# =========================================================
# MODEL DETAILS
# =========================================================
@router.get("/model-details")
async def get_model_prediction_details(text: str, model_name: str = "Multinomial Naive Bayes (Custom)"):
    try:
        # =====================================================
        # MNB CUSTOM
        # =====================================================
        if model_name == "Multinomial Naive Bayes (Custom)":
            model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
            if not os.path.exists(model_path):
                return {"status": "error", "message": "MNB Custom model not found."}

            priors, w_probs, vocab, totals, counts, _ = load_mnb_custom_model(model_path)
            from app.models.Multinomial_Custom import get_prediction_details

            details = get_prediction_details(text, priors, w_probs, vocab, totals, counts)
            details = make_json_safe(details)
            details["model_type"] = "mnb_custom"
            details["input_text"] = text

            return {"status": "success", "data": details}

        # =====================================================
        # XGBOOST CUSTOM
        # =====================================================
        elif model_name == "XGBoost (Custom)":
            from app.models import XGBoost_Custom as xgb_custom_module

            detail_fn = getattr(xgb_custom_module, "get_prediction_details_XGB_Custom", None)
            if detail_fn is None:
                return {
                    "status": "error",
                    "message": "File XGBoost_Custom.py chưa có hàm get_prediction_details_XGB_Custom."
                }

            xgb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_custom.pkl")
            xgb_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer.pkl")

            if not (os.path.exists(xgb_model_path) and os.path.exists(xgb_vectorizer_path)):
                return {
                    "status": "error",
                    "message": "XGBoost Custom model/vectorizer not found."
                }

            (
                base_score_logit,
                trees,
                learning_rate,
                feature_names,
                classes,
                threshold,
                train_time,
                params,
            ) = load_xgb_model(xgb_model_path)

            vectorizer = load_xgb_vectorizer(xgb_vectorizer_path)

            model_data = {
                "base_score_logit": base_score_logit,
                "trees": trees,
                "learning_rate": learning_rate,
                "feature_names": feature_names,
                "classes": classes,
                "threshold": threshold,
                "training_time_sec": train_time,
                "params": params,
            }

            details = detail_fn(
                text_input=text,
                model_data=model_data,
                vectorizer=vectorizer,
            )

            details = make_json_safe(details)
            details["model_type"] = "xgb_custom"
            return {"status": "success", "data": details}

        # =====================================================
        # XGBOOST LIBRARY
        # =====================================================
        elif model_name == "XGBoost (Library)":
            from app.models import XGBoost_Library as xgb_library_module

            detail_fn = getattr(xgb_library_module, "get_prediction_details_XGB_Library", None)
            if detail_fn is None:
                return {
                    "status": "error",
                    "message": "File XGBoost_Library.py chưa có hàm get_prediction_details_XGB_Library."
                }

            xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
            xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")

            if not (os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path)):
                return {
                    "status": "error",
                    "message": "XGBoost Library model/vectorizer not found."
                }

            (
                model,
                feature_names,
                classes,
                threshold,
                train_time,
                params,
            ) = load_xgb_library_model(xgb_lib_model_path)

            vectorizer = load_xgb_library_vectorizer(xgb_lib_vectorizer_path)

            model_data = {
                "model": model,
                "feature_names": feature_names,
                "classes": classes,
                "threshold": threshold,
                "training_time_sec": train_time,
                "params": params,
                "best_params": params.get("best_params") if isinstance(params, dict) else None,
                "best_score_cv": params.get("best_score_cv") if isinstance(params, dict) else None,
            }

            details = detail_fn(
                text_input=text,
                model_data=model_data,
                vectorizer=vectorizer,
            )

            details = make_json_safe(details)
            details["model_type"] = "xgb_library"
            return {"status": "success", "data": details}

        # =====================================================
        # SVM / MNB LIBRARY
        # =====================================================
        elif "SVM" in model_name:
            return {
                "status": "error",
                "message": f"Model '{model_name}' hiện chưa hỗ trợ giao diện giải thích chi tiết."
            }

        elif model_name == "Multinomial Naive Bayes (Library)":
            return {
                "status": "error",
                "message": "MNB Library hiện chưa có hàm giải thích chi tiết riêng trong giao diện này."
            }

        else:
            return {
                "status": "error",
                "message": f"Model '{model_name}' không được hỗ trợ."
            }

    except Exception as e:
        print(f"[model-details] Lỗi với model '{model_name}': {e}")
        return {
            "status": "error",
            "message": f"Không thể lấy chi tiết dự đoán cho model '{model_name}'."
        }


# =========================================================
# CHARTS
# =========================================================
@router.get("/charts/{model_type}")
async def get_model_charts(model_type: str):
    candidates = _get_chart_candidates(model_type)

    if not candidates:
        return {"status": "error", "message": "Loại biểu đồ không hợp lệ!"}

    for file_name in candidates:
        chart_path = os.path.join(settings.BASE_DIR, file_name)
        if os.path.exists(chart_path):
            return FileResponse(chart_path)

    return {"status": "error", "message": f"Biểu đồ {model_type} chưa được tạo!"}


# =========================================================
# PREDICT TEXT
# =========================================================
@router.get("/predict-text")
async def predict_new_text(text: str):
    cleaned_text = clean_text(text)

    custom_preds = {"mnb": -1, "svm": -1, "xgb": -1}
    lib_preds = {"mnb": -1, "svm": -1, "xgb": -1}

    # CUSTOM - MNB
    try:
        mnb_custom_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
        if os.path.exists(mnb_custom_path):
            priors, w_probs, vocab, totals, counts, _ = load_mnb_custom_model(mnb_custom_path)
            label, _ = predict_MNB_Custom(cleaned_text, priors, w_probs, vocab, totals)
            custom_preds["mnb"] = int(label)
    except Exception as e:
        print(f"Lỗi MNB Custom: {e}")

    # CUSTOM - SVM
    try:
        custom_preds["svm"] = _single_predict_svm_custom(cleaned_text)
    except Exception as e:
        print(f"Lỗi SVM Custom: {e}")

    # CUSTOM - XGB
    try:
        xgb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_custom.pkl")
        xgb_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer.pkl")

        if os.path.exists(xgb_model_path) and os.path.exists(xgb_vectorizer_path):
            (
                base_score_logit,
                trees,
                learning_rate,
                feature_names,
                classes,
                threshold,
                _train_time,
                _params,
            ) = load_xgb_model(xgb_model_path)

            vectorizer = load_xgb_vectorizer(xgb_vectorizer_path)

            label, _ = predict_XGB_Custom(
                text_input=cleaned_text,
                base_score_logit=base_score_logit,
                trees=trees,
                learning_rate=learning_rate,
                feature_names=feature_names,
                classes=classes,
                threshold=threshold,
                vectorizer=vectorizer,
            )
            custom_preds["xgb"] = int(label)
    except Exception as e:
        print(f"Lỗi XGBoost Custom: {e}")

    # LIBRARY - MNB
    try:
        lib_pipeline, _ = load_library_model()
        if lib_pipeline:
            label = int(lib_pipeline.predict([cleaned_text])[0])
            lib_preds["mnb"] = label
    except Exception as e:
        print(f"Lỗi MNB Library: {e}")

    # LIBRARY - SVM
    try:
        svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
        svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")

        if os.path.exists(svm_lib_model_path) and os.path.exists(svm_lib_vectorizer_path):
            svm_lib_model = joblib.load(svm_lib_model_path)
            svm_lib_vectorizer = joblib.load(svm_lib_vectorizer_path)

            X_input_tfidf = svm_lib_vectorizer.transform([cleaned_text])
            label = int(svm_lib_model.predict(X_input_tfidf)[0])
            lib_preds["svm"] = label
    except Exception as e:
        print(f"Lỗi SVM Library: {e}")

    # LIBRARY - XGB
    try:
        xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
        xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")

        if os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path):
            (
                model,
                _feature_names,
                classes,
                threshold,
                _train_time,
                _params,
            ) = load_xgb_library_model(xgb_lib_model_path)

            vectorizer = load_xgb_library_vectorizer(xgb_lib_vectorizer_path)

            label, _ = predict_XGB_Library(
                text_input=cleaned_text,
                model=model,
                classes=classes,
                threshold=threshold,
                vectorizer=vectorizer,
            )
            lib_preds["xgb"] = int(label)
    except Exception as e:
        print(f"Lỗi XGBoost Library: {e}")

    custom_vote = _get_majority_vote(custom_preds)
    lib_vote = _get_majority_vote(lib_preds)
    final_vote = lib_vote if lib_vote != -1 else custom_vote

    return {
        "status": "success",
        "data": make_json_safe({
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "custom": custom_preds,
            "library": lib_preds,
            "votes": {
                "custom": custom_vote,
                "library": lib_vote,
                "final": final_vote,
            },
        }),
    }