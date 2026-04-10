from fastapi import APIRouter
import os
import joblib
import numpy as np

from sklearn.pipeline import FeatureUnion
from starlette.responses import FileResponse

from app.core.config import settings
from app.services import ml_service
from app.services.evaluation_service import (
    calculate_performance_metrics,
    save_confusion_matrix_chart,
)
from app.services.ml_service import get_clean_datasets_for_training, clean_text

from app.models.Multinomial_Custom import (
    predict_MNB_Custom,
    load_model,
    get_prediction_details as get_mnb_custom_prediction_details,
)
from app.models.Multinomial_Library import load_library_model

from app.models.SVM_OneSample_Custom import load_model as load_svm_one_model
from app.models.SVM_FullSample_Custom import load_model as load_svm_full_model

from app.models.XGBoost_Custom import (
    predict_XGB_Custom,
    load_model as load_xgb_model,
    load_vectorizer,
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


def _predict_svm_project_labels(model, vectorizer, X_texts):
    """
    Dự đoán cho SVM custom:
    - Giữ nguyên sparse matrix, KHÔNG .toarray()
    - Output project labels: 0 / 4
    """
    X_tfidf = vectorizer.transform(X_texts)
    y_pred_svm = model.predict(X_tfidf)
    return np.where(y_pred_svm == 1, 4, 0).tolist()


def _predict_single_svm_project_label(model, vectorizer, text):
    """
    Dự đoán 1 câu cho SVM custom:
    - Giữ sparse
    - Output project label: 0 / 4
    """
    X_tfidf = vectorizer.transform([text])
    y_pred_svm = model.predict(X_tfidf)
    return int(np.where(y_pred_svm == 1, 4, 0)[0])


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

    vectorizer = load_vectorizer(xgb_vectorizer_path)

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
        y_pred.append(label)

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
        y_pred.append(label)

    return y_pred


def _get_majority_vote(preds_dict):
    valid_preds = [v for v in preds_dict.values() if v != -1]
    if not valid_preds:
        return -1
    return max(set(valid_preds), key=valid_preds.count)


def _safe_sigmoid(x):
    x = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(scores):
    scores = np.array(scores, dtype=float)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    total = np.sum(exp_scores)
    if total == 0:
        return np.ones_like(exp_scores) / len(exp_scores)
    return exp_scores / total


def _get_vectorizer_feature_names(vectorizer):
    """
    Hỗ trợ cả:
    - TfidfVectorizer / CountVectorizer
    - FeatureUnion(word + char)
    """
    if hasattr(vectorizer, "get_feature_names_out"):
        return list(vectorizer.get_feature_names_out())

    if isinstance(vectorizer, FeatureUnion) or hasattr(vectorizer, "transformer_list"):
        feature_names = []
        for transformer_name, transformer in vectorizer.transformer_list:
            if hasattr(transformer, "get_feature_names_out"):
                child_names = transformer.get_feature_names_out()
                child_names = [f"{transformer_name}:{name}" for name in child_names]
                feature_names.extend(child_names)
        return feature_names

    return []


def _build_linear_svm_detail_payload(model_name, text, actual_target, predicted_target, model, vectorizer, source="custom"):
    """
    Giải thích tuyến tính cho:
    - SVM One-Sample Custom
    - SVM Full-Sample Custom
    - SVM Library
    """
    X_vec = vectorizer.transform([text])

    if source == "library":
        weights = model.coef_.ravel()
        bias = float(model.intercept_[0]) if hasattr(model, "intercept_") else 0.0
        raw_score = float(model.decision_function(X_vec)[0])
    else:
        weights = model.w
        bias = float(model.b)
        raw_score = float(model.decision_function(X_vec)[0])

    feature_names = _get_vectorizer_feature_names(vectorizer)

    rows = []
    if hasattr(X_vec, "indices"):
        for idx, tfidf_value in zip(X_vec.indices, X_vec.data):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            weight = float(weights[idx])
            contribution = float(tfidf_value * weight)

            if contribution > 0:
                direction = "Tích cực (đẩy về label 4)"
            elif contribution < 0:
                direction = "Tiêu cực (đẩy về label 0)"
            else:
                direction = "Trung tính"

            rows.append({
                "feature": feature_name,
                "tfidf": float(tfidf_value),
                "weight": weight,
                "contribution": contribution,
                "direction": direction
            })

    rows = sorted(rows, key=lambda x: abs(x["contribution"]), reverse=True)

    prob_pos = _safe_sigmoid(raw_score)
    prob_neg = 1.0 - prob_pos

    return {
        "detail_type": "svm_linear",
        "model_name": model_name,
        "raw_text": text,
        "actual_target": actual_target,
        "predicted_target": predicted_target,
        "decision_score": raw_score,
        "bias": bias,
        "pseudo_prob_neg": prob_neg,
        "pseudo_prob_pos": prob_pos,
        "active_features_count": len(rows),
        "feature_steps": rows[:40]
    }


def _build_mnb_library_detail_payload(model_name, text, actual_target, predicted_target):
    lib_pipeline, _ = load_library_model()
    if not lib_pipeline:
        return {
            "detail_type": "unsupported",
            "model_name": model_name,
            "message": "Chưa tìm thấy model Multinomial Naive Bayes Library."
        }

    vectorizer = lib_pipeline.named_steps["vectorizer"]
    nb_model = lib_pipeline.named_steps["nb"]

    X_counts = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    classes = nb_model.classes_

    class_log_priors = nb_model.class_log_prior_
    feature_log_prob = nb_model.feature_log_prob_

    raw_log_scores = []
    detail_by_class = {}

    active_rows = []
    if hasattr(X_counts, "indices"):
        for idx, count in zip(X_counts.indices, X_counts.data):
            active_rows.append((idx, int(count)))

    for class_idx, class_value in enumerate(classes):
        class_rows = []
        total_log_score = float(class_log_priors[class_idx])

        for idx, count in active_rows:
            token = feature_names[idx]
            log_prob = float(feature_log_prob[class_idx, idx])
            prob = float(np.exp(log_prob))
            step_log_score = float(count * log_prob)
            total_log_score += step_log_score

            class_rows.append({
                "word": token,
                "count": count,
                "log_prob": log_prob,
                "prob": prob,
                "step_log_score": step_log_score
            })

        raw_log_scores.append(total_log_score)
        detail_by_class[str(class_value)] = {
            "prior": float(np.exp(class_log_priors[class_idx])),
            "log_prior": float(class_log_priors[class_idx]),
            "word_steps": class_rows,
            "final_log_score": total_log_score
        }

    normalized_probs = _softmax(raw_log_scores)
    for idx, class_value in enumerate(classes):
        detail_by_class[str(class_value)]["normalized_prob"] = float(normalized_probs[idx])

    return {
        "detail_type": "mnb_library",
        "model_name": model_name,
        "raw_text": text,
        "actual_target": actual_target,
        "predicted_target": predicted_target,
        "classes": detail_by_class
    }


def _build_xgb_generic_detail_payload(model_name, text, actual_target, predicted_target, model_type):
    """
    Fallback cho XGBoost:
    - Không giải từng cây / từng path
    - Nhưng vẫn trả ra chi tiết đủ để modal không bị hỏng
    """
    if model_type == "library":
        xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
        xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")

        if not (os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path)):
            return {
                "detail_type": "unsupported",
                "model_name": model_name,
                "message": "Chưa tìm thấy model XGBoost Library."
            }

        model, _feature_names, classes, threshold, _train_time, _params = load_xgb_library_model(xgb_lib_model_path)
        vectorizer = load_xgb_library_vectorizer(xgb_lib_vectorizer_path)

        X_vec = vectorizer.transform([text])

        prob_neg = None
        prob_pos = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_vec)[0]
            if len(probs) >= 2:
                prob_neg = float(probs[0])
                prob_pos = float(probs[1])

        feature_names = _get_vectorizer_feature_names(vectorizer)
        feature_importances = getattr(model, "feature_importances_", None)

        rows = []
        if hasattr(X_vec, "indices"):
            for idx, val in zip(X_vec.indices, X_vec.data):
                feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                importance = float(feature_importances[idx]) if feature_importances is not None and idx < len(feature_importances) else 0.0
                proxy_contribution = float(val * importance)

                rows.append({
                    "feature": feature_name,
                    "value": float(val),
                    "importance": importance,
                    "proxy_contribution": proxy_contribution
                })

        rows = sorted(rows, key=lambda x: abs(x["proxy_contribution"]), reverse=True)

        return {
            "detail_type": "xgb_generic",
            "model_name": model_name,
            "raw_text": text,
            "actual_target": actual_target,
            "predicted_target": predicted_target,
            "note": "Đây là phân tích gần đúng theo đặc trưng kích hoạt và feature importance, chưa phải giải thích chi tiết từng nhánh cây.",
            "prob_neg": prob_neg,
            "prob_pos": prob_pos,
            "feature_steps": rows[:40]
        }

    return {
        "detail_type": "unsupported",
        "model_name": model_name,
        "message": "XGBoost Custom hiện chưa hỗ trợ bóc chi tiết từng bước trong file bạn gửi."
    }


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

        save_confusion_matrix_chart(y_test, y_pred, "mnb_custom")

        mnb_metrics = calculate_performance_metrics(
            model_name="Multinomial Naive Bayes (Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time,
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
            y_pred = _predict_svm_project_labels(svm_one_model, svm_one_vectorizer, X_test)
            PREDICTION_CACHE[svm_one_key] = y_pred

        save_confusion_matrix_chart(y_test, y_pred, "svm_one_custom")

        svm_one_metrics = calculate_performance_metrics(
            model_name="Linear SVM (One-Sample Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=svm_one_model.training_time_sec,
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
            y_pred = _predict_svm_project_labels(svm_full_model, svm_full_vectorizer, X_test)
            PREDICTION_CACHE[svm_full_key] = y_pred

        save_confusion_matrix_chart(y_test, y_pred, "svm_full_custom")

        svm_full_metrics = calculate_performance_metrics(
            model_name="Linear SVM (Full-Sample Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=svm_full_model.training_time_sec,
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

        save_confusion_matrix_chart(y_test, y_pred, "svm_library")

        svm_lib_metrics = calculate_performance_metrics(
            model_name="Linear SVM (Library)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=getattr(svm_lib_model, "training_time_sec", 0.0),
        )
        results.append(svm_lib_metrics)

    # =========================
    # 5) MNB LIBRARY
    # =========================
    lib_pipeline, lib_time = load_library_model()
    lib_key = "MNB_Library"

    if lib_pipeline:
        if lib_key in PREDICTION_CACHE:
            y_pred_lib = PREDICTION_CACHE[lib_key]
        else:
            y_pred_lib = lib_pipeline.predict(X_test).tolist()
            PREDICTION_CACHE[lib_key] = y_pred_lib

        save_confusion_matrix_chart(y_test, y_pred_lib, "mnb_library")

        lib_metrics = calculate_performance_metrics(
            model_name="Multinomial Naive Bayes (Library)",
            y_true=y_test,
            y_pred=y_pred_lib,
            training_time_sec=lib_time,
        )
        results.append(lib_metrics)
    else:
        results.append({
            "model_name": "Multinomial NB Library (Chưa huấn luyện)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0,
        })

    # =========================
    # 6) XGBOOST CUSTOM
    # =========================
    xgb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_custom.pkl")
    xgb_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer.pkl")
    xgb_model_key = "XGB_Custom"

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

        if xgb_model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[xgb_model_key]
        else:
            y_pred = _predict_xgb_custom_batch(X_test)
            PREDICTION_CACHE[xgb_model_key] = y_pred

        save_confusion_matrix_chart(y_test, y_pred, "xgb_custom")

        xgb_metrics = calculate_performance_metrics(
            model_name="XGBoost (Custom)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time,
        )
        results.append(xgb_metrics)
    else:
        results.append({
            "model_name": "XGBoost (Custom - Chưa huấn luyện)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0,
        })

    # =========================
    # 7) XGBOOST LIBRARY
    # =========================
    xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
    xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")
    xgb_lib_model_key = "XGB_Library"

    if os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path):
        (
            _model,
            _feature_names,
            _classes,
            _threshold,
            train_time,
            _params,
        ) = load_xgb_library_model(xgb_lib_model_path)

        if xgb_lib_model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[xgb_lib_model_key]
        else:
            y_pred = _predict_xgb_library_batch(X_test)
            PREDICTION_CACHE[xgb_lib_model_key] = y_pred

        save_confusion_matrix_chart(y_test, y_pred, "xgb_library")

        xgb_lib_metrics = calculate_performance_metrics(
            model_name="XGBoost (Library)",
            y_true=y_test,
            y_pred=y_pred,
            training_time_sec=train_time,
        )
        results.append(xgb_lib_metrics)
    else:
        results.append({
            "model_name": "XGBoost (Library - Chưa huấn luyện)",
            "training_time_sec": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0,
        })

    return {"status": "success", "data": results}


# =========================================================
# MODEL ERRORS
# =========================================================
@router.get("/model-errors")
async def get_model_errors(model_name: str):
    df_test, X_test, _ = _get_test_data()
    y_pred = []

    if model_name == "Multinomial Naive Bayes (Custom)":
        model_key = "MNB_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
            if os.path.exists(model_path):
                priors, w_probs, vocab, totals, counts, _ = load_model(model_path)
                for text in X_test:
                    label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
                    y_pred.append(label)
                PREDICTION_CACHE[model_key] = y_pred

    elif model_name == "Linear SVM (One-Sample Custom)":
        model_key = "SVM_OneSample_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")
            if os.path.exists(svm_one_path):
                svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
                y_pred = _predict_svm_project_labels(svm_one_model, svm_one_vectorizer, X_test)
                PREDICTION_CACHE[model_key] = y_pred

    elif model_name == "Linear SVM (Full-Sample Custom)":
        model_key = "SVM_FullSample_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
            if os.path.exists(svm_full_path):
                svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
                y_pred = _predict_svm_project_labels(svm_full_model, svm_full_vectorizer, X_test)
                PREDICTION_CACHE[model_key] = y_pred

    elif model_name == "Linear SVM (Library)":
        model_key = "SVM_Library"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
            svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")

            if os.path.exists(svm_lib_model_path) and os.path.exists(svm_lib_vectorizer_path):
                svm_lib_model = joblib.load(svm_lib_model_path)
                svm_lib_vectorizer = joblib.load(svm_lib_vectorizer_path)
                X_test_tfidf = svm_lib_vectorizer.transform(X_test)
                y_pred = svm_lib_model.predict(X_test_tfidf).tolist()
                PREDICTION_CACHE[model_key] = y_pred

    elif model_name == "Multinomial Naive Bayes (Library)":
        model_key = "MNB_Library"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            lib_pipeline, _ = load_library_model()
            if lib_pipeline:
                y_pred = lib_pipeline.predict(X_test).tolist()
                PREDICTION_CACHE[model_key] = y_pred

    elif model_name == "XGBoost (Custom)":
        model_key = "XGB_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            y_pred = _predict_xgb_custom_batch(X_test)
            if y_pred:
                PREDICTION_CACHE[model_key] = y_pred

    elif model_name == "XGBoost (Library)":
        model_key = "XGB_Library"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            y_pred = _predict_xgb_library_batch(X_test)
            if y_pred:
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

    return {"status": "success", "data": result_data}


# =========================================================
# MODEL DETAILS
# =========================================================
@router.get("/model-details")
async def get_model_prediction_details(
    model_name: str,
    text: str,
    target: int | None = None,
    predicted: int | None = None,
):
    actual_target = int(target) if target is not None else None
    predicted_target = int(predicted) if predicted is not None else None

    # =========================
    # MNB CUSTOM
    # =========================
    if model_name == "Multinomial Naive Bayes (Custom)":
        model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
        if not os.path.exists(model_path):
            return {"status": "error", "message": "Model not found"}

        priors, w_probs, vocab, totals, counts, _ = load_model(model_path)
        details = get_mnb_custom_prediction_details(text, priors, w_probs, vocab, totals, counts)

        return {
            "status": "success",
            "data": {
                "detail_type": "mnb_custom",
                "model_name": model_name,
                "raw_text": text,
                "actual_target": actual_target,
                "predicted_target": predicted_target,
                "classes": details
            }
        }

    # =========================
    # MNB LIBRARY
    # =========================
    if model_name == "Multinomial Naive Bayes (Library)":
        payload = _build_mnb_library_detail_payload(
            model_name=model_name,
            text=text,
            actual_target=actual_target,
            predicted_target=predicted_target
        )
        return {"status": "success", "data": payload}

    # =========================
    # SVM ONE-SAMPLE CUSTOM
    # =========================
    if model_name == "Linear SVM (One-Sample Custom)":
        svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")
        if not os.path.exists(svm_one_path):
            return {"status": "error", "message": "Model not found"}

        svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
        payload = _build_linear_svm_detail_payload(
            model_name=model_name,
            text=text,
            actual_target=actual_target,
            predicted_target=predicted_target,
            model=svm_one_model,
            vectorizer=svm_one_vectorizer,
            source="custom"
        )
        return {"status": "success", "data": payload}

    # =========================
    # SVM FULL-SAMPLE CUSTOM
    # =========================
    if model_name == "Linear SVM (Full-Sample Custom)":
        svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
        if not os.path.exists(svm_full_path):
            return {"status": "error", "message": "Model not found"}

        svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
        payload = _build_linear_svm_detail_payload(
            model_name=model_name,
            text=text,
            actual_target=actual_target,
            predicted_target=predicted_target,
            model=svm_full_model,
            vectorizer=svm_full_vectorizer,
            source="custom"
        )
        return {"status": "success", "data": payload}

    # =========================
    # SVM LIBRARY
    # =========================
    if model_name == "Linear SVM (Library)":
        svm_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_model.pkl")
        svm_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_library_vectorizer.pkl")

        if not (os.path.exists(svm_lib_model_path) and os.path.exists(svm_lib_vectorizer_path)):
            return {"status": "error", "message": "Model not found"}

        svm_lib_model = joblib.load(svm_lib_model_path)
        svm_lib_vectorizer = joblib.load(svm_lib_vectorizer_path)

        payload = _build_linear_svm_detail_payload(
            model_name=model_name,
            text=text,
            actual_target=actual_target,
            predicted_target=predicted_target,
            model=svm_lib_model,
            vectorizer=svm_lib_vectorizer,
            source="library"
        )
        return {"status": "success", "data": payload}

    # =========================
    # XGB CUSTOM
    # =========================
    if model_name == "XGBoost (Custom)":
        payload = _build_xgb_generic_detail_payload(
            model_name=model_name,
            text=text,
            actual_target=actual_target,
            predicted_target=predicted_target,
            model_type="custom"
        )
        return {"status": "success", "data": payload}

    # =========================
    # XGB LIBRARY
    # =========================
    if model_name == "XGBoost (Library)":
        payload = _build_xgb_generic_detail_payload(
            model_name=model_name,
            text=text,
            actual_target=actual_target,
            predicted_target=predicted_target,
            model_type="library"
        )
        return {"status": "success", "data": payload}

    return {
        "status": "success",
        "data": {
            "detail_type": "unsupported",
            "model_name": model_name,
            "message": f"Chưa hỗ trợ chi tiết cho model '{model_name}'."
        }
    }


# =========================================================
# CHARTS
# =========================================================
@router.get("/charts/{model_type}")
async def get_model_charts(model_type: str):
    charts_map = {
        "mnb_custom_cm": "cm_mnb_custom.png",
        "mnb_library_cm": "cm_mnb_library.png",
        "svm_one_custom_cm": "cm_svm_one_custom.png",
        "svm_full_custom_cm": "cm_svm_full_custom.png",
        "svm_library_cm": "cm_svm_library.png",
        "xgb_custom_cm": "cm_xgb_custom.png",
        "xgb_library_cm": "cm_xgb_library.png",
        "mnb": "laplace_smoothing_study.png",
        "svm": "cm_svm_library.png",
        "xgb": "cm_xgb_library.png",
    }

    if model_type not in charts_map:
        return {"status": "error", "message": "Loại biểu đồ không hợp lệ!"}

    file_name = charts_map[model_type]
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

    # ==========================================
    # CUSTOM - MNB
    # ==========================================
    try:
        mnb_custom_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
        if os.path.exists(mnb_custom_path):
            priors, w_probs, vocab, totals, counts, _ = load_model(mnb_custom_path)
            label, _ = predict_MNB_Custom(cleaned_text, priors, w_probs, vocab, totals)
            custom_preds["mnb"] = int(label)
    except Exception as e:
        print(f"Lỗi MNB Custom: {e}")

    # ==========================================
    # CUSTOM - SVM
    # Ưu tiên Full-Sample, nếu không có thì fallback One-Sample
    # ==========================================
    try:
        svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
        svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")

        if os.path.exists(svm_full_path):
            svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
            label = _predict_single_svm_project_label(svm_full_model, svm_full_vectorizer, cleaned_text)
            custom_preds["svm"] = label

        elif os.path.exists(svm_one_path):
            svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
            label = _predict_single_svm_project_label(svm_one_model, svm_one_vectorizer, cleaned_text)
            custom_preds["svm"] = label
    except Exception as e:
        print(f"Lỗi SVM Custom: {e}")

    # ==========================================
    # CUSTOM - XGB
    # ==========================================
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

            vectorizer = load_vectorizer(xgb_vectorizer_path)

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

    # ==========================================
    # LIBRARY - MNB
    # ==========================================
    try:
        lib_pipeline, _ = load_library_model()
        if lib_pipeline:
            label = int(lib_pipeline.predict([cleaned_text])[0])
            lib_preds["mnb"] = label
    except Exception as e:
        print(f"Lỗi MNB Library: {e}")

    # ==========================================
    # LIBRARY - SVM
    # ==========================================
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

    # ==========================================
    # LIBRARY - XGB
    # ==========================================
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
        "data": {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "custom": custom_preds,
            "library": lib_preds,
            "votes": {
                "custom": custom_vote,
                "library": lib_vote,
                "final": final_vote,
            },
        },
    }