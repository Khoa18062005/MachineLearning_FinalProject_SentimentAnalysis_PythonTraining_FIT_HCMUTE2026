from fastapi import APIRouter
import os
import joblib
import numpy as np

from typing import Any
from sklearn.pipeline import FeatureUnion
from starlette.responses import FileResponse

from app.core.config import settings
from app.services import ml_service
from app.services.evaluation_service import (
    calculate_performance_metrics,
    save_confusion_matrix_chart,
    save_accuracy_comparison_chart,
)
from app.services.ml_service import get_clean_datasets_for_training, clean_text

from app.models.Multinomial_Custom import (
    predict_MNB_Custom,
    load_model as load_mnb_custom_model,
    get_prediction_details as get_mnb_custom_prediction_details,
)
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


def _predict_mnb_library_batch(X_texts):
    lib_pipeline, _ = load_library_model()
    if not lib_pipeline:
        return []

    y_pred = lib_pipeline.predict(X_texts).tolist()
    return _ensure_int_labels(y_pred)


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


def _predict_svm_one_custom_batch(X_texts):
    svm_one_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_one_sample_custom.pkl")
    if not os.path.exists(svm_one_path):
        return []

    svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
    return _predict_svm_project_labels(
        model=svm_one_model,
        vectorizer=svm_one_vectorizer,
        X_texts=X_texts,
    )


def _predict_svm_full_custom_batch(X_texts):
    svm_full_path = os.path.join(settings.BASE_DIR, "app", "models", "svm_full_sample_custom.pkl")
    if not os.path.exists(svm_full_path):
        return []

    svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
    return _predict_svm_project_labels(
        model=svm_full_model,
        vectorizer=svm_full_vectorizer,
        X_texts=X_texts,
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
        "svm_one_custom_cm": ["cm_svm_one_custom.png"],
        "svm_full_custom_cm": ["cm_svm_full_custom.png"],
        "svm_library_cm": ["cm_svm_library.png"],
        "svm_cm": ["cm_svm_library.png", "cm_svm_full_custom.png", "cm_svm_one_custom.png", "cm_svm.png"],
        "svm_custom_cm": ["cm_svm_full_custom.png", "cm_svm_one_custom.png", "cm_svm.png"],
        "xgb_custom_cm": ["cm_xgb_custom.png", "cm_xgb.png"],
        "xgb_library_cm": ["cm_xgb_library.png", "cm_xgb.png"],
        "xgb_cm": ["cm_xgb_library.png", "cm_xgb_custom.png", "cm_xgb.png"],
        "svm": ["svm_study.png", "cm_svm_library.png"],
        "xgb": ["xgb_study.png", "cm_xgb_library.png"],
        "ensemble_custom_cm": ["cm_ensemble_custom.png"],
        "ensemble_library_cm": ["cm_ensemble_library.png"],
        "ensemble_dual_cm": ["cm_ensemble_dual.png"],
        "accuracy_custom": ["accuracy_comp_custom.png"],
        "accuracy_library": ["accuracy_comp_library.png"],
        "accuracy_dual": ["accuracy_comp_dual.png"],
    }
    return charts_map.get(model_type, [])


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
    - Nếu không có hàm detail riêng thì vẫn trả ra payload đủ để modal không bị hỏng
    """
    if model_type == "custom":
        try:
            from app.models import XGBoost_Custom as xgb_custom_module

            detail_fn = getattr(xgb_custom_module, "get_prediction_details_XGB_Custom", None)
            if detail_fn is not None:
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
                    if isinstance(details, dict):
                        details = make_json_safe(details)
                        details["model_type"] = "xgb_custom"
                        details["detail_type"] = details.get("detail_type", "xgb_custom")
                        return details
        except Exception as e:
            print(f"[xgb-custom-detail] fallback lỗi: {e}")

        return {
            "detail_type": "unsupported",
            "model_name": model_name,
            "message": "XGBoost Custom hiện chưa hỗ trợ bóc chi tiết từng bước trong file bạn gửi."
        }

    if model_type == "library":
        try:
            from app.models import XGBoost_Library as xgb_library_module

            detail_fn = getattr(xgb_library_module, "get_prediction_details_XGB_Library", None)
            if detail_fn is not None:
                xgb_lib_model_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_model_library.pkl")
                xgb_lib_vectorizer_path = os.path.join(settings.BASE_DIR, "app", "models", "XGB_vectorizer_library.pkl")

                if os.path.exists(xgb_lib_model_path) and os.path.exists(xgb_lib_vectorizer_path):
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
                    if isinstance(details, dict):
                        details = make_json_safe(details)
                        details["model_type"] = "xgb_library"
                        details["detail_type"] = details.get("detail_type", "xgb_library")
                        return details
        except Exception as e:
            print(f"[xgb-library-detail] fallback lỗi: {e}")

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
        "message": "Model XGBoost không được hỗ trợ."
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
    preds_custom = {"mnb": [], "svm": [], "xgb": []}
    preds_library = {"mnb": [], "svm": [], "xgb": []}

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

            preds_custom["mnb"] = y_pred
            save_confusion_matrix_chart(y_test, y_pred, "mnb_custom")

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
            svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
            model_key = "SVM_OneSample_Custom"

            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_project_labels(svm_one_model, svm_one_vectorizer, X_test)
                PREDICTION_CACHE[model_key] = y_pred

            save_confusion_matrix_chart(y_test, y_pred, "svm_one_custom")

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
            svm_full_model, svm_full_vectorizer = load_svm_full_model(svm_full_path)
            model_key = "SVM_FullSample_Custom"

            if model_key in PREDICTION_CACHE:
                y_pred = PREDICTION_CACHE[model_key]
            else:
                y_pred = _predict_svm_project_labels(svm_full_model, svm_full_vectorizer, X_test)
                PREDICTION_CACHE[model_key] = y_pred

            preds_custom["svm"] = y_pred
            save_confusion_matrix_chart(y_test, y_pred, "svm_full_custom")

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

            preds_library["svm"] = y_pred
            save_confusion_matrix_chart(y_test, y_pred, "svm_library")

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

            preds_library["mnb"] = y_pred
            save_confusion_matrix_chart(y_test, y_pred, "mnb_library")

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

            preds_custom["xgb"] = y_pred
            save_confusion_matrix_chart(y_test, y_pred, "xgb_custom")

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

            preds_library["xgb"] = y_pred
            save_confusion_matrix_chart(y_test, y_pred, "xgb_library")

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

    # 8. ENSEMBLE VOTING - NHÓM CUSTOM
    y_pred_custom_group = []
    if len(preds_custom["mnb"]) > 0 and len(preds_custom["svm"]) > 0 and len(preds_custom["xgb"]) > 0:
        for i in range(len(y_test)):
            votes = {
                "mnb": preds_custom["mnb"][i],
                "svm": preds_custom["svm"][i],
                "xgb": preds_custom["xgb"][i]
            }
            y_pred_custom_group.append(_get_majority_vote(votes))

        save_confusion_matrix_chart(y_test, y_pred_custom_group, "ensemble_custom")
        results.append(_safe_metrics(
            model_name="Nhóm Custom (Majority Vote)",
            y_true=y_test,
            y_pred=y_pred_custom_group,
            training_time_sec=0.0
        ))

    # 9. ENSEMBLE VOTING - NHÓM LIBRARY
    y_pred_library_group = []
    if len(preds_library["mnb"]) > 0 and len(preds_library["svm"]) > 0 and len(preds_library["xgb"]) > 0:
        for i in range(len(y_test)):
            votes = {
                "mnb": preds_library["mnb"][i],
                "svm": preds_library["svm"][i],
                "xgb": preds_library["xgb"][i]
            }
            y_pred_library_group.append(_get_majority_vote(votes))

        save_confusion_matrix_chart(y_test, y_pred_library_group, "ensemble_library")
        results.append(_safe_metrics(
            model_name="Nhóm Library (Majority Vote)",
            y_true=y_test,
            y_pred=y_pred_library_group,
            training_time_sec=0.0
        ))

    # 10. TRACK-DUAL VALIDATION (KẾT HỢP CẢ 2 NHÓM)
    if len(y_pred_custom_group) > 0 and len(y_pred_library_group) > 0:
        y_pred_dual_group = []
        for i in range(len(y_test)):
            cust_vote = y_pred_custom_group[i]
            lib_vote = y_pred_library_group[i]

            # Logic: Nếu giống thì chọn, khác thì ưu tiên Library
            if cust_vote == lib_vote:
                y_pred_dual_group.append(cust_vote)
            else:
                y_pred_dual_group.append(lib_vote)

        save_confusion_matrix_chart(y_test, y_pred_dual_group, "ensemble_dual")
        results.append(_safe_metrics(
            model_name="Track-Dual Validation (Majority Vote)",
            y_true=y_test,
            y_pred=y_pred_dual_group,
            training_time_sec=0.0
        ))

    # 11. VẼ ĐỒ THỊ ACCURACY THEO TỪNG NHÓM THUẬT TOÁN
    # 11.1 Custom
    custom_list = [r for r in results if "Custom" in r['model_name'] and "Track-Dual" not in r['model_name']]
    if custom_list:
        save_accuracy_comparison_chart(custom_list, "custom")

    # 11.2 Library
    library_list = [r for r in results if "Library" in r['model_name'] and "Track-Dual" not in r['model_name']]
    if library_list:
        save_accuracy_comparison_chart(library_list, "library")

    # 11.3 Nhóm Track-Dual
    dual_list = []
    for r in results:
        if "Track-Dual Validation" in r['model_name']:
            dual_list.append(r)
        elif "Nhóm Custom (Majority Vote)" in r['model_name']:
            temp = r.copy()
            temp['model_name'] = "Nhóm Custom"  # Đổi tên để hàm vẽ chart nhận làm màu xám (sub-model)
            dual_list.append(temp)
        elif "Nhóm Library (Majority Vote)" in r['model_name']:
            temp = r.copy()
            temp['model_name'] = "Nhóm Library"
            dual_list.append(temp)

    if [r for r in dual_list if "Track-Dual" in r['model_name']]:
        save_accuracy_comparison_chart(dual_list, "dual")

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

        elif "Majority Vote" in model_name:
            if "Track-Dual" in model_name:
                mnb_l = PREDICTION_CACHE.get(
                    "MNB_Library") if "MNB_Library" in PREDICTION_CACHE else _predict_mnb_library_batch(X_test)
                svm_l = PREDICTION_CACHE.get(
                    "SVM_Library") if "SVM_Library" in PREDICTION_CACHE else _predict_svm_library_batch(X_test)
                xgb_l = PREDICTION_CACHE.get(
                    "XGB_Library") if "XGB_Library" in PREDICTION_CACHE else _predict_xgb_library_batch(X_test)

                mnb_c = PREDICTION_CACHE.get(
                    "MNB_Custom") if "MNB_Custom" in PREDICTION_CACHE else _predict_mnb_custom_batch(X_test)
                svm_c = PREDICTION_CACHE.get(
                    "SVM_FullSample_Custom") if "SVM_FullSample_Custom" in PREDICTION_CACHE else _predict_svm_full_custom_batch(
                    X_test)
                xgb_c = PREDICTION_CACHE.get(
                    "XGB_Custom") if "XGB_Custom" in PREDICTION_CACHE else _predict_xgb_custom_batch(X_test)

                if len(mnb_l) > 0 and len(svm_l) > 0 and len(xgb_l) > 0 and len(mnb_c) > 0 and len(svm_c) > 0 and len(
                        xgb_c) > 0:
                    for i in range(len(X_test)):
                        lib_vote = _get_majority_vote({"mnb": mnb_l[i], "svm": svm_l[i], "xgb": xgb_l[i]})
                        cust_vote = _get_majority_vote({"mnb": mnb_c[i], "svm": svm_c[i], "xgb": xgb_c[i]})

                        # Logic chốt hạ
                        if cust_vote == lib_vote:
                            y_pred.append(cust_vote)
                        else:
                            y_pred.append(lib_vote)
            else:
                is_library = "Library" in model_name
                if is_library:
                    mnb = PREDICTION_CACHE.get(
                        "MNB_Library") if "MNB_Library" in PREDICTION_CACHE else _predict_mnb_library_batch(X_test)
                    svm = PREDICTION_CACHE.get(
                        "SVM_Library") if "SVM_Library" in PREDICTION_CACHE else _predict_svm_library_batch(X_test)
                    xgb = PREDICTION_CACHE.get(
                        "XGB_Library") if "XGB_Library" in PREDICTION_CACHE else _predict_xgb_library_batch(X_test)
                else:
                    mnb = PREDICTION_CACHE.get(
                        "MNB_Custom") if "MNB_Custom" in PREDICTION_CACHE else _predict_mnb_custom_batch(X_test)
                    svm = PREDICTION_CACHE.get(
                        "SVM_FullSample_Custom") if "SVM_FullSample_Custom" in PREDICTION_CACHE else _predict_svm_full_custom_batch(
                        X_test)
                    xgb = PREDICTION_CACHE.get(
                        "XGB_Custom") if "XGB_Custom" in PREDICTION_CACHE else _predict_xgb_custom_batch(X_test)

                if len(mnb) > 0 and len(svm) > 0 and len(xgb) > 0:
                    for i in range(len(X_test)):
                        votes = {"mnb": mnb[i], "svm": svm[i], "xgb": xgb[i]}
                        y_pred.append(_get_majority_vote(votes))

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
async def get_model_prediction_details(
        model_name: str,
        text: str,
        target: int | None = None,
        predicted: int | None = None,
):
    actual_target = int(target) if target is not None else None
    predicted_target = int(predicted) if predicted is not None else None

    # =========================
    # XỬ LÝ CHI TIẾT CHO NHÓM ENSEMBLE (VOTING) & TRACK-DUAL
    # =========================
    if "Majority Vote" in model_name or "Track-Dual" in model_name:
        # 1. Tiền xử lý văn bản đầu vào
        cleaned_text = clean_text(text)

        # 2. Phân loại xem đang yêu cầu nhóm Library, Custom hay Track-Dual
        is_library = "Library" in model_name
        is_dual = "Track-Dual" in model_name
        votes_detail = []

        try:
            if is_dual:
                # Tính toán kết quả của nhóm Library (3 thuật toán)
                mnb_l = _predict_mnb_library_batch([cleaned_text])[0]
                svm_l = _predict_svm_library_batch([cleaned_text])[0]
                xgb_l = _predict_xgb_library_batch([cleaned_text])[0]
                lib_vote = _get_majority_vote({"mnb": mnb_l, "svm": svm_l, "xgb": xgb_l})

                # Tính toán kết quả của nhóm Custom (3 thuật toán)
                mnb_c = _predict_mnb_custom_batch([cleaned_text])[0]
                svm_c = _predict_svm_one_custom_batch([cleaned_text])[0]
                xgb_c = _predict_xgb_custom_batch([cleaned_text])[0]
                cust_vote = _get_majority_vote({"mnb": mnb_c, "svm": svm_c, "xgb": xgb_c})

                # Trả về 2 lá phiếu đại diện cho 2 nhóm
                votes_detail = [
                    {"name": "Nhóm Custom (Đa số)", "pred": cust_vote},
                    {"name": "Nhóm Library (Đa số)", "pred": lib_vote}
                ]

            elif is_library:
                # Gọi dự đoán từ 3 thuật toán nhóm Library
                mnb_p = _predict_mnb_library_batch([cleaned_text])[0]
                svm_p = _predict_svm_library_batch([cleaned_text])[0]
                xgb_p = _predict_xgb_library_batch([cleaned_text])[0]

                votes_detail = [
                    {"name": "Multinomial NB (Library)", "pred": mnb_p},
                    {"name": "Linear SVM (Library)", "pred": svm_p},
                    {"name": "XGBoost (Library)", "pred": xgb_p}
                ]
            else:
                # Gọi dự đoán từ 3 thuật toán nhóm Custom
                mnb_p = _predict_mnb_custom_batch([cleaned_text])[0]
                svm_p = _predict_svm_one_custom_batch([cleaned_text])[0]
                xgb_p = _predict_xgb_custom_batch([cleaned_text])[0]

                votes_detail = [
                    {"name": "Multinomial NB (Custom)", "pred": mnb_p},
                    {"name": "Linear SVM (Custom)", "pred": svm_p},
                    {"name": "XGBoost (Custom)", "pred": xgb_p}
                ]

            return {
                "status": "success",
                "data": make_json_safe({
                    "detail_type": "ensemble_voting",
                    "model_name": model_name,
                    "raw_text": text,
                    "cleaned_text": cleaned_text,
                    "actual_target": actual_target,
                    "predicted_target": predicted_target,
                    "votes": votes_detail
                })
            }
        except Exception as e:
            print(f"Lỗi khi lấy chi tiết Voting: {e}")
            return {"status": "error", "message": "Không thể tính toán chi tiết phiếu bầu."}

    # =========================
    # MNB CUSTOM
    # =========================
    if model_name == "Multinomial Naive Bayes (Custom)":
        model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
        if not os.path.exists(model_path):
            return {"status": "error", "message": "Model not found"}

        priors, w_probs, vocab, totals, counts, _ = load_mnb_custom_model(model_path)
        details = get_mnb_custom_prediction_details(text, priors, w_probs, vocab, totals, counts)

        return {
            "status": "success",
            "data": make_json_safe({
                "detail_type": "mnb_custom",
                "model_name": model_name,
                "raw_text": text,
                "actual_target": actual_target,
                "predicted_target": predicted_target,
                "classes": details
            })
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
        return {"status": "success", "data": make_json_safe(payload)}

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
        return {"status": "success", "data": make_json_safe(payload)}

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
        return {"status": "success", "data": make_json_safe(payload)}

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
        return {"status": "success", "data": make_json_safe(payload)}

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
        return {"status": "success", "data": make_json_safe(payload)}

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
        return {"status": "success", "data": make_json_safe(payload)}

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