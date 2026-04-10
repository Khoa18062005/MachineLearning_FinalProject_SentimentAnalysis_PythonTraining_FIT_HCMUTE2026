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

    # =========================================================
    # 1) ĐÁNH GIÁ MÔ HÌNH MNB CUSTOM
    # =========================================================
    mnb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
    mnb_model_key = "MNB_Custom"

    if os.path.exists(mnb_model_path):
        priors, w_probs, vocab, totals, train_time = load_model(mnb_model_path)

        if mnb_model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[mnb_model_key]
        else:
            y_pred = []
            for text in X_test:
                label, _ = predict_MNB_Custom(text, priors, w_probs, vocab, totals)
                y_pred.append(label)

            PREDICTION_CACHE[mnb_model_key] = y_pred

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
            mnb_model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
            priors, w_probs, vocab, totals, _ = load_model(mnb_model_path)

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