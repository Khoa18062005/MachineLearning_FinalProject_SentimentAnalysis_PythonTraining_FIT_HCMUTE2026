from fastapi import APIRouter
import os
from app.models.Multinomial_Custom import predict_MNB_Custom, load_model
from app.services import ml_service, evaluation_service
from app.core.config import settings
from app.services.evaluation_service import calculate_performance_metrics
from app.services.ml_service import get_clean_datasets_for_training
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

    # =========================
    # 1) MNB CUSTOM
    # =========================
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
    model_key = "MNB_Custom"

    if os.path.exists(model_path):
        priors, w_probs, vocab, totals, train_time = load_model(model_path)

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

    return {"status": "success", "data": results}


@router.get("/model-errors")
async def get_model_errors(model_name: str):
    _, _, df_test = get_clean_datasets_for_training()
    X_test = df_test['text'].fillna('')

    y_pred = []

    if model_name == "Multinomial Naive Bayes (Custom)":
        model_key = "MNB_Custom"

        if model_key in PREDICTION_CACHE:
            y_pred = PREDICTION_CACHE[model_key]
        else:
            model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
            priors, w_probs, vocab, totals, _ = load_model(model_path)
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
            svm_one_model, svm_one_vectorizer = load_svm_one_model(svm_one_path)
            X_test_tfidf = svm_one_vectorizer.transform(X_test).toarray()
            y_pred_svm = svm_one_model.predict(X_test_tfidf)
            y_pred = np.where(y_pred_svm == 1, 4, 0).tolist()
            PREDICTION_CACHE[model_key] = y_pred

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

    df_result = df_test.copy()
    df_result['predicted'] = y_pred
    df_errors = df_result[df_result['target'] != df_result['predicted']]
    result_data = df_errors[['target', 'predicted', 'text']].to_dict(orient="records")

    return {"status": "success", "data": result_data}