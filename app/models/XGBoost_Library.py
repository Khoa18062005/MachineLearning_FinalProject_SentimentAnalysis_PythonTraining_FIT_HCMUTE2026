import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

from app.services.evaluation_service import save_confusion_matrix_chart


# =========================================================
# FIX PATH ĐỂ CHẠY TRỰC TIẾP FILE NÀY CŨNG IMPORT ĐƯỢC app.*
# =========================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2] if len(CURRENT_FILE.parents) >= 3 else CURRENT_FILE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# LOAD MODEL / VECTORIZER
# =========================================================
def load_model(file_path):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)

    return (
        model_data["model"],
        model_data["feature_names"],
        model_data["classes"],
        model_data["threshold"],
        model_data.get("training_time_sec", 0.0),
        model_data.get("params", {})
    )


def load_vectorizer(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# =========================================================
# PHẦN XỬ LÝ TEXT -> VECTOR
# =========================================================
def fit_text_vectorizer(X_train_text, max_features=5000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )

    X_vec = vectorizer.fit_transform(
        pd.Series(X_train_text).fillna("").astype(str)
    )

    feature_names = list(vectorizer.get_feature_names_out())
    return vectorizer, X_vec, feature_names


def transform_text_with_vectorizer(X_text, vectorizer):
    return vectorizer.transform(
        pd.Series(X_text).fillna("").astype(str)
    )


# =========================================================
# PHẦN XỬ LÝ LABEL
# =========================================================
def encode_target(y):
    if isinstance(y, pd.Series):
        y_array = y.to_numpy()
    else:
        y_array = np.asarray(y)

    classes = list(pd.unique(y_array))

    if len(classes) != 2:
        raise ValueError("XGBoost Library bản này chỉ hỗ trợ binary classification (2 lớp).")

    label_to_int = {classes[0]: 0, classes[1]: 1}
    y_encoded = np.array([label_to_int[label] for label in y_array], dtype=int)

    return y_encoded, classes


# =========================================================
# TRAIN XGBOOST LIBRARY - LUÔN AUTO TUNE
# =========================================================
def train_XGB_Library(
    X_train_text,
    y_train,
    max_features=5000,
    ngram_range=(1, 2),
    threshold=0.5,
    random_state=42,
    n_iter=20,
    cv=3,
    scoring="accuracy"
):
    start_time = time.time()

    vectorizer, X_train_vec, feature_names = fit_text_vectorizer(
        X_train_text=X_train_text,
        max_features=max_features,
        ngram_range=ngram_range
    )

    y_encoded, classes = encode_target(y_train)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=1
    )

    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.5, 1, 2, 5]
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train_vec, y_encoded)

    best_model = search.best_estimator_
    training_duration = round(time.time() - start_time, 2)

    model_data = {
        "model": best_model,
        "feature_names": feature_names,
        "classes": classes,
        "threshold": threshold,
        "training_time_sec": training_duration,
        "params": {
            "max_features": max_features,
            "ngram_range": ngram_range,
            "random_state": random_state,
            "n_iter": n_iter,
            "cv": cv,
            "scoring": scoring,
            "best_params": search.best_params_,
            "best_score_cv": float(search.best_score_)
        },
        "best_params": search.best_params_,
        "best_score_cv": float(search.best_score_)
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "XGB_model_library.pkl")
    vectorizer_path = os.path.join(current_dir, "XGB_vectorizer_library.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    return model_data


# =========================================================
# PREDICT XGBOOST LIBRARY
# =========================================================
def predict_XGB_Library(
    text_input,
    model,
    classes,
    threshold=0.5,
    vectorizer=None
):
    if vectorizer is None:
        raise ValueError("Cần truyền vectorizer khi dự đoán text cho XGBoost Library.")

    if isinstance(text_input, str):
        X_input = [text_input]
    else:
        X_input = text_input

    X_vec = transform_text_with_vectorizer(X_input, vectorizer)

    prob_positive = model.predict_proba(X_vec)[:, 1]
    prob_negative = 1.0 - prob_positive

    pred_int = (prob_positive >= threshold).astype(int)
    predictions = np.where(pred_int == 1, classes[1], classes[0])

    results_df = pd.DataFrame({
        classes[0]: prob_negative,
        classes[1]: prob_positive
    })

    if len(predictions) == 1:
        return predictions[0], results_df.iloc[0].to_dict()

    return predictions, results_df


def evaluate_XGB_Library(X_eval_text, y_eval, model_data, vectorizer):
    y_pred, _ = predict_XGB_Library(
        text_input=list(pd.Series(X_eval_text).fillna("").astype(str)),
        model=model_data["model"],
        classes=model_data["classes"],
        threshold=model_data["threshold"],
        vectorizer=vectorizer
    )

    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred.tolist()
    else:
        y_pred = [y_pred]

    acc = accuracy_score(y_eval, y_pred)
    return float(acc), y_pred


# =========================================================
# GIẢI THÍCH QUYẾT ĐỊNH CHO 1 MẪU
# =========================================================
def get_prediction_details_XGB_Library(text_input, model_data, vectorizer, top_k_features=15):
    model = model_data["model"]
    classes = model_data["classes"]
    threshold = float(model_data["threshold"])

    X_vec = transform_text_with_vectorizer([text_input], vectorizer)
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_vec, feature_names=model_data["feature_names"])

    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    bias = float(contribs[-1])
    feature_contribs = contribs[:-1]

    prob_positive = float(model.predict_proba(X_vec)[0, 1])
    prob_negative = float(1.0 - prob_positive)
    predicted_label = classes[1] if prob_positive >= threshold else classes[0]

    nonzero_idx = X_vec[0].nonzero()[1]
    tfidf_lookup = {int(idx): float(X_vec[0, idx]) for idx in nonzero_idx}

    rows = []
    for idx, contrib in enumerate(feature_contribs):
        tfidf_value = tfidf_lookup.get(idx, 0.0)
        if tfidf_value == 0.0 and abs(float(contrib)) < 1e-12:
            continue

        rows.append({
            "feature": model_data["feature_names"][idx],
            "tfidf_value": tfidf_value,
            "contribution": float(contrib),
            "direction": "ủng hộ lớp dương" if float(contrib) >= 0 else "ủng hộ lớp âm"
        })

    rows = sorted(rows, key=lambda item: abs(item["contribution"]), reverse=True)[:top_k_features]

    booster_feature_gain = booster.get_score(importance_type="gain")
    global_feature_gain = sorted(
        [
            {"feature": feature, "gain": float(gain)}
            for feature, gain in booster_feature_gain.items()
        ],
        key=lambda item: item["gain"],
        reverse=True
    )[:10]

    return {
        "model_type": "xgb_library",
        "input_text": str(text_input),
        "predicted_label": predicted_label,
        "threshold": threshold,
        "probabilities": {
            str(classes[0]): prob_negative,
            str(classes[1]): prob_positive
        },
        "bias": bias,
        "training_summary": {
            "training_time_sec": float(model_data.get("training_time_sec", 0.0)),
            "best_score_cv": model_data.get("best_score_cv"),
            "best_params": model_data.get("best_params"),
            "params": model_data.get("params", {})
        },
        "top_features": rows,
        "global_feature_gain": global_feature_gain,
        "decision_reason": (
            "Bản Library dùng pred_contribs từ booster để giải thích từng feature "
            "đã kéo xác suất về lớp âm hay lớp dương."
        )
    }


# =========================================================
# CHẠY TRAIN
# =========================================================
if __name__ == "__main__":
    from app.services.ml_service import get_clean_datasets_for_training

    df_train, df_val, df_test = get_clean_datasets_for_training()

    X_train = df_train["text"].fillna("")
    y_train = df_train["target"]

    X_test = df_test["text"].fillna("")
    y_test = df_test["target"]

    model_metadata = train_XGB_Library(
        X_train_text=X_train,
        y_train=y_train,
        max_features=5000,
        ngram_range=(1, 2),
        threshold=0.5,
        random_state=42,
        n_iter=20,
        cv=3,
        scoring="accuracy"
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "XGB_model_library.pkl")
    vectorizer_path = os.path.join(current_dir, "XGB_vectorizer_library.pkl")

    with open(model_path, "rb") as f:
        trained_model_data = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        trained_vectorizer = pickle.load(f)

    _, y_pred = evaluate_XGB_Library(
        X_eval_text=X_test,
        y_eval=y_test,
        model_data=trained_model_data,
        vectorizer=trained_vectorizer
    )

    save_confusion_matrix_chart(y_test, y_pred, "xgb_library")

    print("Train xong")
    print("Training time:", model_metadata["training_time_sec"], "giây")
    print("Best params:", model_metadata["best_params"])
    print("Best CV score:", model_metadata["best_score_cv"])
    print("Model saved at:", model_path)
    print("Vectorizer saved at:", vectorizer_path)
    print("Đã lưu confusion matrix: xgb_library_cm.png")
