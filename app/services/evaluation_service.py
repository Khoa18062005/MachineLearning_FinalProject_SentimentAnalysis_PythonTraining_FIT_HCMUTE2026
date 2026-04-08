import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from app.core.config import settings
from app.services.ml_service import get_clean_datasets_for_training

def evaluate_model_performance(y_true, y_pred, training_duration, model_name):
    # (Giữ nguyên hàm này của bạn)
    acc = accuracy_score(y_true, y_pred)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    correct = np.sum(y_true_np == y_pred_np)
    incorrect = len(y_true) - correct

    return {
        "model_name": model_name,
        "training_time_sec": training_duration,
        "correct_predictions": int(correct),
        "incorrect_predictions": int(incorrect),
        "accuracy": round(acc * 100, 2)
    }

def get_general_model_performance(model_path, vectorizer_path, model_name, X_val, y_val):
    """
    Hàm đánh giá chuẩn chung cho mọi mô hình (MNB, SVM, XGBoost...)
    """
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None

    try:
        # 1. Load mô hình và vectorizer tương ứng
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # 2. Lấy thời gian huấn luyện (nếu đã được gắn vào object)
        saved_training_time = getattr(model, 'training_time_sec', 0.0)

        # 3. Tiến hành transform và dự đoán
        X_val_tfidf = vectorizer.transform(X_val)
        y_pred = model.predict(X_val_tfidf)

        # 4. Đóng gói kết quả bằng hàm có sẵn
        return evaluate_model_performance(
            y_true=y_val,
            y_pred=y_pred,
            training_duration=saved_training_time,
            model_name=model_name
        )
    except Exception as e:
        print(f"Lỗi khi đánh giá mô hình {model_name}: {e}")
        return None