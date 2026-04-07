import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from app.core.config import settings
from app.services.ml_service import get_clean_datasets_for_training


def evaluate_model_performance(y_true, y_pred, training_duration, model_name):
    """
    Hàm chuẩn hóa kết quả đánh giá cho tất cả các mô hình.
    """
    acc = accuracy_score(y_true, y_pred)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    correct = np.sum(y_true_np == y_pred_np)
    incorrect = len(y_true) - correct

    return {
        "model_name": model_name,
        "training_time_sec": training_duration,  # Sử dụng biến được truyền vào
        "correct_predictions": int(correct),
        "incorrect_predictions": int(incorrect),
        "accuracy": round(acc * 100, 2)
    }


def get_mnb_real_performance():
    """
    Hàm load mô hình MNB đã huấn luyện và đánh giá trên tập Validation
    để lấy thông số THẬT hiển thị lên giao diện.
    """
    model_path = settings.MODEL_PATH
    vectorizer_path = settings.MODEL_PATH.replace('.pkl', '_vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None

    try:
        # 1. Load mô hình và vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # 🔥 Lấy thời gian huấn luyện đã được lưu sẵn trong mô hình
        # Dùng getattr để lấy, nếu không tìm thấy sẽ mặc định là 0.0 để tránh lỗi
        saved_training_time = getattr(model, 'training_time_sec', 0.0)

        # 2. Lấy tập Validation đã làm sạch
        _, df_val, _ = get_clean_datasets_for_training()
        X_val = df_val['text'].fillna('')
        y_val = df_val['target']

        # 3. Tiến hành dự đoán (không dùng time.time() ở đây nữa)
        X_val_tfidf = vectorizer.transform(X_val)
        y_pred = model.predict(X_val_tfidf)

        # 4. Đóng gói kết quả
        return evaluate_model_performance(
            y_true=y_val,
            y_pred=y_pred,
            training_duration=saved_training_time,
            model_name="Multinomial Naive Bayes"
        )
    except Exception as e:
        print(f"Lỗi khi đánh giá MNB thật: {e}")
        return None