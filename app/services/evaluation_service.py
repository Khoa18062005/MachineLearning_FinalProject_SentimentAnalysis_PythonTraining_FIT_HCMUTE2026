import numpy as np
from sklearn.metrics import accuracy_score

def calculate_performance_metrics(model_name, y_true, y_pred, training_time_sec):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # 1. Tính số lượng dự đoán đúng / sai
    correct = int(np.sum(y_true_np == y_pred_np))
    incorrect = len(y_true) - correct

    # 2. Tính độ chính xác (Accuracy) theo %
    accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)

    # 3. Đóng gói thành Dictionary để trả về API
    return {
        "model_name": model_name,
        "training_time_sec": training_time_sec,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "accuracy": accuracy
    }