import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from app.core.config import settings


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

def save_confusion_matrix_chart(y_true, y_pred, model_name_id):
    """
    Tự động tạo và lưu biểu đồ Confusion Matrix.
    model_name_id: ví dụ 'mnb_custom', 'svm', 'xgb'
    """
    # 1. Tính toán ma trận nhầm lẫn (nhãn của bạn là 0 và 4)
    labels = [0, 4]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 2. Vẽ biểu đồ bằng Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tiêu cực (0)', 'Tích cực (4)'],
                yticklabels=['Tiêu cực (0)', 'Tích cực (4)'])

    plt.title(f'Confusion Matrix - {model_name_id.upper()}')
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')

    # 3. Lưu file
    file_name = f"cm_{model_name_id}.png"
    save_path = os.path.join(settings.BASE_DIR, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Giải phóng bộ nhớ
    return file_name