import os
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.patches as patches
matplotlib.use('Agg')
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
    plt.figure(figsize=(12, 6))
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


import matplotlib.patches as patches  # Thêm dòng này ở đầu file


def save_accuracy_comparison_chart(metrics_list, group_id):
    # 1. Lọc bỏ mô hình One-Sample và các mô hình chưa huấn luyện
    valid_data = [
        m for m in metrics_list
        if m.get('accuracy', 0) > 0 and "one-sample" not in m['model_name'].lower()
    ]

    if not valid_data:
        return None

    labels = []
    accuracies = []
    colors = []

    # Màu sắc thẩm mỹ trên nền trắng
    main_highlight_color = '#ef4444'
    sub_muted_color = '#2A95BF'

    for m in valid_data:
        name = m['model_name']
        display_name = name.replace(" (Custom)", "").replace(" (Library)", "").replace(" (Majority Vote)", "")
        if "Linear SVM" in display_name: display_name = "SVM"
        if "Multinomial Naive Bayes" in display_name: display_name = "MNB"
        if group_id == 'dual':
            if name == "Nhóm Custom":
                display_name = "Nhóm Custom"
            elif name == "Nhóm Library":
                display_name = "Nhóm Library"
            elif "Track-Dual" in name:
                display_name = "Track-Dual"
        else:
            if "Nhóm" in name: display_name = "Ensemble"

        labels.append(display_name)
        accuracies.append(m['accuracy'])

        if "Majority Vote" in name:
            colors.append(main_highlight_color)
        else:
            colors.append(sub_muted_color)

    plt.figure(figsize=(18, 10), facecolor='white')

    # Tạo viền trắng bo góc
    fig = plt.gcf()
    fig.patch.set_alpha(0)
    rect = patches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                  boxstyle="round,pad=0.01,rounding_size=0.04",
                                  facecolor="white", edgecolor="none",
                                  transform=fig.transFigure, zorder=-1)
    fig.patches.append(rect)

    ax = plt.gca()
    ax.set_facecolor('none')  # Để 'none' để thấy nền trắng bo góc bên dưới

    bars = plt.bar(labels, accuracies, color=colors, alpha=0.85, width=0.6, zorder=3)

    # Tinh chỉnh tiêu đề và trục
    plt.title(f'So sánh Hiệu suất Nhóm {group_id.upper()}', fontsize=20, pad=20, fontweight='bold', color='#1e293b')
    plt.ylim(0, 110)
    plt.ylabel('Accuracy (%)', color='#475569', fontsize=16)
    plt.xticks(fontsize=16, fontweight='medium', color='#475569')
    plt.yticks(fontsize=16, color='#475569')

    # Kẻ lưới mờ ngang
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    # Hiển thị số liệu trên đầu cột
    for bar in bars:
        height = bar.get_height()
        is_red = bar.get_facecolor() == matplotlib.colors.to_rgba(main_highlight_color, 0.85)
        plt.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height}%', ha='center', va='bottom',
                 fontsize=16 if is_red else 16,
                 fontweight='bold' if is_red else 'normal',
                 color='#0f172a' if is_red else '#64748b')

    # Loại bỏ đường viền hộp biểu đồ
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')

    plt.tight_layout(pad=3.0)

    # 3. Lưu file với cấu hình bạn yêu cầu
    file_name = f"accuracy_comp_{group_id}.png"
    save_path = os.path.join(settings.BASE_DIR, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    return file_name