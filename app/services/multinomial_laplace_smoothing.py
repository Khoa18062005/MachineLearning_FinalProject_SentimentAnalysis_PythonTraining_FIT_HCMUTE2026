import os
from app.core.config import settings
import matplotlib.pyplot as plt
import numpy as np
import app.models.Multinomial_Custom as mnb
from app.services.ml_service import get_clean_datasets_for_training

def plot_laplace_results(alphas, accuracies):
    plt.figure(figsize=(10, 6))
    # Vẽ đường biểu đồ chính
    plt.plot(alphas, accuracies, marker='o', linestyle='-', color='#3b82f6',
             linewidth=2, markersize=5, label='Accuracy theo Alpha')

    # Kẻ đường thẳng đứng tại Alpha = 1.0 (Laplace Smoothing)
    plt.axvline(x=1.0, color='#ef4444', linestyle='--', label='Laplace Smoothing (α=1)')

    # Thêm các thông tin bổ trợ
    plt.title('Chứng minh tính tối ưu của Laplace Smoothing', fontsize=14, fontweight='bold')
    plt.xlabel('Hệ số làm trơn Alpha ($\\alpha$)', fontsize=12)
    plt.ylabel('Độ chính xác Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Lưu và hiển thị
    plt.tight_layout()
    save_path = os.path.join(settings.BASE_DIR, 'laplace_smoothing_study.png')
    plt.savefig(save_path, dpi=300)
    plt.show()

def run_laplace_study():
    df_train, _, df_test = get_clean_datasets_for_training()
    X_train, y_train = df_train['text'], df_train['target']
    X_test, y_test = df_test['text'], df_test['target']

    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_custom.pkl")
    priors, _, vocab, totals, counts, _ = mnb.load_model(model_path)

    # Tạo giới hạn dãy laplace smoothing
    alphas = np.linspace(0, 1, 30)
    accuracies = []
    print('- Đang thực hiện đánh khởi tạo đánh giá')

    for alpha in alphas:
        mnb.laplace_smoothing = alpha
        word_probs = mnb.calculate_word_probs(counts, totals, vocab)
        correct = 0
        total_count = len(X_test)
        for text, true_label in zip(X_test, y_test):
            # Hàm predict sẽ tự động dùng alpha mới cho các từ chưa thấy (OOV)
            prediction, _ = mnb.predict_MNB_Custom(text, priors, word_probs, vocab, totals)
            if prediction == true_label:
                correct += 1

            # LƯU KẾT QUẢ: Tính % độ chính xác
        accuracy = (correct / total_count) * 100
        accuracies.append(accuracy)

    plot_laplace_results(alphas, accuracies)

if __name__ == "__main__":
    run_laplace_study()