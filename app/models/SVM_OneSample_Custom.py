import os
import time
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.ml_service import get_clean_datasets_for_training
from app.core.pandas_helper import setup_pandas_display

setup_pandas_display()


class LinearSVMOneSample:
    """
    Linear SVM from scratch - huấn luyện theo cơ chế one-sample / stochastic
    Objective:
        J_i(w, b) = (lambda_/2) * ||w||^2 + max(0, 1 - y_i (w^T x_i + b))

    Nhãn yêu cầu:
        y ∈ {-1, +1}
    """

    def __init__(self, learning_rate=0.01, lambda_=0.0001, epochs=5, shuffle=True, random_state=42):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.epochs = epochs
        self.shuffle = shuffle
        self.random_state = random_state

        self.w = None
        self.b = 0.0
        self.loss_history = []
        self.training_time_sec = 0.0

    def _hinge_loss_full(self, X, y):
        """
        Tính full objective trên toàn bộ tập:
            J(w,b) = (lambda_/2)||w||^2 + (1/n) * sum_i max(0, 1 - y_i(w^Tx_i+b))
        """
        # ✅ Sparse-safe: X.dot(w) hoạt động đúng cả sparse lẫn dense
        scores = np.asarray(X.dot(self.w)).ravel() + self.b
        margins = y * scores
        hinge_losses = np.maximum(0, 1 - margins)
        reg = (self.lambda_ / 2.0) * np.sum(self.w ** 2)
        return reg + np.mean(hinge_losses)

    def fit(self, X, y):
        """
        X: scipy sparse matrix shape (n_samples, n_features)  ← KHÔNG cần toarray()
        y: numpy array shape (n_samples,), với giá trị {-1, +1}
        """
        start_time = time.time()

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)

            if self.shuffle:
                rng.shuffle(indices)

            for idx in indices:
                # ✅ Chỉ dense hoá 1 hàng tại một thời điểm → không tốn RAM
                x_i = np.asarray(X[idx].todense()).ravel()
                y_i = y[idx]

                # 1) Tính score
                z_i = np.dot(self.w, x_i) + self.b

                # 2) Tính margin
                margin_i = y_i * z_i

                # 3) Tính subgradient của objective one-sample:
                #    J_i = (lambda_/2)||w||^2 + max(0, 1 - y_i(w^Tx_i+b))
                if margin_i >= 1:
                    # Không vi phạm margin → chỉ có gradient regularization
                    grad_w = self.lambda_ * self.w
                    grad_b = 0.0
                else:
                    # Vi phạm margin → gradient regularization + hinge
                    grad_w = self.lambda_ * self.w - y_i * x_i
                    grad_b = -y_i

                # 4) Cập nhật tham số
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b

            # Theo dõi objective full sau mỗi epoch
            epoch_loss = self._hinge_loss_full(X, y)
            self.loss_history.append(float(epoch_loss))

            if epoch % 1 == 0:
                print(f"[One-Sample SVM] Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss:.6f}")

        end_time = time.time()
        self.training_time_sec = round(end_time - start_time, 2)
        return self

    def decision_function(self, X):
        # ✅ Sparse-safe
        return np.asarray(X.dot(self.w)).ravel() + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


def encode_target_for_svm(y_series):
    """
    Map target:
        0 -> -1
        4 -> +1
    """
    return y_series.map({0: -1, 4: 1}).astype(int)


def train_svm_one_sample_model(X_train_text, y_train, learning_rate=0.01, lambda_=0.0001, epochs=5):
    """
    Huấn luyện trọn gói:
    - TF-IDF
    - SVM One Sample from scratch
    """
    print("- Đang vector hóa TF-IDF cho One-Sample SVM...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    # ✅ KHÔNG gọi .toarray() → giữ nguyên sparse để tiết kiệm RAM

    y_train_svm = encode_target_for_svm(y_train).to_numpy()

    print(f"- Kích thước ma trận train: {X_train_tfidf.shape}")

    model = LinearSVMOneSample(
        learning_rate=learning_rate,
        lambda_=lambda_,
        epochs=epochs,
        shuffle=True,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train_svm)
    return model, vectorizer


def evaluate_svm_one_sample(model, vectorizer, X_eval_text, y_eval):
    X_eval_tfidf = vectorizer.transform(X_eval_text)
    # ✅ KHÔNG gọi .toarray() → giữ nguyên sparse

    y_eval_svm = encode_target_for_svm(y_eval).to_numpy()
    y_pred_svm = model.predict(X_eval_tfidf)

    # Trả lại nhãn theo format gốc project: -1/+1 -> 0/4
    y_pred_project = np.where(y_pred_svm == 1, 4, 0)
    y_true_project = np.where(y_eval_svm == 1, 4, 0)

    accuracy = (y_pred_project == y_true_project).mean()
    print(f"[One-Sample SVM] Validation Accuracy: {accuracy * 100:.2f}%")

    return y_true_project, y_pred_project


def save_model(model, vectorizer, file_path):
    model_data = {
        "w": model.w,
        "b": model.b,
        "learning_rate": model.learning_rate,
        "lambda_": model.lambda_,
        "epochs": model.epochs,
        "loss_history": model.loss_history,
        "training_time_sec": model.training_time_sec,
        "vectorizer": vectorizer
    }
    with open(file_path, "wb") as f:
        pickle.dump(model_data, f)


def load_model(file_path):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)

    model = LinearSVMOneSample(
        learning_rate=model_data["learning_rate"],
        lambda_=model_data["lambda_"],
        epochs=model_data["epochs"]
    )
    model.w = model_data["w"]
    model.b = model_data["b"]
    model.loss_history = model_data["loss_history"]
    model.training_time_sec = model_data.get("training_time_sec", 0.0)

    vectorizer = model_data["vectorizer"]
    return model, vectorizer


if __name__ == "__main__":
    print("- BẮT ĐẦU HUẤN LUYỆN SVM ONE-SAMPLE (FROM SCRATCH) 🚀")

    df_train, df_val, df_test = get_clean_datasets_for_training()

    X_train = df_train["text"].fillna("")
    y_train = df_train["target"]

    X_val = df_val["text"].fillna("")
    y_val = df_val["target"]

    model, vectorizer = train_svm_one_sample_model(
        X_train_text=X_train,
        y_train=y_train,
        learning_rate=0.005,
        lambda_= 5e-6,
        epochs=40
    )

    evaluate_svm_one_sample(model, vectorizer, X_val, y_val)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "svm_one_sample_custom.pkl")
    save_model(model, vectorizer, save_path)

    print(f"✅ Đã lưu model One-Sample tại: {save_path}")
    print("🎉 HOÀN THÀNH!")