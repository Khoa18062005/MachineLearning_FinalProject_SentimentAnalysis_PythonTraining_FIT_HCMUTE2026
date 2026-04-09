import os
import time
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.ml_service import get_clean_datasets_for_training
from app.core.pandas_helper import setup_pandas_display

setup_pandas_display()


class LinearSVMFullSample:
    """
    Linear SVM from scratch - huấn luyện theo cơ chế full-batch / full-sample

    Objective:
        J(w, b) = (lambda_/2)||w||^2 + (1/n) * sum_i max(0, 1 - y_i(w^T x_i + b))

    Bản nâng cấp:
    - Full-batch gradient chuẩn
    - Dùng Adam optimizer from scratch để tối ưu ổn hơn
    - Có early stopping
    - Có lưu best weights theo loss tốt nhất
    """

    def __init__(
        self,
        learning_rate=0.05,
        lambda_=0.00001,
        epochs=300,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        patience=20,
        min_delta=1e-5
    ):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.epochs = epochs

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.patience = patience
        self.min_delta = min_delta

        self.w = None
        self.b = 0.0

        self.loss_history = []
        self.training_time_sec = 0.0

        self.best_loss = float("inf")
        self.best_w = None
        self.best_b = 0.0

    def _objective(self, X, y):
        """
        Tính objective full:
            J(w, b) = (lambda_/2)||w||^2 + mean(hinge)
        """
        scores = np.asarray(X.dot(self.w)).ravel() + self.b
        margins = y * scores
        hinge_losses = np.maximum(0.0, 1.0 - margins)
        reg = (self.lambda_ / 2.0) * np.sum(self.w ** 2)
        return float(reg + np.mean(hinge_losses))

    def _compute_full_gradient(self, X, y):
        """
        Tính full-batch gradient cho toàn bộ tập.
        """
        n_samples = X.shape[0]

        scores = np.asarray(X.dot(self.w)).ravel() + self.b
        margins = y * scores
        violating_mask = margins < 1

        grad_w = self.lambda_ * self.w
        grad_b = 0.0

        if np.any(violating_mask):
            X_violate = X[violating_mask]
            y_violate = y[violating_mask]

            # Sparse-safe:
            # sum(y_i * x_i) trên các sample vi phạm margin
            weighted_X = X_violate.multiply(y_violate[:, None])
            grad_w -= (1.0 / n_samples) * np.asarray(weighted_X.sum(axis=0)).ravel()
            grad_b -= (1.0 / n_samples) * np.sum(y_violate)

        return grad_w, grad_b

    def fit(self, X, y):
        start_time = time.time()

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        # Adam states cho w
        m_w = np.zeros(n_features, dtype=np.float64)
        v_w = np.zeros(n_features, dtype=np.float64)

        # Adam states cho b
        m_b = 0.0
        v_b = 0.0

        no_improve_count = 0

        for epoch in range(1, self.epochs + 1):
            # 1) Tính gradient full-batch
            grad_w, grad_b = self._compute_full_gradient(X, y)

            # 2) Adam update cho w
            m_w = self.beta1 * m_w + (1.0 - self.beta1) * grad_w
            v_w = self.beta2 * v_w + (1.0 - self.beta2) * (grad_w ** 2)

            m_w_hat = m_w / (1.0 - self.beta1 ** epoch)
            v_w_hat = v_w / (1.0 - self.beta2 ** epoch)

            self.w -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.eps)

            # 3) Adam update cho b
            m_b = self.beta1 * m_b + (1.0 - self.beta1) * grad_b
            v_b = self.beta2 * v_b + (1.0 - self.beta2) * (grad_b ** 2)

            m_b_hat = m_b / (1.0 - self.beta1 ** epoch)
            v_b_hat = v_b / (1.0 - self.beta2 ** epoch)

            self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

            # 4) Theo dõi loss
            epoch_loss = self._objective(X, y)
            self.loss_history.append(epoch_loss)

            print(
                f"[Full-Sample SVM] Epoch {epoch}/{self.epochs} "
                f"- loss: {epoch_loss:.6f}"
            )

            # 5) Lưu best weights
            if epoch_loss < self.best_loss - self.min_delta:
                self.best_loss = epoch_loss
                self.best_w = self.w.copy()
                self.best_b = float(self.b)
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 6) Early stopping
            if no_improve_count >= self.patience:
                print(f"[Full-Sample SVM] Early stopping tại epoch {epoch}")
                break

        # 7) Khôi phục bộ trọng số tốt nhất
        if self.best_w is not None:
            self.w = self.best_w.copy()
            self.b = float(self.best_b)

        end_time = time.time()
        self.training_time_sec = round(end_time - start_time, 2)
        return self

    def decision_function(self, X):
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


def train_svm_full_sample_model(
    X_train_text,
    y_train,
    learning_rate=0.05,
    lambda_=0.00001,
    epochs=300
):
    """
    Huấn luyện trọn gói:
    - TF-IDF
    - SVM Full Sample from scratch
    """
    print("- Đang vector hóa TF-IDF cho Full-Sample SVM...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=80000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    # ✅ KHÔNG gọi .toarray() → giữ nguyên sparse để tiết kiệm RAM

    y_train_svm = encode_target_for_svm(y_train).to_numpy()

    print(f"- Kích thước ma trận train: {X_train_tfidf.shape}")

    model = LinearSVMFullSample(
        learning_rate=learning_rate,
        lambda_=lambda_,
        epochs=epochs,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        patience=25,
        min_delta=1e-5
    )

    model.fit(X_train_tfidf, y_train_svm)
    return model, vectorizer


def evaluate_svm_full_sample(model, vectorizer, X_eval_text, y_eval):
    X_eval_tfidf = vectorizer.transform(X_eval_text)
    # ✅ KHÔNG gọi .toarray() → giữ nguyên sparse

    y_eval_svm = encode_target_for_svm(y_eval).to_numpy()
    y_pred_svm = model.predict(X_eval_tfidf)

    # Trả lại nhãn theo format gốc project: -1/+1 -> 0/4
    y_pred_project = np.where(y_pred_svm == 1, 4, 0)
    y_true_project = np.where(y_eval_svm == 1, 4, 0)

    accuracy = (y_pred_project == y_true_project).mean()
    print(f"[Full-Sample SVM] Validation Accuracy: {accuracy * 100:.2f}%")

    return y_true_project, y_pred_project


def save_model(model, vectorizer, file_path):
    model_data = {
        "w": model.w,
        "b": model.b,
        "learning_rate": model.learning_rate,
        "lambda_": model.lambda_,
        "epochs": model.epochs,
        "beta1": model.beta1,
        "beta2": model.beta2,
        "eps": model.eps,
        "patience": model.patience,
        "min_delta": model.min_delta,
        "loss_history": model.loss_history,
        "training_time_sec": model.training_time_sec,
        "best_loss": model.best_loss,
        "vectorizer": vectorizer
    }

    with open(file_path, "wb") as f:
        pickle.dump(model_data, f)


def load_model(file_path):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)

    model = LinearSVMFullSample(
        learning_rate=model_data["learning_rate"],
        lambda_=model_data["lambda_"],
        epochs=model_data["epochs"],
        beta1=model_data.get("beta1", 0.9),
        beta2=model_data.get("beta2", 0.999),
        eps=model_data.get("eps", 1e-8),
        patience=model_data.get("patience", 50),
        min_delta=model_data.get("min_delta", 1e-8)
    )

    model.w = model_data["w"]
    model.b = model_data["b"]
    model.loss_history = model_data["loss_history"]
    model.training_time_sec = model_data.get("training_time_sec", 0.0)
    model.best_loss = model_data.get("best_loss", float("inf"))

    vectorizer = model_data["vectorizer"]
    return model, vectorizer


if __name__ == "__main__":
    print("- BẮT ĐẦU HUẤN LUYỆN SVM FULL-SAMPLE (FROM SCRATCH) 🚀")

    df_train, df_val, df_test = get_clean_datasets_for_training()

    X_train = df_train["text"].fillna("")
    y_train = df_train["target"]

    X_val = df_val["text"].fillna("")
    y_val = df_val["target"]

    model, vectorizer = train_svm_full_sample_model(
        X_train_text=X_train,
        y_train=y_train,
        learning_rate=0.05,
        lambda_=0.00001,
        epochs=1000
    )

    evaluate_svm_full_sample(model, vectorizer, X_val, y_val)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "svm_full_sample_custom.pkl")
    save_model(model, vectorizer, save_path)

    print(f"✅ Đã lưu model Full-Sample tại: {save_path}")
    print("🎉 HOÀN THÀNH!")
