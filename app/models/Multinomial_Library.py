import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import từ các module của bạn
from app.services.ml_service import get_clean_datasets_for_training
from app.core.pandas_helper import setup_pandas_display
from app.core.config import settings

setup_pandas_display()


def train_mnb_model(X_train, y_train):
    # --- BẮT ĐẦU ĐO THỜI GIAN HUẤN LUYỆN ---
    start_time = time.time()
    print("- Đang Vector hóa dữ liệu tập Train (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"- Đã tạo ma trận đặc trưng: {X_train_tfidf.shape}")

    print("- Đang huấn luyện mô hình Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)

    # --- KẾT THÚC ĐO ---
    end_time = time.time()
    training_duration = round(end_time - start_time, 2)

    # Gắn thẳng thời gian vào object model trước khi return
    model.training_time_sec = training_duration

    print(f"- Huấn luyện xong trong {training_duration} giây!")

    return model, vectorizer


def evaluate_model(model, vectorizer, X_eval, y_eval, dataset_name="Validation"):
    print(f"\n- Đang đánh giá mô hình trên tập {dataset_name}...")
    # Chỉ transform, KHÔNG fit lại trên tập đánh giá
    X_eval_tfidf = vectorizer.transform(X_eval)

    # Dự đoán
    y_pred = model.predict(X_eval_tfidf)

    # Tính toán và in kết quả
    acc = accuracy_score(y_eval, y_pred)
    print(f"📊 Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print("📈 Báo cáo chi tiết (Classification Report):")
    print(classification_report(y_eval, y_pred))

    return acc, y_pred

if __name__ == "__main__":
    print("- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MULTINOMIAL NAIVE BAYES 🚀\n")

    # 1. Lấy dữ liệu sạch từ ml_service.py
    print("Đang tải dữ liệu...")
    df_train, df_val, df_test = get_clean_datasets_for_training()

    # Xử lý các dòng rỗng do quá trình làm sạch sinh ra
    X_train = df_train['text'].fillna('')
    y_train = df_train['target']

    X_val = df_val['text'].fillna('')
    y_val = df_val['target']

    # 2. Gọi hàm huấn luyện độc lập
    trained_model, tfidf_vectorizer = train_mnb_model(X_train, y_train)

    # 3. Gọi hàm đánh giá độc lập
    evaluate_model(trained_model, tfidf_vectorizer, X_val, y_val, dataset_name="Validation")

    # 4. Lưu mô hình và vectorizer
    print("\n- Đang lưu mô hình vào hệ thống...")
    os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)

    joblib.dump(trained_model, settings.MODEL_PATH)

    vectorizer_path = settings.MODEL_PATH.replace('.pkl', '_vectorizer.pkl')
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    print(f"✅ Đã lưu Model tại: {settings.MODEL_PATH}")
    print(f"✅ Đã lưu Vectorizer tại: {vectorizer_path}")
    print("🎉 KẾT THÚC THÀNH CÔNG!")