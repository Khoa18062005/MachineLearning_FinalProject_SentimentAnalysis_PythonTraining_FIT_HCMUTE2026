import joblib # Dùng joblib để lưu model sklearn hiệu quả hơn pickle
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from app.core.config import settings
from app.services.ml_service import get_clean_datasets_for_training


def train_MNB_Library(X_train, y_train):
    start_time = time.time()

    # 1. Tạo Pipeline: Kết hợp Vectorizer và Model thành một quy trình duy nhất
    # CountVectorizer thay thế cho get_vocabulary và count_words của bạn
    # MultinomialNB thay thế cho calculate_word_probs và probs_prior
    model_pipeline = Pipeline([
        ('vectorizer', CountVectorizer()), 
        ('nb', MultinomialNB(alpha=1.0)) # alpha=1.0 chính là Laplace Smoothing
    ])

    # 2. Huấn luyện
    model_pipeline.fit(X_train, y_train)

    end_time = time.time()
    training_duration = round(end_time - start_time, 2)

    # 3. Lưu mô hình
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_library.pkl")
    joblib.dump({
        'pipeline': model_pipeline,
        'training_time': training_duration
    }, model_path)

    return training_duration

def load_library_model():
    model_path = os.path.join(settings.BASE_DIR, "app", "models", "MNB_model_library.pkl")
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        return data['pipeline'], data['training_time']
    return None, 0
if __name__ == "__main__":
    # Lấy dữ liệu sạch từ ml_service.py
    df_train, df_val, df_test = get_clean_datasets_for_training()

    # Phân chia feature và target
    X_train = df_train['text']
    y_train = df_train['target']
    X_val = df_val['text']
    y_val = df_val['target']
    X_test = df_test['text']
    y_test = df_test['target']
    model_metadata = train_MNB_Library(X_train, y_train)