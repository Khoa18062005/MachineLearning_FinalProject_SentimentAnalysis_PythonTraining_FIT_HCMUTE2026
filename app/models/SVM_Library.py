import os
import time
import joblib
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


from app.services.ml_service import get_clean_datasets_for_training
from app.core.pandas_helper import setup_pandas_display

setup_pandas_display()


def build_feature_union(config):
    """
    Tạo bộ vectorizer kết hợp:
    - Word TF-IDF
    - Char TF-IDF

    Ưu điểm:
    - Word nắm nghĩa từ/cụm từ
    - Char nắm lỗi chính tả, kéo chữ, biến thể từ, slang
    - FeatureUnion vẫn có .transform(), nên tương thích endpoint cũ
    """
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        max_features=config["word_max_features"],
        ngram_range=config["word_ngram_range"],
        min_df=config["word_min_df"],
        max_df=config["word_max_df"],
        sublinear_tf=config["sublinear_tf"],
        use_idf=True,
        smooth_idf=True,
        norm="l2"
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        max_features=config["char_max_features"],
        ngram_range=config["char_ngram_range"],
        min_df=config["char_min_df"],
        max_df=1.0,
        sublinear_tf=config["sublinear_tf"],
        use_idf=True,
        smooth_idf=True,
        norm="l2"
    )

    feature_union = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer)
    ])

    return feature_union


def train_single_config(X_train, y_train, X_val, y_val, config, config_index=None, total_configs=None):
    """
    Train 1 cấu hình và trả về kết quả.
    """
    print("\n" + "=" * 100)

    if config_index is not None and total_configs is not None:
        print(f"ĐANG THỬ CẤU HÌNH [{config_index}/{total_configs}]")
    else:
        print("ĐANG THỬ CẤU HÌNH")

    for k, v in config.items():
        print(f"- {k}: {v}")

    print("=" * 100)

    start_time = time.time()

    print("- Đang xây dựng FeatureUnion (word + char TF-IDF)...")
    vectorizer = build_feature_union(config)

    print("- Đang fit TF-IDF trên tập train...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print(f"- Shape train TF-IDF: {X_train_tfidf.shape}")
    print(f"- Shape val   TF-IDF: {X_val_tfidf.shape}")

    print("- Đang huấn luyện LinearSVC...")
    model = LinearSVC(
        C=config["C"],
        class_weight=config["class_weight"],
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_val_tfidf)
    accuracy = (y_pred == y_val).mean()

    end_time = time.time()
    training_duration = round(end_time - start_time, 2)
    model.training_time_sec = training_duration

    print(f"✅ Validation Accuracy: {accuracy * 100:.4f}%")
    print(f"⏱ Training Time      : {training_duration}s")

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": float(accuracy),
        "training_time_sec": training_duration,
        "config": config,
        "y_pred": y_pred
    }


def train_svm_library_model(X_train, y_train, X_val, y_val):
    """
    Huấn luyện SVM Library bản nâng cấp có tune nhiều cấu hình.
    Sau cùng trả về best model + best vectorizer.

    Giữ nguyên target project:
        0 = negative
        4 = positive
    """
    configs = [
        {
            "word_max_features": 60000,
            "word_ngram_range": (1, 2),
            "word_min_df": 2,
            "word_max_df": 0.95,
            "char_max_features": 30000,
            "char_ngram_range": (3, 5),
            "char_min_df": 2,
            "sublinear_tf": True,
            "C": 0.5,
            "class_weight": None,
            "max_iter": 10000,
            "tol": 1e-4
        },
        {
            "word_max_features": 80000,
            "word_ngram_range": (1, 2),
            "word_min_df": 2,
            "word_max_df": 0.95,
            "char_max_features": 50000,
            "char_ngram_range": (3, 5),
            "char_min_df": 2,
            "sublinear_tf": True,
            "C": 1.0,
            "class_weight": None,
            "max_iter": 12000,
            "tol": 1e-4
        },
        {
            "word_max_features": 100000,
            "word_ngram_range": (1, 2),
            "word_min_df": 2,
            "word_max_df": 0.95,
            "char_max_features": 60000,
            "char_ngram_range": (3, 5),
            "char_min_df": 2,
            "sublinear_tf": True,
            "C": 2.0,
            "class_weight": None,
            "max_iter": 15000,
            "tol": 1e-4
        },
        {
            "word_max_features": 80000,
            "word_ngram_range": (1, 3),
            "word_min_df": 2,
            "word_max_df": 0.95,
            "char_max_features": 50000,
            "char_ngram_range": (3, 5),
            "char_min_df": 2,
            "sublinear_tf": True,
            "C": 1.0,
            "class_weight": None,
            "max_iter": 15000,
            "tol": 1e-4
        },
        {
            "word_max_features": 80000,
            "word_ngram_range": (1, 2),
            "word_min_df": 3,
            "word_max_df": 0.98,
            "char_max_features": 50000,
            "char_ngram_range": (3, 6),
            "char_min_df": 3,
            "sublinear_tf": True,
            "C": 1.5,
            "class_weight": None,
            "max_iter": 15000,
            "tol": 1e-4
        }
    ]

    best_result = None

    for idx, config in enumerate(configs, start=1):
        result = train_single_config(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            config_index=idx,
            total_configs=len(configs)
        )

        if best_result is None or result["accuracy"] > best_result["accuracy"]:
            best_result = result
            print("\n🌟 ĐÃ CẬP NHẬT BEST MODEL")
            print(f"- Best Accuracy hiện tại: {best_result['accuracy'] * 100:.4f}%")
            print(f"- Best Config hiện tại: {best_result['config']}")

    best_result["model"].training_time_sec = best_result["training_time_sec"]
    return best_result["model"], best_result["vectorizer"], best_result


def evaluate_svm_library_model(model, vectorizer, X_eval, y_eval):
    X_eval_tfidf = vectorizer.transform(X_eval)
    y_pred = model.predict(X_eval_tfidf)

    accuracy = (y_pred == y_eval).mean()
    print(f"[SVM Library] Validation Accuracy: {accuracy * 100:.4f}%")

    print("\n[SVM Library] Classification Report:")
    print(classification_report(y_eval, y_pred))

    return y_eval.to_numpy(), y_pred


if __name__ == "__main__":
    print("- BẮT ĐẦU HUẤN LUYỆN SVM LIBRARY 🚀")

    df_train, df_val, df_test = get_clean_datasets_for_training()

    X_train = df_train["text"].fillna("")
    y_train = df_train["target"]

    X_val = df_val["text"].fillna("")
    y_val = df_val["target"]

    model, vectorizer, best_result = train_svm_library_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )

    print("\n" + "#" * 100)
    print("🏆 BEST RESULT - SVM LIBRARY")
    print(f"- Best Validation Accuracy: {best_result['accuracy'] * 100:.4f}%")
    print(f"- Best Training Time      : {best_result['training_time_sec']}s")
    print("- Best Config:")
    for k, v in best_result["config"].items():
        print(f"    {k}: {v}")
    print("#" * 100)

    evaluate_svm_library_model(model, vectorizer, X_val, y_val)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "svm_library_model.pkl")
    vectorizer_path = os.path.join(current_dir, "svm_library_vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"✅ Đã lưu model Library tại: {model_path}")
    print(f"✅ Đã lưu vectorizer tại: {vectorizer_path}")
    print("🎉 HOÀN THÀNH!")