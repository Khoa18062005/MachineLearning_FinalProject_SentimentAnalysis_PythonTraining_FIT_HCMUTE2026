import os
import time
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid

from app.services.ml_service import get_clean_datasets_for_training
from app.core.pandas_helper import setup_pandas_display

setup_pandas_display()

# =========================================================
# GRID SEARCH SETTINGS
# =========================================================
# Để 1 cho an toàn RAM trên Windows.
# Nếu máy mạnh hơn nhiều RAM, có thể tăng lên 2 hoặc -1.
GRID_N_JOBS = 2
GRID_CV_FOLDS = 3
GRID_VERBOSE = 2
GRID_SCORING = "accuracy"


# =========================================================
# PIPELINES
# =========================================================
def build_word_only_pipeline():
    """
    Pipeline chỉ dùng word TF-IDF + LinearSVC
    """
    return Pipeline([
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="word",
                use_idf=True,
                smooth_idf=True,
                norm="l2"
            )
        ),
        (
            "svm",
            LinearSVC(random_state=42)
        )
    ])


def build_word_char_pipeline():
    """
    Pipeline dùng FeatureUnion:
    - word TF-IDF
    - char TF-IDF
    - LinearSVC
    """
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        use_idf=True,
        smooth_idf=True,
        norm="l2"
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        use_idf=True,
        smooth_idf=True,
        norm="l2"
    )

    feature_union = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer)
    ])

    return Pipeline([
        ("vectorizer", feature_union),
        ("svm", LinearSVC(random_state=42))
    ])


# =========================================================
# PARAM GRIDS
# =========================================================
def get_word_only_param_grid():
    """
    Grid exhaustive cho word-only.
    GridSearchCV sẽ thử toàn bộ tổ hợp ở đây.
    """
    return {
        "vectorizer__max_features": [80000, 100000, 120000],
        "vectorizer__ngram_range": [(1, 2), (1, 3)],
        "vectorizer__min_df": [2, 3],
        "vectorizer__max_df": [0.95, 0.98],
        "vectorizer__sublinear_tf": [True],
        "svm__C": [0.5, 1.0, 1.5, 2.0, 3.0],
        "svm__class_weight": [None],
        "svm__max_iter": [20000],
        "svm__tol": [1e-4]
    }


def get_word_char_param_grid():
    """
    Grid exhaustive cho word + char union.
    GridSearchCV sẽ thử toàn bộ tổ hợp ở đây.
    """
    return {
        "vectorizer__word_tfidf__max_features": [80000, 100000],
        "vectorizer__word_tfidf__ngram_range": [(1, 2)],
        "vectorizer__word_tfidf__min_df": [2],
        "vectorizer__word_tfidf__max_df": [0.95],
        "vectorizer__word_tfidf__sublinear_tf": [True],

        "vectorizer__char_tfidf__max_features": [10000, 20000],
        "vectorizer__char_tfidf__ngram_range": [(3, 5), (3, 6)],
        "vectorizer__char_tfidf__min_df": [2, 3],
        "vectorizer__char_tfidf__max_df": [1.0],
        "vectorizer__char_tfidf__sublinear_tf": [True],

        "svm__C": [1.0, 1.5, 2.0],
        "svm__class_weight": [None],
        "svm__max_iter": [20000],
        "svm__tol": [1e-4]
    }


# =========================================================
# GRID SEARCH HELPERS
# =========================================================
def print_grid_summary(grid_name, param_grid):
    total_combinations = len(list(ParameterGrid(param_grid)))
    print("\n" + "=" * 100)
    print(f"GRID SEARCH: {grid_name}")
    print(f"- Số tổ hợp tham số: {total_combinations}")
    print(f"- CV folds          : {GRID_CV_FOLDS}")
    print(f"- Tổng số lần fit   : {total_combinations * GRID_CV_FOLDS}")
    print(f"- n_jobs            : {GRID_N_JOBS}")
    print("=" * 100)


def run_grid_search(grid_name, pipeline, param_grid, X_train_cv, y_train_cv):
    """
    Chạy 1 GridSearchCV hoàn chỉnh cho một pipeline.
    """
    print_grid_summary(grid_name, param_grid)

    cv = StratifiedKFold(
        n_splits=GRID_CV_FOLDS,
        shuffle=True,
        random_state=42
    )

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=GRID_SCORING,
        cv=cv,
        n_jobs=GRID_N_JOBS,
        verbose=GRID_VERBOSE,
        refit=True,
        return_train_score=True,
        error_score="raise"
    )

    grid_search.fit(X_train_cv, y_train_cv)

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    print("\n" + "#" * 100)
    print(f"✅ HOÀN THÀNH GRID SEARCH: {grid_name}")
    print(f"- Best CV Accuracy : {grid_search.best_score_ * 100:.4f}%")
    print(f"- Training Time    : {duration}s")
    print("- Best Params:")
    for k, v in grid_search.best_params_.items():
        print(f"    {k}: {v}")
    print("#" * 100)

    return grid_search, duration


def save_grid_results_csv(grid_search, output_path):
    """
    Lưu toàn bộ kết quả CV ra CSV để tiện xem lại.
    """
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(
        by=["rank_test_score", "mean_test_score"],
        ascending=[True, False]
    )
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Đã lưu kết quả grid search tại: {output_path}")
    return results_df


def extract_model_and_vectorizer_from_grid(grid_search, training_time_sec):
    """
    Tách model + vectorizer từ best_estimator_ để tương thích endpoint hiện tại.
    Endpoint của bạn đang load model và vectorizer riêng.
    """
    best_pipeline = grid_search.best_estimator_
    best_vectorizer = best_pipeline.named_steps["vectorizer"]
    best_model = best_pipeline.named_steps["svm"]

    # Gắn training time để endpoint đọc được
    best_model.training_time_sec = training_time_sec

    return best_model, best_vectorizer


def evaluate_svm_library_model(model, vectorizer, X_eval, y_eval, label_name="Evaluation"):
    X_eval_tfidf = vectorizer.transform(X_eval)
    y_pred = model.predict(X_eval_tfidf)

    accuracy = (y_pred == y_eval).mean()
    print(f"[SVM Library] {label_name} Accuracy: {accuracy * 100:.4f}%")

    print(f"\n[SVM Library] {label_name} Classification Report:")
    print(classification_report(y_eval, y_pred))

    return y_eval.to_numpy(), y_pred


def choose_best_search(word_grid_search, word_time, union_grid_search, union_time):
    """
    So sánh best CV score của 2 không gian tìm kiếm.
    """
    if union_grid_search is None:
        return "word_only", word_grid_search, word_time

    if word_grid_search.best_score_ >= union_grid_search.best_score_:
        return "word_only", word_grid_search, word_time
    else:
        return "word_char_union", union_grid_search, union_time


def save_best_summary_txt(output_path, best_name, best_grid_search, selected_search_time, total_search_time):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("BEST RESULT - SVM LIBRARY GRID SEARCH\n")
        f.write("=" * 80 + "\n")
        f.write(f"Best Search Space     : {best_name}\n")
        f.write(f"Best CV Accuracy      : {best_grid_search.best_score_ * 100:.4f}%\n")
        f.write(f"Selected Search Time  : {selected_search_time}s\n")
        f.write(f"Total Search Time     : {total_search_time}s\n")
        f.write("Best Params:\n")
        for k, v in best_grid_search.best_params_.items():
            f.write(f"  - {k}: {v}\n")

    print(f"✅ Đã lưu best summary tại: {output_path}")


# =========================================================
# MAIN TRAIN FUNCTION
# =========================================================
def train_svm_library_model_with_gridsearch():
    """
    Chiến lược:
    - Lấy train / val / test từ project
    - Gộp train + val để chạy GridSearchCV
    - Dùng Stratified K-Fold CV để tune
    - Giữ test hoàn toàn độc lập để đánh giá cuối
    """
    print("- BẮT ĐẦU HUẤN LUYỆN SVM LIBRARY BẰNG GRID SEARCH CV 🚀")

    df_train, df_val, df_test = get_clean_datasets_for_training()

    X_train = df_train["text"].fillna("")
    y_train = df_train["target"]

    X_val = df_val["text"].fillna("")
    y_val = df_val["target"]

    X_test = df_test["text"].fillna("")
    y_test = df_test["target"]

    # Gộp train + val để tối đa dữ liệu cho GridSearchCV
    X_train_cv = pd.concat([X_train, X_val], axis=0)
    y_train_cv = pd.concat([y_train, y_val], axis=0)

    print(f"- Số mẫu train ban đầu : {len(X_train)}")
    print(f"- Số mẫu val ban đầu   : {len(X_val)}")
    print(f"- Số mẫu test giữ riêng: {len(X_test)}")
    print(f"- Số mẫu dùng GridSearchCV (train + val): {len(X_train_cv)}")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # =====================================================
    # 1) WORD ONLY GRID SEARCH
    # =====================================================
    word_pipeline = build_word_only_pipeline()
    word_param_grid = get_word_only_param_grid()

    word_grid_search, word_time = run_grid_search(
        grid_name="WORD ONLY",
        pipeline=word_pipeline,
        param_grid=word_param_grid,
        X_train_cv=X_train_cv,
        y_train_cv=y_train_cv
    )

    word_csv_path = os.path.join(current_dir, "svm_library_gridsearch_word_only_results.csv")
    save_grid_results_csv(word_grid_search, word_csv_path)

    # =====================================================
    # 2) WORD + CHAR GRID SEARCH
    # =====================================================
    union_pipeline = build_word_char_pipeline()
    union_param_grid = get_word_char_param_grid()

    union_grid_search, union_time = run_grid_search(
        grid_name="WORD + CHAR UNION",
        pipeline=union_pipeline,
        param_grid=union_param_grid,
        X_train_cv=X_train_cv,
        y_train_cv=y_train_cv
    )

    union_csv_path = os.path.join(current_dir, "svm_library_gridsearch_word_char_results.csv")
    save_grid_results_csv(union_grid_search, union_csv_path)

    # =====================================================
    # 3) CHỌN BEST SEARCH SPACE
    # =====================================================
    best_name, best_grid_search, selected_search_time = choose_best_search(
        word_grid_search=word_grid_search,
        word_time=word_time,
        union_grid_search=union_grid_search,
        union_time=union_time
    )

    total_search_time = round(word_time + union_time, 2)

    print("\n" + "#" * 100)
    print("🏆 BEST RESULT - SVM LIBRARY GRID SEARCH")
    print(f"- Best Search Space : {best_name}")
    print(f"- Best CV Accuracy  : {best_grid_search.best_score_ * 100:.4f}%")
    print(f"- Search Time       : {selected_search_time}s")
    print(f"- Total Time        : {total_search_time}s")
    print("- Best Params:")
    for k, v in best_grid_search.best_params_.items():
        print(f"    {k}: {v}")
    print("#" * 100)

    # =====================================================
    # 4) TÁCH BEST MODEL + BEST VECTORIZER
    # GridSearchCV với refit=True đã fit lại best model
    # trên toàn bộ X_train_cv rồi
    # =====================================================
    # Gắn training_time_sec = tổng thời gian tune toàn bộ
    model, vectorizer = extract_model_and_vectorizer_from_grid(
        best_grid_search,
        training_time_sec=total_search_time
    )

    # =====================================================
    # 5) EVALUATE ON TEST
    # =====================================================
    print("\n--- ĐÁNH GIÁ CUỐI TRÊN TEST ---")
    y_true_test, y_pred_test = evaluate_svm_library_model(
        model=model,
        vectorizer=vectorizer,
        X_eval=X_test,
        y_eval=y_test,
        label_name="Test"
    )

    # =====================================================
    # 6) SAVE MODEL + VECTORIZER
    # =====================================================
    model_path = os.path.join(current_dir, "svm_library_model.pkl")
    vectorizer_path = os.path.join(current_dir, "svm_library_vectorizer.pkl")
    summary_path = os.path.join(current_dir, "svm_library_best_summary.txt")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"✅ Đã lưu model Library tại: {model_path}")
    print(f"✅ Đã lưu vectorizer tại: {vectorizer_path}")

    save_best_summary_txt(
        output_path=summary_path,
        best_name=best_name,
        best_grid_search=best_grid_search,
        selected_search_time=selected_search_time,
        total_search_time=total_search_time
    )

    return model, vectorizer, {
        "best_search_space": best_name,
        "best_cv_score": float(best_grid_search.best_score_),
        "best_params": best_grid_search.best_params_,
        "selected_search_time_sec": selected_search_time,
        "total_search_time_sec": total_search_time,
        "y_true_test": y_true_test,
        "y_pred_test": y_pred_test,
    }


if __name__ == "__main__":
    model, vectorizer, best_info = train_svm_library_model_with_gridsearch()

    print("\n🎉 HOÀN THÀNH GRID SEARCH CHO SVM LIBRARY!")
    print(f"- Best Search Space: {best_info['best_search_space']}")
    print(f"- Best CV Score    : {best_info['best_cv_score'] * 100:.4f}%")
    print("- Best Params:")
    for k, v in best_info["best_params"].items():
        print(f"    {k}: {v}")