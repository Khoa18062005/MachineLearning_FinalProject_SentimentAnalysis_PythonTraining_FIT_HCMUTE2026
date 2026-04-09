import pandas as pd
from nltk.lm import vocabulary
import pickle
import time
import os
from app.services.ml_service import get_clean_datasets_for_training
from app.core.pandas_helper import setup_pandas_display
from collections import defaultdict
setup_pandas_display()
laplace_smoothing = 1
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    return (model_data['probs_prior'],
            model_data['word_probs'],
            model_data['vocab'],
            model_data['total_words_class'],
            model_data['word_counts'],
            model_data.get('training_time_sec', 0.0))

def get_vocabulary(X_train):
    vocab = set()
    for text in X_train:
        words = text.split()
        vocab.update(words)
    return list(vocab)

def count_words(X_train, y_train):
    word_counts_class = {}
    total_words_class = {}
    classes = y_train.unique()
    for c in classes:
        word_counts_class[c] = defaultdict(int)
        total_words_class[c] = 0
    for text, label in zip(X_train, y_train):
        words = str(text).split()
        for word in words:
            word_counts_class[label][word] += 1
            total_words_class[label] += 1
    return word_counts_class, total_words_class

def calculate_word_probs(word_counts_class, total_words_class, vocabulary: list):
    word_probs = {}
    v_len = len(vocabulary)
    for c in word_counts_class.keys():
        word_probs[c] = {}
        N_c = total_words_class[c]
        for word in vocabulary:
            count_w_c = word_counts_class[c][word]
            prob = (count_w_c + laplace_smoothing) / (N_c + v_len)
            word_probs[c][word] = prob
    return word_probs

# Hàm huấn luyện mô hình
def train_MNB_Custom(X_train, y_train):
    start_time = time.time()
    # 1. Tạo chuỗi không trùng lập
    vocab = get_vocabulary(X_train)

    # 2. Đếm số lần xuất hiện của từng từ
    word_counts_class, total_words_class = count_words(X_train, y_train)

    # 3. Tính xác suất tiên nghiệm P(y)
    probs_prior = {}
    class_counts = y_train.value_counts()
    for c in class_counts.index:
        probs_prior[c] = class_counts[c] / len(y_train)

    # 4. Tính xác suất của từng đặc trưng P(wi|c)
    word_probs = calculate_word_probs(word_counts_class, total_words_class, vocab)

    # Kết thúc bấm giờ
    end_time = time.time()
    training_duration = round(end_time - start_time, 2)

    # Lưu models
    model_data = {
        'vocab': vocab,
        'probs_prior': probs_prior,
        'word_probs': word_probs,
        'word_counts': word_counts_class,
        'total_words_class': total_words_class,
        'training_time_sec': training_duration
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "MNB_model_custom.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    return model_data

def predict_MNB_Custom(text_input, probs_prior, word_probs, vocab, total_words_class):
    words = str(text_input).split()
    v_len = len(vocab)
    results = {}

    for c in probs_prior.keys():
        # Bắt đầu với log hoặc xác suất tiên nghiệm P(Y)
        score = probs_prior[c]
        N_c = total_words_class[c]

        for word in words:
            if word in word_probs[c]:
                # Nếu từ đã học: Tra bảng lấy xác suất
                score *= word_probs[c][word]
            else:
                # Nếu từ mới (OOV): Tính xác suất mặc định theo Laplace
                score *= laplace_smoothing / (N_c + (laplace_smoothing * v_len))

        results[c] = score

    # Trả về nhãn có điểm cao nhất
    prediction = max(results, key=results.get)
    return prediction, results

def get_prediction_details(text_input, probs_prior, word_probs, vocab, total_words_class, word_counts):
    words = str(text_input).split()
    v_len = len(vocab)
    details = {}

    for c in probs_prior.keys():
        class_details = {
            "prior": probs_prior[c],
            "total_words_in_class": total_words_class[c],
            "vocab_size": v_len,
            "word_steps": []
        }

        current_score = probs_prior[c]
        for word in words:
            # Lấy xác suất và số lần xuất hiện trực tiếp (không tính ngược nữa)
            default_prob = laplace_smoothing / (total_words_class[c] + (laplace_smoothing * v_len))
            prob = word_probs[c].get(word, default_prob)
            count_w_c = word_counts[c].get(word, 0)

            class_details["word_steps"].append({
                "word": word,
                "count_w_c": count_w_c, # Số liệu thực tế từ lúc train
                "prob": prob
            })
            current_score *= prob

        class_details["final_score"] = current_score
        details[str(c)] = class_details # Đảm bảo key là string để JSON không lỗi

    return details

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

    # Huấn luyện mô hình
    model_metadata = train_MNB_Custom(X_train, y_train)

    # Tải mô hình
    priors, w_probs, vocab, totals, counts, train_time = load_model("MNB_model_custom.pkl")
    print(f"Thời gian huấn luyện load từ file: {train_time} giây")