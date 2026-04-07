import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from app.core.nlp_config import lemmatizer, CUSTOM_STOP_WORDS

COLUMNS = ["target", "id", "date", "flag", "user", "text"]
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data_training/Data_Emotion.csv")
limit_max = 200000
# Các tập dữ liệu chưa được tiền xử lý
GLOBAL_TRAIN = None
GLOBAL_VAL = None
GLOBAL_TEST = None
# Dữ liệu đã được tiền xử lý
GLOBAL_CLEAN_TRAIN = None
GLOBAL_CLEAN_VAL = None
GLOBAL_CLEAN_TEST = None

def _load_raw_df(limit=limit_max):
    return pd.read_csv(DATA_PATH, names=COLUMNS, nrows=limit, encoding='latin-1')

def get_total_sample(df):
    return f'{df.shape[0]}'

def get_data_preview(limit=limit_max):
    df = _load_raw_df(limit)
    return df.to_dict(orient="records")

def get_data_features(limit=limit_max):
    df = _load_raw_df(limit)
    # Tái sử dụng df và chỉ lọc cột cần thiết
    features = ['target', 'text']
    df = df[features]
    df['needs_processing'] = df['text'].apply(check_needs_processing)
    return df.to_dict(orient="records")

def check_needs_processing(text):
    """Kiểm tra xem văn bản có chứa các thành phần cần xử lý không"""
    text = str(text)
    # 1. Kiểm tra URL, đường link (http, https, www)
    if re.search(r"http\S+|www\S+|https\S+", text, re.IGNORECASE):
        return True
    # 2. Kiểm tra mention (@) hoặc hashtag (#)
    if re.search(r"[@#]\w+", text):
        return True
    # 3. Kiểm tra số
    if re.search(r"\d+", text):
        return True
    # 4. Kiểm tra ký tự đặc biệt, emoji (Tìm các ký tự KHÔNG phải là chữ cái (a-z, A-Z) và khoảng trắng)
    if re.search(r"[^a-zA-Z\s]", text):
        return True
    return False

def split_and_prepare_datasets(limit=limit_max):
    """Hàm cắt dữ liệu và lưu thẳng vào 3 biến toàn cục"""
    global GLOBAL_TRAIN, GLOBAL_VAL, GLOBAL_TEST

    # Nếu biến GLOBAL_TRAIN đã có dữ liệu (khác None) thì bỏ qua, không tính toán lại
    if GLOBAL_TRAIN is not None:
        return

    # 1. Tải dữ liệu và lấy 2 cột cần thiết
    df = _load_raw_df(limit)
    df = df[['target', 'text']]

    # 2. Cắt 20% làm tập Test, 80% còn lại là Train + Val
    df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Từ 80% trên, cắt lấy 10% làm Validation
    df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=42)

    # 4. Đánh dấu cờ 'needs_processing' cho cả 3 tập
    df_train['needs_processing'] = df_train['text'].apply(check_needs_processing)
    df_val['needs_processing'] = df_val['text'].apply(check_needs_processing)
    df_test['needs_processing'] = df_test['text'].apply(check_needs_processing)

    # 5. Gán kết quả vào 3 biến toàn cục để lưu trữ
    GLOBAL_TRAIN = df_train
    GLOBAL_VAL = df_val
    GLOBAL_TEST = df_test

def get_dataset_by_name(set_name="train", limit=limit_max):
    """Hàm tổng quát: trả tập dữ liệu theo yêu cầu Frontend"""
    # 1. Gọi hàm cắt dữ liệu (Nó sẽ tự kiểm tra xem đã cắt chưa)
    split_and_prepare_datasets(limit)

    # 2. Kiểm tra xem Frontend đang yêu cầu tập nào
    if set_name == "train":
        df_requested = GLOBAL_TRAIN
    elif set_name == "val":
        df_requested = GLOBAL_VAL
    elif set_name == "test":
        df_requested = GLOBAL_TEST
    else:
        df_requested = GLOBAL_TRAIN  # Mặc định nếu truyền sai thì trả về train

    # 3. Đếm số lượng mẫu của cả 3 tập để Frontend hiển thị
    counts = {
        "train_count": len(GLOBAL_TRAIN),
        "val_count": len(GLOBAL_VAL),
        "test_count": len(GLOBAL_TEST)
    }

    # 4. Trả kết quả về
    return {
        "records": df_requested.to_dict(orient="records"),
        "counts": counts
    }

def get_datasets_for_training(limit=limit_max):
    """Hàm này dùng ở Backend để lấy 3 tập dữ liệu đưa vào huấn luyện mô hình sau này"""
    split_and_prepare_datasets(limit)
    return GLOBAL_TRAIN, GLOBAL_VAL, GLOBAL_TEST


def clean_text(text):
    """Hàm tiền xử lý văn bản (Bám sát theo yêu cầu 5 bước)"""
    # 0. Chuyển thành chữ thường
    text = str(text).lower()

    # 1. Xoá URL, đường liên kết
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.IGNORECASE)

    # 2. Hoá hastag (#) và mention (@)
    text = re.sub(r"[@#]\w+", "", text)

    # 3. Xóa số
    text = re.sub(r"\d+", "", text)

    # 4. Xóa ký tự đặc biệt
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 5. Xóa stopwords và Lemmatization
    words = text.split()
    cleaned_words = []
    for word in words:
        # Sử dụng biến CUSTOM_STOP_WORDS đã import từ nlp_config.py
        if word not in CUSTOM_STOP_WORDS:
            # Sử dụng công cụ lemmatizer đã import
            lemma_word = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma_word)

    # Ghép lại thành câu hoàn chỉnh
    text = " ".join(cleaned_words)

    return text

def prepare_clean_datasets(limit=limit_max):
    """Hàm tạo 3 tập dữ liệu sạch và lưu vào biến toàn cục"""
    global GLOBAL_CLEAN_TRAIN, GLOBAL_CLEAN_VAL, GLOBAL_CLEAN_TEST

    # Nếu đã có dữ liệu sạch rồi thì không cần làm lại
    if GLOBAL_CLEAN_TRAIN is not None:
        return

    # 1. Chắc chắn rằng 3 tập dữ liệu gốc đã được cắt
    split_and_prepare_datasets(limit)

    # 2. Tạo bản sao (.copy()) để việc dọn dẹp không làm mất dữ liệu gốc
    df_clean_train = GLOBAL_TRAIN.copy()
    df_clean_val = GLOBAL_VAL.copy()
    df_clean_test = GLOBAL_TEST.copy()

    # 3. Áp dụng hàm clean_text cho cột 'text' của cả 3 tập
    df_clean_train['text'] = df_clean_train['text'].apply(clean_text)
    df_clean_val['text'] = df_clean_val['text'].apply(clean_text)
    df_clean_test['text'] = df_clean_test['text'].apply(clean_text)

    # 4. Cập nhật lại cờ 'needs_processing' thành False (vì dữ liệu đã sạch bong, không cần in màu đỏ ở frontend nữa)
    df_clean_train['needs_processing'] = False
    df_clean_val['needs_processing'] = False
    df_clean_test['needs_processing'] = False

    # 5. Lưu vào biến toàn cục mới
    GLOBAL_CLEAN_TRAIN = df_clean_train
    GLOBAL_CLEAN_VAL = df_clean_val
    GLOBAL_CLEAN_TEST = df_clean_test


def get_clean_dataset_by_name(set_name="train", limit=limit_max):
    """Hàm lấy tập dữ liệu SẠCH để trả về cho Frontend"""
    prepare_clean_datasets(limit)

    if set_name == "train":
        df_requested = GLOBAL_CLEAN_TRAIN
    elif set_name == "val":
        df_requested = GLOBAL_CLEAN_VAL
    elif set_name == "test":
        df_requested = GLOBAL_CLEAN_TEST
    else:
        df_requested = GLOBAL_CLEAN_TRAIN

    counts = {
        "train_count": len(GLOBAL_CLEAN_TRAIN),
        "val_count": len(GLOBAL_CLEAN_VAL),
        "test_count": len(GLOBAL_CLEAN_TEST)
    }

    return {
        "records": df_requested.to_dict(orient="records"),
        "counts": counts
    }


def get_clean_datasets_for_training(limit=limit_max):
    """
    Hàm này dùng để xuất 3 tập dữ liệu SẠCH ra ngoài cho các file huấn luyện mô hình.
    """
    # Đảm bảo dữ liệu đã được làm sạch trước khi trả về
    prepare_clean_datasets(limit)
    return GLOBAL_CLEAN_TRAIN, GLOBAL_CLEAN_VAL, GLOBAL_CLEAN_TEST
