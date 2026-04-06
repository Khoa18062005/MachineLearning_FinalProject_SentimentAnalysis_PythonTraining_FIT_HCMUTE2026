import pandas as pd
import os

COLUMNS = ["target", "id", "date", "flag", "user", "text"]
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data_training/Data_Emotion.csv")

def _load_raw_df(limit=200):
    return pd.read_csv(DATA_PATH, names=COLUMNS, nrows=limit, encoding='latin-1')

def get_data_preview(limit=200):
    df = _load_raw_df(limit)
    return df.to_dict(orient="records")

def get_data_features(limit=200):
    df = _load_raw_df(limit)
    # Tái sử dụng df và chỉ lọc cột cần thiết
    features = ['target', 'text']
    return df[features].to_dict(orient="records")