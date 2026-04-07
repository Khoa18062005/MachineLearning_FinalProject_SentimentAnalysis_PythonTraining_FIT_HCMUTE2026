import pandas as pd
from app.services.ml_service import get_clean_datasets_for_training
from app.core.pandas_helper import setup_pandas_display
setup_pandas_display()

# Hàm huấn luyện mô hình
def train_model():
    return f'dạy múa đi các bạn trẻ'

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

    print(df_train.head())

