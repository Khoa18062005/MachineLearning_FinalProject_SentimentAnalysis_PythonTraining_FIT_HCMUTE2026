import os
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    # Cấu hình Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # Cấu hình CORS (Cho phép Frontend truy cập)
    ALLOWED_ORIGINS: list = [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
    ]

    # Quản lý đường dẫn (Path Management)
    # Lấy thư mục gốc của dự án (FinalProject_ML_Backend_Python)
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Đường dẫn tới file dữ liệu 1.6 triệu dòng
    DATA_PATH: str = os.path.join(BASE_DIR, "data_training", "Data_Emotion.csv")

    # Nơi lưu trữ mô hình sau khi huấn luyện (Sẽ dùng ở các bước sau)
    MODEL_PATH: str = os.path.join(BASE_DIR, "app", "models", "sentiment_model.pkl")

    # 5. Cấu hình Dữ liệu
    CSV_COLUMNS: list = ["target", "id", "date", "flag", "user", "text"]
    CSV_ENCODING: str = "latin-1"

    class Config:
        case_sensitive = True
settings = Settings()