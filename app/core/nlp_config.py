import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def setup_nltk():
    """Hàm tải các gói dữ liệu ngôn ngữ (Có xử lý lỗi SSL trên macOS)"""
    # Xử lý lỗi chứng chỉ SSL (Rất hay gặp trên Mac khi dùng thư viện tải file)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Tải trực tiếp (Nếu máy bạn có rồi, nó sẽ tự động bỏ qua cực nhanh)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# 1. Gọi hàm tải dữ liệu ngay khi file được đọc
setup_nltk()

# 2. Khởi tạo công cụ Lemmatization (Đưa từ về nguyên thể)
lemmatizer = WordNetLemmatizer()

# 3. Lấy danh sách Stopwords tiếng Anh mặc định
_default_stop_words = set(stopwords.words('english'))

# 4. Trừ đi các từ phủ định để giữ lại chúng
_words_to_keep = {'not', 'never', 'no'}
CUSTOM_STOP_WORDS = _default_stop_words - _words_to_keep

print("✅ Đã khởi tạo cấu hình NLTK và Stopwords thành công.")