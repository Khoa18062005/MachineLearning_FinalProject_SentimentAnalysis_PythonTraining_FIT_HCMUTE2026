import pandas as pd
def setup_pandas_display():
    """Cấu hình cách hiển thị DataFrame"""
    # 1. Hiển thị tất cả các cột (không dùng dấu ...)
    pd.set_option('display.max_columns', None)
    # 2. Hiển thị đầy đủ nội dung trong mỗi cột (không bị cắt bớt chữ trong text)
    pd.set_option('display.max_colwidth', None)
    # 3. Hiển thị số lượng dòng mong muốn (ví dụ 100 dòng)
    pd.set_option('display.max_rows', 100)
    # 4. Đảm bảo toàn bộ bảng được in trên cùng một khối
    pd.set_option('display.expand_frame_repr', False)