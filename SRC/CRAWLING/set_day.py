import os
import pandas as pd
import glob

def standardize_date_file(file_path):
    """
    Đọc file CSV, chuẩn hóa cột Date về định dạng YYYY-MM-DD.
    Hỗ trợ đọc cả định dạng VN (dd/mm/yyyy) và US (mm/dd/yyyy).
    """
    try:
        # Đọc file CSV
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Kiểm tra xem có cột Date không
        if 'Date' not in df.columns:
            print(f"  Bỏ qua: {os.path.basename(file_path)} (Không có cột Date)")
            return

        # CHUYỂN ĐỔI NGÀY THÁNG
        # dayfirst=True: Ưu tiên hiểu ngày đứng trước (VD: 23/10 -> ngày 23 tháng 10)
        # errors='coerce': Nếu dòng nào lỗi data (như 'unknown') thì biến thành NaT để không crash
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

        # Loại bỏ các dòng mà ngày tháng bị lỗi (NaT)
        df = df.dropna(subset=['Date'])

        # Format lại thành chuỗi chuẩn YYYY-MM-DD (Ví dụ: 2024-10-30)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # Lưu đè lại file cũ
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f" Đã chuẩn hóa (YYYY-MM-DD): {os.path.basename(file_path)}")

    except Exception as e:
        print(f" Lỗi khi xử lý file {os.path.basename(file_path)}: {str(e)}")

def process_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f" Không tìm thấy thư mục: {folder_path}")
        return
    
    # Lấy tất cả file .csv
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"\n📂 Đang xử lý thư mục: {folder_path}")
    print(f"   Tìm thấy {len(files)} files.")
    
    for f in files:
        standardize_date_file(f)

if __name__ == '__main__':
    # Tự động tìm đường dẫn dựa trên vị trí file set_day.py
    # Giả sử cấu trúc: D:\NghienCuu\NCT3\MODEL\DLINEAR+NODE\set_day.py
    # Data nằm ở:      D:\NghienCuu\NCT3\DATASET\
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Lùi lại 3 cấp thư mục để về thư mục gốc NCT3 (tùy cấu trúc máy bạn)
    # Nếu file này nằm trong MODEL/DLINEAR+NODE thì phải lùi ra ../.. mới thấy DATASET
    
    # Cách an toàn: Sử dụng đường dẫn tuyệt đối mà bạn đã cung cấp trong log trước
    base_price = r"D:\NghienCuu\NCT3\DATASET\PRICE"
    base_sentiment = r"D:\NghienCuu\NCT3\DATASET\SENTIMENT"

    # Nếu không tìm thấy đường dẫn tuyệt đối (trường hợp chạy máy khác), thử dùng đường dẫn tương đối
    if not os.path.exists(base_price):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        base_price = os.path.join(base_dir, 'DATASET', 'PRICE')
        base_sentiment = os.path.join(base_dir, 'DATASET', 'SENTIMENT')

    print("========================================================")
    print("BẮT ĐẦU CHUẨN HÓA DỮ LIỆU NGÀY THÁNG")
    print("Mục tiêu: Chuyển tất cả về YYYY-MM-DD để khớp lệnh Merge")
    print("========================================================")

    process_folder(base_price)
    process_folder(base_sentiment)
    
    print("\n HOÀN TẤT! Bây giờ bạn có thể chạy lại train_multi_model_system.py")