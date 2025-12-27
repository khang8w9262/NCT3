import pandas as pd
import numpy as np
import sys
import os
import warnings
from collections import deque
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
import glob

# Tắt cảnh báo
warnings.filterwarnings('ignore')

# =============================================================================
# CẤU HÌNH HỆ THỐNG
# =============================================================================

SHORT_TERM_WINDOW = 3
MEDIUM_TERM_WINDOW = 14
VOLATILITY_WINDOW = 7
TREND_WINDOW = 30

ALPHA_RELEVANCE = 0.7
CONFIDENCE_THRESHOLD = 0.6
RELEVANCE_SCALE = 20.0

INTENSITY_MAP = {
    "đột biến": 1.0, "kỷ lục": 1.0, "sụp đổ": 1.0, "bùng nổ": 1.0, 
    "thảm hại": 1.0, "lao dốc": 1.0, "tăng trần": 1.0, "giảm sàn": 1.0,
    "khủng": 1.0, "phá sản": 1.0, "tăng vọt": 0.95, "giảm sâu": 0.9,
    "mạnh": 0.8, "lớn": 0.7, "cao": 0.7, 
    "khá": 0.6, "đáng kể": 0.6, "rõ rệt": 0.6, "vượt": 0.6,
    "nhẹ": 0.2, "ít": 0.2, "chậm": 0.2, "dần": 0.2, "đi ngang": 0.1, "hơi": 0.2
}

MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

# =============================================================================
# KHỞI TẠO MODEL
# =============================================================================
print(" Đang tải model PhoBERT...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f" Đã tải xong model trên thiết bị: {device}")
except Exception as e:
    print(f" Lỗi tải model: {e}")
    sys.exit(1)

# =============================================================================
# CÁC HÀM TÍNH TOÁN (CORE FUNCTIONS)
# =============================================================================

def get_probabilities(text):
    if not isinstance(text, str) or not text.strip():
        return {'P_pos': 0.0, 'P_neg': 0.0, 'P_neu': 1.0}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    id2label = getattr(model.config, "id2label", {})
    res = {'P_neg': 0.0, 'P_pos': 0.0, 'P_neu': 0.0}
    
    mapped = False
    if id2label:
        for i, p in enumerate(probs):
            label = str(id2label.get(i, "")).upper()
            if "NEG" in label: res['P_neg'] = float(p)
            elif "POS" in label: res['P_pos'] = float(p)
            elif "NEU" in label: res['P_neu'] = float(p)
        if res['P_pos'] + res['P_neg'] + res['P_neu'] > 0:
            mapped = True
            
    if not mapped:
        if len(probs) == 3:
            res['P_neg'], res['P_pos'], res['P_neu'] = float(probs[0]), float(probs[1]), float(probs[2])
        else:
            res['P_neg'], res['P_pos'] = float(probs[0]), float(probs[-1])

    return res

def calculate_intensity(text):
    text_lower = str(text).lower()
    max_intensity = 0.5
    matched_word = "None"
    
    for phrase, score in INTENSITY_MAP.items():
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text_lower):
            if score > max_intensity:
                max_intensity = score
                matched_word = phrase
    return max_intensity, matched_word

def calculate_relevance(text, title, target_code, scale=RELEVANCE_SCALE):
    target = str(target_code).strip()
    text_lower = str(text).lower()
    title_lower = str(title).lower()
    
    if not target: return 0.0

    pattern = r"\b" + re.escape(target.lower()) + r"\b"
    count_entity = len(re.findall(pattern, text_lower))
    
    words = text_lower.split()
    total_words = len(words) if words else 1
    
    freq_ratio = count_entity / total_words
    freq_component = min(freq_ratio * scale, 1.0)
    
    title_match = 1.0 if re.search(pattern, title_lower) else 0.0
    
    relevance = ALPHA_RELEVANCE * title_match + (1 - ALPHA_RELEVANCE) * freq_component
    return round(float(relevance), 6)

def calculate_trend_slope(history):
    if len(history) < 2: return 0.0
    y = np.array(history, dtype=float)
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# =============================================================================
# TEXT CLEANING: remove emojis / icons / control chars
# =============================================================================
def clean_text_remove_icons(text):
    """Remove emojis, pictographs, variation selectors and control characters."""
    if not isinstance(text, str):
        return text

    # remove variation selectors
    text = re.sub(r'[\uFE00-\uFE0F]', '', text)

    # common emoji / pictograph ranges
    emoji_pattern = re.compile(
        '['
        '\U0001F600-\U0001F64F'  # emoticons
        '\U0001F300-\U0001F5FF'  # symbols & pictographs
        '\U0001F680-\U0001F6FF'  # transport & map
        '\U0001F1E0-\U0001F1FF'  # flags
        '\U00002700-\U000027BF'  # dingbats
        '\U000024C2-\U0001F251'
        ']', flags=re.UNICODE)

    text = emoji_pattern.sub('', text)

    # miscellaneous symbols
    text = re.sub(r'[\u2600-\u26FF\u2B00-\u2BFF]', '', text)

    # control chars
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================================
# HÀM XỬ LÝ 1 FILE (WORKER)
# =============================================================================

def process_sentiment_analysis(input_file, output_file, target_code):
    print(f"\n Đang xử lý: {target_code}")
    print(f"   Input: {os.path.basename(input_file)}")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f" Lỗi đọc file: {e}")
        return

    if 'Date' in df.columns:
        df['DateObj'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('DateObj').reset_index(drop=True)
    else:
        print(" Cảnh báo: Không có cột 'Date'.")

    history_sentiment = deque(maxlen=TREND_WINDOW)
    history_short_term = deque(maxlen=2)

    results = []
    total = len(df)
    
    for idx, row in df.iterrows():
        content = str(row.get('Content', '') or '')
        header = str(row.get('Header', row.get('Title', '')) or '')

        # Remove icons/emojis from text fields before inference
        header = clean_text_remove_icons(header)
        content = clean_text_remove_icons(content)

        # Core Logic
        probs = get_probabilities(content)
        raw_sentiment = probs['P_pos'] - probs['P_neg']
        confidence = max(probs['P_pos'], probs['P_neu'], probs['P_neg'])
        final_sentiment = raw_sentiment if confidence > CONFIDENCE_THRESHOLD else 0.0
        
        intensity, match_word = calculate_intensity(content)
        impact_score = final_sentiment * intensity
        relevance_score = calculate_relevance(content, header, target_code)
        
        # Temporal
        history_sentiment.append(final_sentiment)
        hist_list = list(history_sentiment)
        
        short_slice = hist_list[-min(len(hist_list), SHORT_TERM_WINDOW):]
        short_term_score = np.mean(short_slice) if short_slice else 0.0
        
        med_slice = hist_list[-min(len(hist_list), MEDIUM_TERM_WINDOW):]
        medium_term_score = np.mean(med_slice) if med_slice else 0.0
        
        momentum = 0.0
        if len(history_short_term) > 0:
            momentum = short_term_score - history_short_term[-1]
        history_short_term.append(short_term_score)
        
        vol_slice = hist_list[-min(len(hist_list), VOLATILITY_WINDOW):]
        volatility = np.std(vol_slice) if len(vol_slice) > 1 else 0.0
        
        trend = calculate_trend_slope(hist_list)
        
        results.append({
            'Date': row.get('Date', ''),
            'Header': header,
            'Content': content,
            'sentiment_score': round(final_sentiment, 6),
            'impact_score': round(impact_score, 6),
            'relevance_score': round(relevance_score, 6),
            'confidence': round(confidence, 6),
            'short_term_score': round(short_term_score, 6),
            'medium_term_score': round(medium_term_score, 6),
            'sentiment_momentum': round(momentum, 6),
            'sentiment_volatility': round(volatility, 6),
            'sentiment_trend': round(trend, 8),
            'prob_pos': round(probs['P_pos'], 4),
            'prob_neg': round(probs['P_neg'], 4),
            'intensity_match': match_word
        })
        
        if idx % 50 == 0: # Giảm log in ra để đỡ rối khi chạy nhiều file
            sys.stdout.write(f"\r   ⏳ Progress: {idx+1}/{total}")
            sys.stdout.flush()

    df_out = pd.DataFrame(results)
    # Bỏ 3 cột không cần thiết nếu có
    for col in ['prob_pos', 'prob_neg', 'intensity_match']:
        if col in df_out.columns:
            df_out = df_out.drop(columns=[col])
    # Lọc icon/emojis khỏi toàn bộ các cột dạng chuỗi (làm sạch lần cuối trước khi lưu)
    for col in df_out.select_dtypes(include='object').columns:
        df_out[col] = df_out[col].apply(clean_text_remove_icons)
    # Sắp xếp lại theo ngày tăng dần nếu có cột Date
    if 'Date' in df_out.columns:
        # Chuyển Date về datetime để sort
        df_out['DateObj'] = pd.to_datetime(df_out['Date'], dayfirst=True, errors='coerce')
        df_out = df_out.sort_values('DateObj').drop(columns=['DateObj']).reset_index(drop=True)
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n    Xong! Output: {os.path.basename(output_file)}")

# =============================================================================
# TỰ ĐỘNG QUÉT VÀ CHẠY (BATCH PROCESSING)
# =============================================================================

def scan_and_process_batch():
    # 1. Xác định đường dẫn tương đối từ vị trí script
    # Script đang ở: NCT3/SRC/CRAWLING/calcu_sentiment.py
    # Data đang ở:   NCT3/DATASET/SENTIMENT/
    
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    # Lùi lại 2 cấp: CRAWLING -> SRC -> NCT3 (Root)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    
    # Tạo đường dẫn tới thư mục SENTIMENT
    sentiment_dir = os.path.join(project_root, 'DATASET', 'SENTIMENT')
    
    print(f" Đang quét thư mục: {sentiment_dir}")
    
    if not os.path.exists(sentiment_dir):
        print(f" Không tìm thấy thư mục DATASET/SENTIMENT tại: {sentiment_dir}")
        print("Vui lòng kiểm tra lại cấu trúc thư mục.")
        return

    # 2. Lấy danh sách tất cả file CSV
    all_files = glob.glob(os.path.join(sentiment_dir, "*.csv"))
    
    if not all_files:
        print(" Không tìm thấy file .csv nào trong thư mục.")
        return

    count = 0
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        # 3. Logic lọc: Bỏ qua file đã là kết quả (chứa '_sentiment')
        if "_sentiment.csv" in filename:
            continue
            
        # 4. Xác định Output và Target
        # Input: VNM.csv -> Target: VNM -> Output: VNM_sentiment.csv
        target_name = os.path.splitext(filename)[0] # Lấy tên file bỏ đuôi .csv
        output_filename = f"{target_name}_sentiment.csv"
        output_path = os.path.join(sentiment_dir, output_filename)
        
        # Kiểm tra nếu file output đã tồn tại thì có thể bỏ qua (tùy chọn)
        # Ở đây mình cứ chạy đè lên (overwrite) để cập nhật mới nhất
        
        # 5. Gọi hàm xử lý
        process_sentiment_analysis(file_path, output_path, target_name)
        count += 1
        
    print(f"\n{'='*40}")
    print(f"🎉 HOÀN TẤT BATCH PROCESSING!")
    print(f"Đã xử lý thành công: {count} file.")
    print(f"{'='*40}")

if __name__ == "__main__":
    # Nếu người dùng truyền tham số cụ thể, chạy chế độ cũ (Single File)
    if len(sys.argv) >= 4:
        process_sentiment_analysis(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Nếu không truyền tham số, chạy chế độ tự động quét (Batch Mode)
        scan_and_process_batch()