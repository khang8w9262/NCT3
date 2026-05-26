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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Đảm bảo in tiếng Việt ổn định trên terminal không phải UTF-8.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

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
    "thảm hại": 1.0, "lao dốc": 1.0, "tăng trần": 1.0, "giảm sàn": 1.0, "nhất trong lịch_sử": 1.0,
    "khủng": 1.0, "phá sản": 1.0, "tăng vọt": 0.95, "giảm sâu": 0.9,
    "mạnh": 0.8, "lớn": 0.7, "cao": 0.7, 
    "khá": 0.6, "đáng kể": 0.6, "rõ rệt": 0.6, "vượt": 0.6,
    "nhẹ": 0.2, "ít": 0.2, "chậm": 0.2, "dần": 0.2, "đi ngang": 0.1, "hơi": 0.2
}

MODEL_NAME = "wonrax/phobert-base-vietnamese-sentiment"

SENTIMENT_9_COLS = [
    'sentiment_score',
    'impact_score',
    'relevance_score',
    'confidence',
    'short_term_score',
    'medium_term_score',
    'sentiment_momentum',
    'sentiment_volatility',
    'sentiment_trend'
]

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


def get_sentiment_chart_dir():
    """Trả về thư mục lưu biểu đồ sentiment: NCT3/CHART/SENTIMENT."""
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    chart_dir = os.path.join(project_root, 'CHART', 'SENTIMENT')
    os.makedirs(chart_dir, exist_ok=True)
    return chart_dir

def save_post_analysis_plots(df_out, output_file, target_code):
    """Lưu 4 loại biểu đồ phân tích:
    1) Ma trận tương quan (Seaborn Heatmap PNG & CSV).
    2) Biểu đồ chỉ số theo thời gian (Plotly Time Series HTML).
    3) Heatmap thời gian x chỉ số (Plotly Heatmap HTML).
    4) Biểu đồ Radar động có thanh trượt thời gian (Plotly Animated Radar HTML).
    """
    available_cols = [c for c in SENTIMENT_9_COLS if c in df_out.columns]
    if len(available_cols) != len(SENTIMENT_9_COLS):
        missing = [c for c in SENTIMENT_9_COLS if c not in df_out.columns]
        print(f"   [CẢNH BÁO] Thiếu cột cho biểu đồ: {missing}. Bỏ qua vẽ biểu đồ.")
        return

    # Chuẩn bị dữ liệu số
    metric_df = df_out[available_cols].apply(pd.to_numeric, errors='coerce')
    metric_df = metric_df.dropna(how='all')
    
    if metric_df.empty:
        print("   [CẢNH BÁO] Không có dữ liệu số hợp lệ để vẽ biểu đồ.")
        return

    # Chuẩn bị dữ liệu thời gian cho Plotly
    # Cần tạo bản sao có cột Date chuẩn để làm trục thời gian
    # Nếu có nhiều dòng cùng 1 ngày, ta sẽ GROUP BY DATE (lấy trung bình) để vẽ biểu đồ theo ngày cho gọn.
    plot_df = df_out.copy()
    if 'DateObj' not in plot_df.columns and 'Date' in plot_df.columns:
         plot_df['DateObj'] = pd.to_datetime(plot_df['Date'], dayfirst=True, errors='coerce')
    
    # Chỉ giữ lại các dòng có Date hợp lệ
    plot_df = plot_df.dropna(subset=['DateObj']).sort_values('DateObj')
    
    if plot_df.empty:
        print("   [CẢNH BÁO] Không có cột ngày tháng hợp lệ (Date) để vẽ biểu đồ theo thời gian.")
        # Vẫn vẽ Correlation Matrix được vì không cần thread thời gian
    else:
        # Group by Date: Lấy trung bình các chỉ số trong cùng 1 ngày
        # Chuyển đổi DateObj về dạng chuỗi ngày trước khi Groupby
        plot_df['DateStr'] = plot_df['DateObj'].dt.strftime('%Y-%m-%d')
        
        # Group by DateStr: Đảm bảo mỗi ngày chỉ có 1 dòng duy nhất (tránh DuplicateError)
        daily_df = plot_df.groupby('DateStr')[available_cols].mean().reset_index()
        
        # Sắp xếp lại theo thời gian (vì groupby trên chuỗi có thể làm sai thứ tự)
        daily_df['DateObj_tmp'] = pd.to_datetime(daily_df['DateStr'])
        daily_df = daily_df.sort_values('DateObj_tmp').drop(columns=['DateObj_tmp'])
    
    out_dir = get_sentiment_chart_dir()
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    
    print(f"   Đang tạo biểu đồ cho {target_code}...")

    # ==================================================================
    # 1. CORRELATION MATRIX (MA TRẬN TƯƠNG QUAN) - Giữ nguyên logic cũ
    # ==================================================================
    corr_df = metric_df.corr()
    corr_csv_path = os.path.join(out_dir, f"{base_name}_correlation.csv")
    corr_df.to_csv(corr_csv_path, encoding='utf-8-sig')

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
        square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'}, ax=ax_corr
    )
    ax_corr.set_title(f"Correlation Matrix - {target_code}")
    fig_corr.tight_layout()
    corr_plot_path = os.path.join(out_dir, f"{base_name}_correlation.png")
    fig_corr.savefig(corr_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_corr)

    # Nếu không có dữ liệu daily (time-series), dừng ở đây
    if plot_df.empty or daily_df.empty:
        print(f"    Correlation Matrix saved: {os.path.basename(corr_plot_path)}")
        return

    # ==================================================================
    # 2. TIME SERIES CHART (BIỂU ĐỒ 9 CHỈ SỐ THEO THỜI GIAN)
    # ==================================================================
    # Vẽ line chart cho 9 chỉ số
    fig_ts = px.line(
        daily_df, x='DateStr', y=available_cols,
        title=f"Sentiment Indicators Over Time - {target_code}",
        markers=True
    )
    fig_ts.update_layout(xaxis_title="Date", yaxis_title="Score", hovermode="x unified")
    
    ts_plot_path = os.path.join(out_dir, f"{base_name}_timeseries.html")
    fig_ts.write_html(ts_plot_path)

    # ==================================================================
    # 3. HEATMAP (TIME vs METRICS)
    # ==================================================================
    # Trục X: Date, Trục Y: Metrics, Màu: Giá trị
    # Cần transpose hoặc melt? Dễ nhất là dùng px.imshow với dữ liệu Matrix
    # Matrix: Hàng = Ngày, Cột = Chỉ số
    heatmap_data = daily_df.set_index('DateStr')[available_cols].transpose()
    
    fig_hm = px.imshow(
        heatmap_data,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        aspect="auto",
        title=f"Sentiment Heatmap Over Time - {target_code}",
        color_continuous_scale="RdBu_r", # Red-Blue reverse (Blue=Pos, Red=Neg)
        origin='lower'
    )
    hm_plot_path = os.path.join(out_dir, f"{base_name}_heatmap.html")
    fig_hm.write_html(hm_plot_path)

    # ==================================================================
    # 4. ANIMATED RADAR CHART (RADAR CÓ THANH TRƯỢT THỜI GIAN)
    # ==================================================================
    # Cần melt dữ liệu về dạng Long Format: Date | Variable | Value
    radar_df = daily_df.melt(id_vars=['DateStr'], value_vars=available_cols, var_name='Metric', value_name='Value')
    
    # Chuẩn hóa dữ liệu về [0, 1] cho Radar đẹp hơn (nếu cần). 
    # Tuy nhiên, nếu muốn giữ giá trị thực (-1 đến 1) thì range_r=[-1, 1].
    # Model sentiment của ta từ -1 đến 1 (hoặc 0 đến 1 cho confidence).
    # Để an toàn và dễ nhìn, ta set range cố định [-1, 1] hoặc [min, max] của toàn bộ data.
    r_min = -1.0
    r_max = 1.0 # Sentiment score max là 1, Impact có thể > 1 nếu intensity cao? Intensity max 1.0 => Impact max 1.0. Trend, Volatility có thể khác.
    
    # Kiểm tra max thực tế để set range
    real_max = radar_df['Value'].max()
    real_min = radar_df['Value'].min()
    if real_max > 1.0: r_max = real_max
    if real_min < -1.0: r_min = real_min
    
    # Tạo Radar
    fig_radar = px.line_polar(
        radar_df, r='Value', theta='Metric',
        animation_frame='DateStr', # Thanh trượt thời gian
        line_close=True,
        range_r=[r_min, r_max],
        title=f"Animated Sentiment Radar - {target_code}",
        template="plotly_dark",
    )
    # Làm chậm tốc độ animation một chút
    fig_radar.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000 
    
    radar_plot_path = os.path.join(out_dir, f"{base_name}_radar_animated.html")
    fig_radar.write_html(radar_plot_path)

    print(f"    1. Correlation Matrix: {os.path.basename(corr_plot_path)}")
    print(f"    2. Time Series Chart:  {os.path.basename(ts_plot_path)}")
    print(f"    3. Heatmap Chart:      {os.path.basename(hm_plot_path)}")
    print(f"    4. Animated Radar:     {os.path.basename(radar_plot_path)}")
    print(f"    Folder: {out_dir}")

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
    results = []
    total = len(df)
    
    for pos, (_, row) in enumerate(df.iterrows()):
        content = str(row.get('Content', '') or '')
        header = str(row.get('Header', row.get('Title', '')) or '')

        # Remove icons/emojis from text fields before inference
        header = clean_text_remove_icons(header)
        content = clean_text_remove_icons(content)

        # ---------------------------------------------------------------------
        # 1. SENTIMENT SCORE (Điểm chỉ số cảm xúc)
        # ---------------------------------------------------------------------
        # Sử dụng model PhoBERT để dự đoán xác suất: Pos (Tích cực), Neg (Tiêu cực), Neu (Trung lập).
        # Công thức: raw = P_pos - P_neg (Range: -1 đến 1)
        # Lọc nhiễu: Nếu độ tự tin (Confidence) thấp hơn ngưỡng (0.6) -> Về 0.
        probs = get_probabilities(content)
        raw_sentiment = probs['P_pos'] - probs['P_neg']
        confidence = max(probs['P_pos'], probs['P_neu'], probs['P_neg'])
        final_sentiment = raw_sentiment if confidence > CONFIDENCE_THRESHOLD else 0.0
        
        # ---------------------------------------------------------------------
        # 2. INTENSITY & IMPACT (Cường độ & Tác động)
        # ---------------------------------------------------------------------
        # Intensity: Dựa trên từ khóa mạnh (ví dụ: "đột biến" -> 1.0, "nhẹ" -> 0.2).
        # Impact Score: Kết hợp Sentiment * Intensity. Ví dụ tin tốt mà từ ngữ mạnh -> Impact cao.
        intensity, match_word = calculate_intensity(content)
        impact_score = final_sentiment * intensity

        # ---------------------------------------------------------------------
        # 3. RELEVANCE SCORE (Mức độ liên quan)
        # ---------------------------------------------------------------------
        # Đánh giá bài viết có nói nhiều về mã cổ phiếu (Target) hay không.
        # Kết hợp: Xuất hiện ở tiêu đề (Title Match) + Tần suất xuất hiện trong nội dung (Frequency).
        relevance_score = calculate_relevance(content, header, target_code)
        
        # ---------------------------------------------------------------------
        # 4. TEMPORAL METRICS (Chỉ số theo thời gian/chuỗi)
        # ---------------------------------------------------------------------
        # Lưu lịch sử sentiment để tính Moving Average và Trend.
        history_sentiment.append(final_sentiment)
        hist_list = list(history_sentiment)
        
        # a. Short Term Score: Trung bình trượt ngắn hạn (3 ngày/bài gần nhất)
        short_slice = hist_list[-min(len(hist_list), SHORT_TERM_WINDOW):]
        short_term_score = np.mean(short_slice) if short_slice else 0.0
        
        # b. Medium Term Score: Trung bình trượt trung hạn (14 ngày/bài gần nhất)
        med_slice = hist_list[-min(len(hist_list), MEDIUM_TERM_WINDOW):]
        medium_term_score = np.mean(med_slice) if med_slice else 0.0
        
        # c. Volatility: Độ biến động (Độ lệch chuẩn - Std Dev) trong 7 ngày gần nhất.
        # Volatility cao -> Tâm lý thị trường đang dao động mạnh, bất ổn.
        vol_slice = hist_list[-min(len(hist_list), VOLATILITY_WINDOW):]
        volatility = np.std(vol_slice) if len(vol_slice) > 1 else 0.0
        
        # d. Trend Slope: Hệ số góc (Slope) của đường xu hướng tuyến tính trong 30 ngày.
        # Slope > 0: Trend tăng. Slope < 0: Trend giảm.
        trend = calculate_trend_slope(hist_list)
        
        results.append({
            'Date': row.get('Date', ''),
            'Header': header,
            'Content': content,
            # Core Sentiment
            'sentiment_score': round(final_sentiment, 6),    # Đã lọc qua threshold
            'impact_score': round(impact_score, 6),          # Sentiment * Intensity
            'relevance_score': round(relevance_score, 6),    # Độ liên quan tới mã
            'confidence': round(confidence, 6),              # Độ tin cậy của model
            
            # Moving Averages
            'short_term_score': round(short_term_score, 6),  # MA(3)
            'medium_term_score': round(medium_term_score, 6),# MA(14)
            
            # Advanced Indicators (Momentum sẽ được tính sau khi sort)
            'sentiment_momentum': 0.0,                       # Placeholder
            'sentiment_volatility': round(volatility, 6),    # Độ bất ổn tâm lý
            'sentiment_trend': round(trend, 8),              # Xu hướng dài hạn (Slope)
            
            # Debug/Extra info (Will be dropped later)
            'prob_pos': round(probs['P_pos'], 4),
            'prob_neg': round(probs['P_neg'], 4),
            'intensity_match': match_word
        })
        
        if pos % 50 == 0: # Giảm log in ra để đỡ rối khi chạy nhiều file
            sys.stdout.write(f"\r   Progress: {pos+1}/{total}")
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
    
    # =========================================================================
    # TÍNH MOMENTUM SAU KHI ĐÃ SORT XONG
    # =========================================================================
    # Công thức: momentum[t] = short_term_score[t] - short_term_score[t-1]
    # Tính SAU khi sort để đảm bảo verify được trực tiếp trong file output:
    # momentum của dòng N = short_term của dòng N - short_term của dòng N-1
    df_out['sentiment_momentum'] = df_out['short_term_score'].diff().fillna(0).round(6)
    
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    save_post_analysis_plots(df_out, output_file, target_code)
    print(f"\n    Xong! Output: {os.path.basename(output_file)}")

# =============================================================================
# TỰ ĐỘNG QUÉT VÀ CHẠY (BATCH PROCESSING)
# =============================================================================

def scan_and_process_batch(target_filter=None):
    # 1. Xác định đường dẫn tương đối từ vị trí script
    # Script đang ở: NCT3/SRC/CRAWLING/calcu_sentiment.py
    # Data đang ở:   NCT3/DATASET/TEST/SENTIMENT/
    
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    # Lùi lại 2 cấp: CRAWLING -> SRC -> NCT3 (Root)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    
    # Tạo đường dẫn tới thư mục SENTIMENT
    sentiment_dir = os.path.join(project_root, 'DATASET', 'TRAIN', 'SENTIMENT')
    
    print(f" Đang quét thư mục: {sentiment_dir}")
    
    if not os.path.exists(sentiment_dir):
        print(f" Không tìm thấy thư mục DATASET/TRAIN/SENTIMENT tại: {sentiment_dir}")
        print("Vui lòng kiểm tra lại cấu trúc thư mục.")
        return

    # 2. Lấy danh sách tất cả file CSV
    all_files = glob.glob(os.path.join(sentiment_dir, "*.csv"))
    
    if not all_files:
        print(" Không tìm thấy file .csv nào trong thư mục.")
        return

    # Chỉ xử lý file nguồn (VD: ALIBABA.csv), bỏ qua file kết quả/artifact đã sinh ra
    source_files = []
    skipped_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        upper_name = base_name.upper()

        # Bỏ qua các file output cũ nếu có
        if upper_name.endswith("_SENTIMENT"):
            skipped_files.append((filename, "already sentiment output"))
            continue

        if "_CORRELATION" in upper_name or "_HEATMAP" in upper_name or "_RADAR" in upper_name:
            skipped_files.append((filename, "chart artifact"))
            continue

        # Lọc theo target user nhập (nếu có)
        if target_filter:
            # Kiểm tra tên file có chứa mã target không (VD: ALIBABA chứa BABA? Ở đây giả sử nhập ALIBABA)
            # Hoặc so sánh exact substring
            if target_filter.upper() not in upper_name:
                continue

        source_files.append(file_path)

    print(f" Tìm thấy {len(source_files)} file nguồn cần xử lý.")
    if target_filter:
        print(f" (Đã lọc theo mã: {target_filter})")
        
    if skipped_files:
        print(f" Bỏ qua {len(skipped_files)} file kết quả/artifact.")

    if not source_files:
        print(" Không có file nguồn hợp lệ để xử lý.")
        return

    count = 0
    for file_path in source_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        target_code = base_name
        output_filename = f"{base_name}_sentiment.csv"
        output_path = os.path.join(sentiment_dir, output_filename)

        # 5. Gọi hàm xử lý
        process_sentiment_analysis(file_path, output_path, target_code)
        count += 1
        
    print(f"\n{'='*40}")
    print(f" HOÀN TẤT BATCH PROCESSING!")
    print(f"Đã xử lý thành công: {count} file.")    
    print(f"{'='*40}")

if __name__ == "__main__":
    try:
        # Nếu người dùng truyền tham số cụ thể, chạy chế độ cũ (Single File)
        if len(sys.argv) >= 4:
            process_sentiment_analysis(sys.argv[1], sys.argv[2], sys.argv[3])
        elif len(sys.argv) == 2:
             # Batch mode với tham số lọc (chạy từ dòng lệnh)
             scan_and_process_batch(sys.argv[1])
        else:
            # Chức năng chọn tên chứng khoán (nhập tay)
            target_filter = input("Nhập mã cổ phiếu muốn phân tích (Để trống để chạy toàn bộ): ").strip()
            # Nếu không truyền tham số, chạy chế độ tự động quét (Batch Mode)
            scan_and_process_batch(target_filter)
    except KeyboardInterrupt:
        print("\n Đã dừng chương trình theo yêu cầu người dùng (Ctrl+C).")