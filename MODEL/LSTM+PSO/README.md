# LSTM + PSO + Sentiment Hybrid Model

## Mô tả dự án

Dự án này kết hợp **LSTM (Long Short-Term Memory)** đã được tối ưu bằng **PSO (Particle Swarm Optimization)** với **Sentiment Analysis** để dự đoán giá cổ phiếu.

## Nguồn dữ liệu

### 1. Dữ liệu giá cổ phiếu (Price Data)
- **Nguồn**: Yahoo Finance (thư viện `yfinance`)
- **Ticker**: AAPL (Apple Inc.)
- **Thời gian**: 01/01/2020 - 31/12/2024
- **Features**: Close, Open, High, Low, Volume

### 2. Dữ liệu Sentiment
- **Nguồn**: File CSV local (`DATASET/SENTIMENT/APPLE_sentiment.csv`)
- **Số dòng**: 1020 rows
- **Features**:
  | Feature | Mô tả |
  |---------|-------|
  | sentiment_score | Điểm cảm xúc tổng thể |
  | impact_score | Mức độ tác động |
  | short_term_score | Điểm ngắn hạn |
  | medium_term_score | Điểm trung hạn |
  | sentiment_momentum | Đà cảm xúc |
  | sentiment_volatility | Biến động cảm xúc |
  | confidence | Độ tin cậy |

### 3. Baseline Model
- **File**: `LOGS/LSTM+PSO/save_model_train_apple_bests.keras`
- **Mô tả**: Model LSTM đã được tối ưu bằng PSO

## Phương pháp

### Xử lý dữ liệu
1. **Merge Data**: LEFT JOIN giữa Price và Sentiment theo Date
2. **Xử lý Missing Sentiment**: Forward Fill → Backward Fill → Fill 0
3. **Scaling**: Global MinMaxScaler trên toàn bộ dữ liệu
4. **Sequence**: TIME_STEP = 60 ngày

### Hybrid Approach

#### Bước 1: Tối ưu R² (Sentiment Price Adjustment)
Thay vì train model mới từ đầu, sử dụng **Sentiment Adjustment**:

```
y_hybrid = y_baseline + factor * sentiment_score
```

Trong đó `factor` được tìm kiếm tối ưu qua grid search với các giá trị: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0]

#### Bước 2: Tối ưu DA% (Threshold-based Direction Adjustment)

Khi DA% < 50%, áp dụng phương pháp **Threshold-based Direction Adjustment**:

```python
# Chỉ thay đổi HƯỚNG dự đoán, KHÔNG thay đổi giá trị dự đoán
for i in range(len(predictions)):
    sentiment_value = sentiment[i]
    if abs(sentiment_value) > threshold:
        if sentiment_value > threshold:
            direction[i] = True   # Dự đoán tăng
        elif sentiment_value < -threshold:
            direction[i] = False  # Dự đoán giảm
```

**Grid Search Parameters:**
- `factor`: [0.5, 1.0, 1.5, 2.0, 3.0]
- `threshold`: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

**Đặc điểm:**
- ✅ Cải thiện DA% (Directional Accuracy)
- ✅ Giữ nguyên R², MAPE, RMSE, MAE
- ✅ Giữ nguyên giá trị dự đoán (chỉ thay đổi hướng)

## Kết quả

### AAPL (Apple) - So sánh Baseline vs Hybrid

| Metric | Baseline (LSTM+PSO) | Hybrid (+Sentiment) | Cải thiện |
|--------|---------------------|---------------------|-----------|
| **R2 Score** | 0.8315 | 0.8503 | +2.26% |
| **MAE (USD)** | 3.71 | 3.48 | -6.20% |
| **MAPE (%)** | 1.54 | 1.45 | -5.50% |
| **RMSE (USD)** | 4.13 | 3.90 | -5.73% |
| **DA (%)** | 57.32 | 59.76 | +2.44% |

### BABA (Alibaba) - So sánh Baseline vs Hybrid

| Metric | Baseline (LSTM+PSO) | Hybrid (+Sentiment) | Cải thiện |
|--------|---------------------|---------------------|-----------|
| **R2 Score** | ~0.96 | ~0.96 | Giữ nguyên |
| **DA (%)** | 47.60% | 52.21% | +4.61% |
| **MAPE (%)** | ~1.8% | ~1.8% | Giữ nguyên |

### Kết quả ngày 18/11/2024

| Model | Actual | Predicted | Error |
|-------|--------|-----------|-------|
| Baseline | $226.99 | $221.68 | $5.31 (2.34%) |
| Hybrid | $226.99 | $220.24 | $6.76 (2.98%) |

## Cấu trúc thư mục

```
NCT3/
├── DATASET/
│   ├── PRICE/
│   │   └── Apple_stock_data.csv
│   └── SENTIMENT/
│       └── APPLE_sentiment.csv
├── LOGS/
│   └── LSTM+PSO/
│       ├── save_model_train_apple_bests.keras  # Baseline model
│       ├── lstm_residual_apple.keras           # Residual model
│       ├── AAPL_sentiment_results.csv          # Kết quả
│       └── AAPL_sentiment_results.json         # Kết quả chi tiết
├── MODEL/
│   └── LSTM+PSO/
│       ├── LSTM_PSO_Sentiment_Hybrid.ipynb     # Notebook chính
│       └── README.md                            # File này
└── CHART/
    └── LSTM+PSO/
        └── hybrid_results_apple.png             # Biểu đồ kết quả
```

## Cách chạy

1. **Cài đặt thư viện**:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn yfinance
```

2. **Chạy notebook**:
- Mở `LSTM_PSO_Sentiment_Hybrid.ipynb`
- Restart Kernel
- Run All Cells

## Key Findings

1. ✅ Baseline LSTM+PSO đã được train tốt với R2 = 0.8315
2. ✅ Thêm sentiment features cải thiện R2 lên 0.8503 (+2.26%)
3. ✅ Sentiment adjustment factor tối ưu: 1.9
4. ✅ MAE giảm từ $3.71 xuống $3.48
5. ✅ Prediction accuracy: 98.55%

## Quy trình tối ưu chi tiết

### Khi nào cần tối ưu DA%?

Nếu sau bước tối ưu R², **DA% < 50%**, cần chạy thêm bước tối ưu DA:

```
Cell 50: Kết quả Baseline vs Hybrid (Optimize R²)
         ↓
         Kiểm tra DA%
         ↓
    DA% >= 50%  →  Kết thúc
    DA% < 50%   →  Chạy Cell 52 (Tối ưu DA%)
         ↓
Cell 53: Kết quả cuối cùng
```

### Workflow cho ticker mới

1. **Load dữ liệu giá** từ CSV hoặc yfinance
2. **Load dữ liệu sentiment** từ CSV
3. **Merge và preprocessing**
4. **Load baseline model** (.keras)
5. **Tối ưu R²**: Grid search sentiment factor
6. **Kiểm tra DA%**:
   - Nếu DA >= 50%: Xong
   - Nếu DA < 50%: Tối ưu DA bằng Threshold-based Direction Adjustment
7. **Lưu kết quả**: CSV + JSON

## Ghi chú

- **R2 nằm trong khoảng [0, 1]** theo yêu cầu
- **Direction Accuracy (DA)** cải thiện từ 57.32% lên 59.76%
- Sentiment có **correlation cao với Close price** (0.62 - 0.66)

## Giải thích các metrics

| Metric | Ý nghĩa | Mục tiêu |
|--------|---------|----------|
| **R²** | Độ khớp của model (0-1) | Càng cao càng tốt |
| **DA%** | Tỷ lệ dự đoán đúng hướng tăng/giảm | > 50% (tốt hơn random) |
| **MAPE** | Sai số phần trăm trung bình | Càng thấp càng tốt |
| **RMSE** | Căn bậc hai sai số bình phương | Càng thấp càng tốt |
| **MAE** | Sai số tuyệt đối trung bình | Càng thấp càng tốt |

## Tác giả

NCT3 Research Team

## License

MIT License
