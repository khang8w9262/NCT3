Project: NCT3


NCT3/
├─ README.md
├─ readme.txt
├─ CHART/
│  ├─ DLINEAR+NODE/
│  └─ LSTM+PSO/
├─ DATASET/
│  ├─ PRICE/
│  └─ SENTIMENT/
├─ LOGS/
│  └─ DLINEAR+NODE/
├─ MODEL/
│  ├─ DLinear/
│  ├─ DLINEAR+NODE/
│  └─ LSTM+PSO/
└─ SRC/
   └─ CRAWLING/


Cấu trúc thư mục chính
- CHART/
  - Chứa các biểu đồ kết quả thí nghiệm (visualizations).
  - Thư mục con ví dụ: `DLINEAR+NODE/`, `LSTM+PSO/` tương ứng với từng tổ hợp mô hình/thuật toán.

- DATASET/
  - PRICE/: Dữ liệu giá cổ phiếu và các phiên bản (raw/adjusted) cho từng mã (ví dụ: `Apple.csv`, `Alibaba_stock_data_adj_close.csv`).
  - SENTIMENT/: Dữ liệu sentiment (mỗi mã có file sentiment riêng, ví dụ `APPLE_sentiment.csv`).

- LOGS/
  - Chứa checkpoints mô hình (`*.pt`), lịch sử huấn luyện (`*_training_history.json`), cấu hình/metadata thử nghiệm (`*_Hybrid.json`) và báo cáo/evaluation (`*.csv`).
  - Thư mục con theo mô hình/experiments (ví dụ `DLINEAR+NODE/`, `LSTM+PSO/`).
  - Tên file phổ biến: `*_best_sentiment_model.pt` (checkpoint tốt nhất với dữ liệu sentiment), `*_full_sentiment_best_sentiment_model.pt`, `*_nan_sentiment_best_sentiment_model.pt` (các biến thể xử lý thiếu dữ liệu), `*_Hybrid.pt` (mô hình hybrid), `*_training_history.json` (loss/metrics theo epoch).

- MODEL/
  - Lưu trữ outputs đã xử lý, kết quả model ở dạng CSV hoặc các artefact liên quan (ví dụ `DLinear/` chứa file `.csv` kết quả cho AAPL, ...).

- SRC/
  - Mã nguồn (scripts, mô-đun) của dự án.
  - `CRAWLING/` chứa script thu thập dữ liệu (crawlers hoặc extractor).

File ở thư mục gốc
- `README.md`: file markdown chính (nội dung hiện có trong repo).

Gợi ý sử dụng nhanh
- Để xem dữ liệu thô: mở `DATASET/PRICE` và `DATASET/SENTIMENT`.
- Để chạy hoặc sửa code: xem `SRC/` (tìm file train/eval cụ thể theo mô-đun).
- Checkpoints và lịch sử huấn luyện: kiểm tra `LOGS/` tương ứng với mô hình/experiment.
- Biểu đồ kết quả: xem `CHART/` theo thư mục mô hình.

Ghi chú
- Tên file và cấu trúc thư mục có thể được mở rộng theo từng thí nghiệm; nếu cần chi tiết cho một subfolder cụ thể (ví dụ tên file checkpoint theo thử nghiệm), hãy cho biết mã/mô hình muốn giải thích để cập nhật thêm.

Người liên hệ
- Chủ repo: người phát triển trong repo (liên hệ theo thông tin dự án).
