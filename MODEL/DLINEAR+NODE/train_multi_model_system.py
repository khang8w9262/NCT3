# #!/usr/bin/env python3
# """
# MULTI-MODEL TRAINING SYSTEM
# ===========================
# Train 3 loại mô hình cho từng chứng khoán:
# 1. LSTM Basic - Price data only
# 2. LSTM Sentiment - Price + Sentiment
# 3. Hybrid Model - Advanced architecture
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import os
# import json
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')


# class BasicLSTMModel(nn.Module):
#     """LSTM cơ bản chỉ dùng price data"""

#     def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, dropout=0.2):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.lstm = nn.LSTM(
#             input_dim, hidden_dim, num_layers,
#             dropout=dropout, batch_first=True
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#     def forward(self, x):
#         # LSTM forward
#         lstm_out, _ = self.lstm(x)

#         # Use last timestep output
#         last_output = lstm_out[:, -1, :]

#         # Final prediction
#         output = self.fc(last_output)
#         return output


# class SentimentDLinearNodeModel(nn.Module):
#     """DLinear + NODE kết hợp price + sentiment"""

#     def __init__(self, seq_len=30, price_dim=5, sentiment_dim=10, hidden_dim=128, dropout=0.2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.price_dim = price_dim
#         self.sentiment_dim = sentiment_dim

#         # DLinear components for price
#         self.price_decomp = nn.Sequential(
#             nn.Linear(seq_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, seq_len)
#         )

#         # DLinear components for sentiment
#         self.sentiment_decomp = nn.Sequential(
#             nn.Linear(seq_len, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, seq_len)
#         )

#         # NODE (Neural ODE) component
#         self.node_layers = nn.Sequential(
#             nn.Linear(price_dim + sentiment_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, price_dim + sentiment_dim)
#         )

#         # Final prediction layers
#         self.predictor = nn.Sequential(
#             nn.Linear((price_dim + sentiment_dim) * seq_len, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#     def forward(self, price_x, sentiment_x):
#         batch_size, seq_len, _ = price_x.shape

#         # Combine price and sentiment data
#         # [batch, seq, price_dim + sentiment_dim]
#         combined_input = torch.cat([price_x, sentiment_x], dim=-1)

#         # Apply DLinear decomposition on price features
#         price_trend = self.price_decomp(
#             price_x.transpose(1, 2)).transpose(1, 2)

#         # Apply DLinear decomposition on sentiment features
#         sentiment_trend = self.sentiment_decomp(
#             sentiment_x.transpose(1, 2)).transpose(1, 2)

#         # Combine trends
#         combined_trend = torch.cat([price_trend, sentiment_trend], dim=-1)

#         # Apply NODE (Neural ODE approximation with residual connection)
#         node_output = combined_input + self.node_layers(combined_input)

#         # Combine DLinear trends and NODE output
#         final_features = combined_trend + node_output

#         # Flatten for prediction
#         flattened = final_features.reshape(batch_size, -1)

#         # Final prediction
#         output = self.predictor(flattened)
#         return output

#  # Lấy class HybridDLinearNodeModel (15/12/2025)


# class HybridDLinearNodeModel(nn.Module):

#     """Hybrid DLinear + NODE model với price + residual"""

#     def __init__(self, seq_len=30, price_dim=5, residual_dim=5, hidden_dim=128, dropout=0.2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.price_dim = price_dim
#         self.residual_dim = residual_dim

#         # DLinear trend decomposition
#         self.trend_dlinear = nn.Sequential(
#             nn.Linear(seq_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, seq_len)
#         )

#         # DLinear residual decomposition
#         self.residual_dlinear = nn.Sequential(
#             nn.Linear(seq_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, seq_len)
#         )

#         # NODE (Neural ODE) component for dynamic modeling
#         self.node_func = nn.Sequential(
#             nn.Linear(price_dim + residual_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, price_dim + residual_dim)
#         )

#         # Feature fusion
#         self.fusion = nn.Sequential(
#             nn.Linear((price_dim + residual_dim) * seq_len, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )

#         # Final predictor
#         self.predictor = nn.Linear(hidden_dim // 2, 1)

#     def forward(self, price_x, residual_x):
#         batch_size, seq_len, _ = price_x.shape

#         # Combine price and residual data
#         # [batch, seq, price_dim + residual_dim]
#         combined_input = torch.cat([price_x, residual_x], dim=-1)

#         # DLinear trend extraction on price
#         price_trend = self.trend_dlinear(
#             price_x.transpose(1, 2)).transpose(1, 2)

#         # DLinear residual extraction
#         residual_processed = self.residual_dlinear(
#             residual_x.transpose(1, 2)).transpose(1, 2)

#         # Combine DLinear outputs
#         dlinear_output = torch.cat([price_trend, residual_processed], dim=-1)

#         # NODE dynamics (approximating differential equation)
#         node_output = combined_input + self.node_func(combined_input)

#         # Hybrid fusion: DLinear + NODE
#         hybrid_features = dlinear_output + node_output

#         # Flatten for final prediction
#         flattened = hybrid_features.reshape(batch_size, -1)

#         # Feature fusion and prediction
#         fused = self.fusion(flattened)
#         output = self.predictor(fused)

#         return output


# # Lấy data giá (Bỏ sentiment 15/12/2025)
# class MultiModelDataProcessor:
#     """Data processor cho 3 loại model"""

#     def __init__(self):
#         self.price_scalers = {}
#         self.sentiment_scalers = {}
#         self.feature_scalers = {}

#     def load_stock_data(self, stock_name, data_paths):
#         """Load data từ multiple paths"""
#         data_frames = []

#         for path in data_paths:
#             if os.path.exists(path):
#                 try:
#                     df = pd.read_csv(path, encoding='utf-8-sig')
#                     print(
#                         f"Loaded {stock_name} from {path}: {len(df)} records")
#                     data_frames.append(df)
#                 except Exception as e:
#                     print(f"Error loading {path}: {e}")

#         if not data_frames:
#             return None

#         # Combine data
#         combined_df = pd.concat(data_frames, ignore_index=True)

#         # Handle Vietnamese columns
#         """Load and merge price + sentiment data on Date, ưu tiên ngày có giá, nếu có sentiment thì merge, nếu chỉ có sentiment thì bỏ qua"""
#         if len(data_paths) != 2:
#             print(f"[SKIP] {stock_name}: Need both price and sentiment file.")
#             return None
#         price_path, sentiment_path = data_paths
#         try:
#             price_df = pd.read_csv(price_path, encoding='utf-8-sig')
#             sentiment_df = pd.read_csv(sentiment_path, encoding='utf-8-sig')
#             print(
#                 f"Loaded price: {len(price_df)} | sentiment: {len(sentiment_df)}")
#         except Exception as e:
#             print(f"Error loading data for {stock_name}: {e}")
#             return None

#         # Rename columns for consistency
#         column_mapping = {
#             'Ngày': 'Date', 'Lần cuối': 'Close', 'Mở': 'Open',
#             # Chỉnh sửa lại format data set
#             'Cao': 'High', 'Thấp': 'Low', 'KL': 'Volume', '% Thay đổi': 'Change_Pct'

#         }
#         combined_df = combined_df.rename(columns=column_mapping)

#         # Parse date and clean data
#         combined_df['Date'] = pd.to_datetime(
#             combined_df['Date'], format='%d/%m/%Y', errors='coerce')  # Định dạng date format theo yfinance

#         # Clean numeric columns
#         for col in ['Close', 'Open', 'High', 'Low']:
#             if col in combined_df.columns:
#                 combined_df[col] = pd.to_numeric(
#                     combined_df[col].astype(str).str.replace(',', ''),
#         price_df=price_df.rename(columns=column_mapping)
#         sentiment_df=sentiment_df.rename(columns=column_mapping)

#         # Parse date
#        # Bỏ format cứng, để pandas tự nhận diện YYYY-MM-DD
#         price_df['Date']=pd.to_datetime(price_df['Date'], errors='coerce')
#         sentiment_df['Date']=pd.to_datetime(
#             sentiment_df['Date'], errors='coerce')

#         # Merge: left join, chỉ lấy ngày có giá, nếu có sentiment thì merge, nếu không thì NaN
#         merged_df=pd.merge(price_df, sentiment_df, on='Date',
#                            suffixes=('', '_sentiment'), how='left')

#         # Clean numeric columns
#         for col in ['Close', 'Open', 'High', 'Low']:
#             if col in merged_df.columns:
#                 merged_df[col]=pd.to_numeric(
#                     merged_df[col].astype(str).str.replace(',', ''),
#                     errors='coerce'
#                 )

#         # Remove duplicates and sort
#         combined_df=combined_df.drop_duplicates(
#             subset=['Date']).sort_values('Date')
#         combined_df=combined_df.dropna(subset=['Close', 'Date'])

#         print(f"Final {stock_name} dataset: {len(combined_df)} records")
#         return combined_df

#         merged_df=merged_df.drop_duplicates(
#             subset=['Date']).sort_values('Date')
#         drop_cols=[col for col in ['Close', 'Date']
#             if col in merged_df.columns]
#         if drop_cols:
#             merged_df=merged_df.dropna(subset=drop_cols)
#         print(f"Final {stock_name} merged dataset: {len(merged_df)} records")
#         return merged_df

#     def create_basic_features(self, df):
#         """Tạo basic price features cho LSTM cơ bản"""
#         features=pd.DataFrame()

#         # Basic OHLC
#         features['close']=df['Close']
#         features['open']=df['Open'] if 'Open' in df.columns else df['Close']
#         features['high']=df['High'] if 'High' in df.columns else df['Close']
#         features['low']=df['Low'] if 'Low' in df.columns else df['Close']

#         # Volume (if available)
#         if 'Volume' in df.columns:
#             features['volume']=pd.to_numeric(df['Volume'], errors='coerce')
#         else:
#             features['volume']=1000000

#         # Fill NaN values
#         features=features.fillna(method='bfill').fillna(0)

#         return features

#     def create_sentiment_features(self, df):
#         """Tạo sentiment features"""
#         sentiment_features=pd.DataFrame()

#         # Check if sentiment columns exist
#         sentiment_cols=['sentiment_score', 'impact_score', 'relevance_score',
#                           'confidence', 'short_term_score', 'medium_term_score']

#         found_sentiment=False
#         for col in sentiment_cols:
#             if col in df.columns:
#                 sentiment_features[col]=pd.to_numeric(
#                     df[col], errors='coerce')
#                 found_sentiment=True

#         # If no sentiment data, create dummy features
#         if not found_sentiment:
#             print("No sentiment data found, creating dummy features...")
#             for col in sentiment_cols:
#                 sentiment_features[col]=0.0

#         # Additional derived sentiment features
#         if 'sentiment_score' in sentiment_features.columns:
#             sentiment_features['sentiment_momentum']=sentiment_features['sentiment_score'].rolling(
#                 5).mean()
#             sentiment_features['sentiment_volatility']=sentiment_features['sentiment_score'].rolling(
#                 10).std()
#             sentiment_features['sentiment_trend']=sentiment_features['sentiment_score'].diff(
#             )
#         else:
#             sentiment_features['sentiment_momentum']=0.0
#             sentiment_features['sentiment_volatility']=0.1
#             sentiment_features['sentiment_trend']=0.0

#         # Normalize sentiment to [-1, 1] range
#         for col in sentiment_features.columns:
#             if sentiment_features[col].std() > 0:
#                 sentiment_features[col]=(
#                     sentiment_features[col] - sentiment_features[col].mean()) / sentiment_features[col].std()
#             sentiment_features[col]=np.clip(
#                 sentiment_features[col], -3, 3) / 3  # Scale to [-1, 1]

#         # Fill NaN values
#         sentiment_features=sentiment_features.fillna(0)

#         return sentiment_features

#     def create_hybrid_features(self, df):
#         """Tạo comprehensive features cho hybrid model"""
#         features=pd.DataFrame()

#         # Price features
#         features['close']=df['Close']
#         features['open']=df['Open'] if 'Open' in df.columns else df['Close']
#         features['high']=df['High'] if 'High' in df.columns else df['Close']
#         features['low']=df['Low'] if 'Low' in df.columns else df['Close']

#         # Technical indicators
#         for window in [5, 10, 20]:
#             features[f'sma_{window}']=df['Close'].rolling(window).mean()
#             features[f'std_{window}']=df['Close'].rolling(window).std()
#             features[f'rsi_{window}']=self.calculate_rsi(df['Close'], window)

# Price changes and momentum
features['returns'] = df['Close'].pct_change()
features['returns_lag1'] = features['returns'].shift(1)
features['returns_lag2'] = features['returns'].shift(2)

# MACD
exp1 = df['Close'].ewm(span=12).mean()
exp2 = df['Close'].ewm(span=26).mean()
features['macd'] = exp1 - exp2
features['macd_signal'] = features['macd'].ewm(span=9).mean()

#         # Bollinger Bands
#         sma_20=df['Close'].rolling(20).mean()
#         std_20=df['Close'].rolling(20).std()
#         features['bb_upper']=sma_20 + (std_20 * 2)
#         features['bb_lower']=sma_20 - (std_20 * 2)
#         features['bb_position']=(
#             df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

#         # Volume features
#         if 'Volume' in df.columns:
#             features['volume']=pd.to_numeric(df['Volume'], errors='coerce')
#             features['volume_sma']=features['volume'].rolling(10).mean()
#             features['volume_ratio']=features['volume'] /
#                 features['volume_sma']
#         else:
#             features['volume']=1000000
#             features['volume_sma']=1000000
#             features['volume_ratio']=1.0

#         # Sentiment features (if available)
#         sentiment_cols=['sentiment_score', 'impact_score', 'confidence']
#         for col in sentiment_cols:
#             if col in df.columns:
#                 features[col]=pd.to_numeric(df[col], errors='coerce')
#             else:
#                 features[col]=0.0

#         # Fill NaN values
#         features=features.fillna(method='bfill').fillna(0)

#         return features

#     def calculate_rsi(self, prices, window=14):
#         """Calculate RSI"""
#         delta=prices.diff()
#         gain=(delta.where(delta > 0, 0)).rolling(window=window).mean()
#         loss=(-delta.where(delta < 0, 0)).rolling(window=window).mean()
#         rs=gain / loss
#         rsi=100 - (100 / (1 + rs))
#         return rsi.fillna(50)

#     def prepare_basic_sequences(self, features_df, prices, stock_name, seq_len=60, scaler_key_override=None):
#         """Prepare sequences cho Basic LSTM"""
#         # Scale features
#         scaler_key=scaler_key_override if scaler_key_override else f"{stock_name}_basic"
#         if scaler_key not in self.price_scalers:
#             self.price_scalers[scaler_key]=StandardScaler()
#             features_scaled=self.price_scalers[scaler_key].fit_transform(
#                 features_df)
#         else:
#             features_scaled=self.price_scalers[scaler_key].transform(
#                 features_df)

#         # Scale prices
#         if scaler_key not in self.feature_scalers:
#             self.feature_scalers[scaler_key]=StandardScaler()
#             prices_scaled=self.feature_scalers[scaler_key].fit_transform(
#                 prices.reshape(-1, 1)).flatten()
#         else:
#             prices_scaled=self.feature_scalers[scaler_key].transform(
#                 prices.reshape(-1, 1)).flatten()

#         # Create sequences
#         X, y=[], []
#         for i in range(seq_len, len(features_scaled)):
#             X.append(features_scaled[i-seq_len:i])
#             y.append(prices_scaled[i])

#         return np.array(X), np.array(y)

#     def prepare_sentiment_sequences(self, price_features, sentiment_features, prices, stock_name, seq_len=60):
#         """Prepare sequences cho Sentiment LSTM"""
#         # Scale price features
#         price_key=f"{stock_name}_price"
#         if price_key not in self.price_scalers:
#             self.price_scalers[price_key]=StandardScaler()
#             price_scaled=self.price_scalers[price_key].fit_transform(
#                 price_features)
#         else:
#             price_scaled=self.price_scalers[price_key].transform(
#                 price_features)

#         # Scale sentiment features
#         sentiment_key=f"{stock_name}_sentiment"
#         if sentiment_key not in self.sentiment_scalers:
#             self.sentiment_scalers[sentiment_key]=StandardScaler()
#             sentiment_scaled=self.sentiment_scalers[sentiment_key].fit_transform(
#                 sentiment_features)
#         else:
#             sentiment_scaled=self.sentiment_scalers[sentiment_key].transform(
#                 sentiment_features)

#         # Scale prices
#         target_key=f"{stock_name}_target"
#         if target_key not in self.feature_scalers:
#             self.feature_scalers[target_key]=StandardScaler()
#             prices_scaled=self.feature_scalers[target_key].fit_transform(
#                 prices.reshape(-1, 1)).flatten()
#         else:
#             prices_scaled=self.feature_scalers[target_key].transform(
#                 prices.reshape(-1, 1)).flatten()

#         # Create sequences
#         X_price, X_sentiment, y=[], [], []
#         min_len=min(len(price_scaled), len(sentiment_scaled))

#         for i in range(seq_len, min_len):
#             X_price.append(price_scaled[i-seq_len:i])
#             X_sentiment.append(sentiment_scaled[i-seq_len:i])
#             y.append(prices_scaled[i])

#         return np.array(X_price), np.array(X_sentiment), np.array(y)

#     def prepare_hybrid_sequences(self, features_df, prices, stock_name, seq_len=60):
#         """Prepare sequences cho Hybrid model"""
#         # Scale features
#         scaler_key=f"{stock_name}_hybrid"
#         if scaler_key not in self.feature_scalers:
#             self.feature_scalers[scaler_key]=StandardScaler()
#             features_scaled=self.feature_scalers[scaler_key].fit_transform(
#                 features_df)
#         else:
#             features_scaled=self.feature_scalers[scaler_key].transform(
#                 features_df)

#         # Scale prices
#         target_key=f"{stock_name}_hybrid_target"
#         if target_key not in self.price_scalers:
#             self.price_scalers[target_key]=StandardScaler()
#             prices_scaled=self.price_scalers[target_key].fit_transform(
#                 prices.reshape(-1, 1)).flatten()
#         else:
#             prices_scaled=self.price_scalers[target_key].transform(
#                 prices.reshape(-1, 1)).flatten()

#         # Create sequences
#         X, y=[], []
#         for i in range(seq_len, len(features_scaled)):
#             X.append(features_scaled[i-seq_len:i])
#             y.append(prices_scaled[i])

#         return np.array(X), np.array(y)


# class MultiModelTrainer:
#     """Trainer cho 3 loại model"""

#     def __init__(self, save_dir='model'):
#         self.save_dir=save_dir
#         self.processor=MultiModelDataProcessor()
#         self.device=torch.device(
#             'cuda' if torch.cuda.is_available() else 'cpu')

#         os.makedirs(save_dir, exist_ok=True)
#         print(f"Using device: {self.device}")

#     def train_single_model(self, model, train_loader, val_loader, model_name, stock_name, epochs=100):
#         """Train một model"""
#         print(f"\nTraining {model_name} for {stock_name}...")

#         optimizer=optim.Adam(model.parameters(), lr=0.001)
#         scheduler=optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='max', factor=0.8, patience=10)
#         criterion=nn.MSELoss()

#         best_r2=-float('inf')
#         patience_counter=0
#         patience=20

#         history={'train_loss': [], 'val_loss': [], 'val_r2': []}

#         for epoch in range(epochs):
#             # Training
#             model.train()
#             train_losses=[]

#             for batch_data in train_loader:
#                 if model_name == 'sentiment':
#                     # Sentiment DLinear+NODE model expects 2 inputs: price + sentiment
#                     X_price, X_sentiment, y=batch_data
#                     X_price=X_price.to(self.device)
#                     X_sentiment=X_sentiment.to(self.device)
#                     y=y.to(self.device).unsqueeze(1)

#                     optimizer.zero_grad()
#                     outputs=model(X_price, X_sentiment)
#                 elif model_name == 'hybrid':
#                     # Hybrid DLinear+NODE model expects 2 inputs: price + residual
#                     X_price, X_residual, y=batch_data
#                     X_price=X_price.to(self.device)
#                     X_residual=X_residual.to(self.device)
#                     y=y.to(self.device).unsqueeze(1)

#                     optimizer.zero_grad()
#                     outputs=model(X_price, X_residual)
#                 else:
#                     # Basic LSTM model expects 1 input
#                     X, y=batch_data
#                     X=X.to(self.device)
#                     y=y.to(self.device).unsqueeze(1)

#                     optimizer.zero_grad()
#                     outputs=model(X)

#                 loss=criterion(outputs, y)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(), max_norm=1.0)
#                 optimizer.step()

#                 train_losses.append(loss.item())

#             avg_train_loss=np.mean(train_losses)

#             # Validation
#             model.eval()
#             val_losses=[]
#             all_preds, all_targets=[], []

#             with torch.no_grad():
#                 for batch_data in val_loader:
#                     if model_name == 'sentiment':
#                         X_price, X_sentiment, y=batch_data
#                         X_price=X_price.to(self.device)
#                         X_sentiment=X_sentiment.to(self.device)
#                         y=y.to(self.device).unsqueeze(1)
#                         outputs=model(X_price, X_sentiment)
#                     elif model_name == 'hybrid':
#                         X_price, X_residual, y=batch_data
#                         X_price=X_price.to(self.device)
#                         X_residual=X_residual.to(self.device)
#                         y=y.to(self.device).unsqueeze(1)
#                         outputs=model(X_price, X_residual)
#                     else:
#                         X, y=batch_data
#                         X=X.to(self.device)
#                         y=y.to(self.device).unsqueeze(1)
#                         outputs=model(X)

#                     loss=criterion(outputs, y)
#                     val_losses.append(loss.item())

#                     all_preds.extend(outputs.cpu().numpy().flatten())
#                     all_targets.extend(y.cpu().numpy().flatten())

#             avg_val_loss=np.mean(val_losses)
#             val_r2=r2_score(all_targets, all_preds)

#             # Learning rate scheduling
#             scheduler.step(val_r2)

#             # Save best model
#             if val_r2 > best_r2:
#                 best_r2=val_r2
#                 patience_counter=0

#                 # Save model
#                 model_path=f"{self.save_dir}/{stock_name}_{model_name}_model.pt"
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'best_r2': best_r2,
#                     'epoch': epoch,
#                     'model_type': model_name,
#                     'stock_name': stock_name
#                 }, model_path)

#                 status="NEW BEST!"
#             else:
#                 patience_counter += 1
#                 status=f"Plateau ({patience_counter}/{patience})"

#             # Record history
#             history['train_loss'].append(avg_train_loss)
#             history['val_loss'].append(avg_val_loss)
#             history['val_r2'].append(val_r2)

#             # Progress reporting
#             if epoch % 20 == 0 or val_r2 > best_r2:
#                 print(
#                     f"  Epoch {epoch:3d}: Loss={avg_train_loss:.6f}, Val_R²={val_r2:.4f} - {status}")

#             # Early stopping
#             if patience_counter >= patience:
#                 print(f"  Early stopping after {epoch} epochs")
#                 break

#         print(f"  Final R² for {model_name}: {best_r2:.4f} ({best_r2:.2%})")
#         return best_r2, history

#     def train_all_models_for_stock(self, stock_name, data_paths, epochs=100):
#         """Train tất cả 3 model cho 1 chứng khoán"""
#         print(f"\n{'='*60}")
#         print(f"TRAINING ALL MODELS FOR {stock_name}")
#         print(f"{'='*60}")

#         # Load data
#         df=self.processor.load_stock_data(stock_name, data_paths)
#         if df is None or len(df) < 200:
#             print(f"Insufficient data for {stock_name}")
#             return None

#         if 'Close' not in df.columns:
#             print(
#                 f"[SKIP] {stock_name}: No 'Close' column found in data. Skipping training for this stock.")
#             return None
#         prices=df['Close'].values
#         results={}

#         # 1. Train Basic LSTM
#         print(f"\n1. BASIC LSTM MODEL")
#         print("-" * 30)
#         basic_features=self.processor.create_basic_features(df)
#         X_basic, y_basic=self.processor.prepare_basic_sequences(
#             basic_features, prices, stock_name)

#         if len(X_basic) > 0:
#             # Split data
#             train_size=int(0.8 * len(X_basic))
#             X_train, X_val=X_basic[:train_size], X_basic[train_size:]
#             y_train, y_val=y_basic[:train_size], y_basic[train_size:]

#             # Create data loaders
#             train_dataset=TensorDataset(torch.FloatTensor(
#                 X_train), torch.FloatTensor(y_train))
#             val_dataset=TensorDataset(
#                 torch.FloatTensor(X_val), torch.FloatTensor(y_val))
#             train_loader=DataLoader(
#                 train_dataset, batch_size=32, shuffle=True)
#             val_loader=DataLoader(val_dataset, batch_size=32)

#             # Create and train model
#             basic_model=BasicLSTMModel(
#                 input_dim=X_basic.shape[2]).to(self.device)
#             basic_r2, basic_history=self.train_single_model(
#                 basic_model, train_loader, val_loader, 'basic', stock_name, epochs
#             )
#             results['basic']={'r2': basic_r2, 'history': basic_history}

#         # 2. Train Sentiment LSTM (if sentiment data available)
#         print(f"\n2. SENTIMENT DLINEAR+NODE MODEL")
#         print("-" * 30)
#         price_features=self.processor.create_basic_features(df)
#         sentiment_features=self.processor.create_sentiment_features(df)

#         X_price, X_sentiment, y_sentiment=self.processor.prepare_sentiment_sequences(
#             price_features, sentiment_features, prices, stock_name
#         )

#         if len(X_price) > 0:
#             # Split data
#             train_size=int(0.8 * len(X_price))
#             X_price_train, X_price_val=X_price[:
#                                                  train_size], X_price[train_size:]
#             X_sent_train, X_sent_val=X_sentiment[:
#                                                    train_size], X_sentiment[train_size:]
#             y_sent_train, y_sent_val=y_sentiment[:
#                                                    train_size], y_sentiment[train_size:]

#             # Create data loaders
#             train_dataset=TensorDataset(
#                 torch.FloatTensor(X_price_train),
#                 torch.FloatTensor(X_sent_train),
#                 torch.FloatTensor(y_sent_train)
#             )
#             val_dataset=TensorDataset(
#                 torch.FloatTensor(X_price_val),
#                 torch.FloatTensor(X_sent_val),
#                 torch.FloatTensor(y_sent_val)
#             )
#             train_loader=DataLoader(
#                 train_dataset, batch_size=32, shuffle=True)
#             val_loader=DataLoader(val_dataset, batch_size=32)

#             # Create and train model
#             sentiment_model=SentimentDLinearNodeModel(
#                 seq_len=X_price.shape[1],
#                 price_dim=X_price.shape[2],
#                 sentiment_dim=X_sentiment.shape[2]
#             ).to(self.device)
#             sentiment_r2, sentiment_history=self.train_single_model(
#                 sentiment_model, train_loader, val_loader, 'sentiment', stock_name, epochs
#             )
#             results['sentiment']={
#                 'r2': sentiment_r2, 'history': sentiment_history}

#         # 3. Train Hybrid Model (DLinear + NODE với price + residual)
#         print(f"\n3. HYBRID DLINEAR+NODE MODEL")
#         print("-" * 30)

#         # Create separate price and residual features for hybrid model
#         price_features=self.processor.create_basic_features(df)

#         # Create residual features (price residuals from trend)
#         residual_features=df[['Open', 'High',
#                                 'Low', 'Close', 'Volume']].copy()

#         # Clean volume data - convert strings like '69.84M' to numeric
#         residual_features['Volume']=pd.to_numeric(
#             residual_features['Volume'], errors='coerce')

#         # Simple residual calculation: actual - moving average
#         for col in ['Open', 'High', 'Low', 'Close']:
#             residual_features[col]=residual_features[col] -
#                 residual_features[col].rolling(window=20, min_periods=1).mean()

#         # Keep same column names as basic features for scaler compatibility
#         residual_features.columns=['open', 'high', 'low', 'close', 'volume']
#         residual_features=residual_features.fillna(0)

#         # Prepare sequences for hybrid model - use separate scaler key for residual
#         X_price_hybrid, y_price_hybrid=self.processor.prepare_basic_sequences(
#             price_features, prices, stock_name)

#         # Use custom scaler key for residual features to avoid name conflict
#         original_scaler_key=f"{stock_name}_basic"
#         residual_scaler_key=f"{stock_name}_residual"

#         # The prepare_basic_sequences method will automatically create and fit the scaler if it doesn't exist
#         X_residual, y_hybrid=self.processor.prepare_basic_sequences(
#             residual_features, prices, stock_name, scaler_key_override=residual_scaler_key)

#         if len(X_price_hybrid) > 0 and len(X_residual) > 0:
#             # Ensure same length
#             min_len=min(len(X_price_hybrid), len(X_residual))
#             X_price_hybrid=X_price_hybrid[:min_len]
#             X_residual=X_residual[:min_len]
#             y_hybrid=y_hybrid[:min_len]

#             # Split data
#             train_size=int(0.8 * len(X_price_hybrid))
#             X_price_train, X_price_val=X_price_hybrid[:
#                                                         train_size], X_price_hybrid[train_size:]
#             X_resid_train, X_resid_val=X_residual[:
#                                                     train_size], X_residual[train_size:]
#             y_train, y_val=y_hybrid[:train_size], y_hybrid[train_size:]

#             # Create data loaders with 2 inputs
#             train_dataset=TensorDataset(
#                 torch.FloatTensor(X_price_train),
#                 torch.FloatTensor(X_resid_train),
#                 torch.FloatTensor(y_train)
#             )
#             val_dataset=TensorDataset(
#                 torch.FloatTensor(X_price_val),
#                 torch.FloatTensor(X_resid_val),
#                 torch.FloatTensor(y_val)
#             )
#             train_loader=DataLoader(
#                 train_dataset, batch_size=32, shuffle=True)
#             val_loader=DataLoader(val_dataset, batch_size=32)

#             # Create and train hybrid DLinear+NODE model
#             hybrid_model=HybridDLinearNodeModel(
#                 seq_len=X_price_hybrid.shape[1],
#                 price_dim=X_price_hybrid.shape[2],
#                 residual_dim=X_residual.shape[2]
#             ).to(self.device)
#             hybrid_r2, hybrid_history=self.train_single_model(
#                 hybrid_model, train_loader, val_loader, 'hybrid', stock_name, epochs
#             )
#             results['hybrid']={'r2': hybrid_r2, 'history': hybrid_history}

#         # Save results summary
#         summary_path=f"{self.save_dir}/{stock_name}_training_summary.json"
#         with open(summary_path, 'w') as f:
#             json.dump({
#                 'stock_name': stock_name,
#                 'timestamp': datetime.now().isoformat(),
#                 'results': {k: {'r2': v['r2']} for k, v in results.items()},
#                 'data_info': {
#                     'total_records': len(df),
#                     'date_range': f"{df['Date'].min()} to {df['Date'].max()}"
#                 }
#             }, f, indent=2, default=str)

#         return results


# def main():
#     # Setup paths
#     base_dir=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#     base_price=os.path.join(base_dir, 'DATASET', 'PRICE')
#     base_sentiment=os.path.join(base_dir, 'DATASET', 'SENTIMENT')

#     print("MULTI-MODEL TRAINING SYSTEM")
#     print("============================================================")
#     print("Training 3 models for each stock:")
#     print("1. Basic LSTM (price only)")
#     print("2. Sentiment DLinear+NODE (price + sentiment)")
#     print("3. Hybrid DLinear+NODE (price + residual)")
#     print()

#     # Define stocks and their data paths
#     stocks_config={
#         'META': ['CamXuc/META.csv', 'Train/META.csv'],
#         'FPT': ['CamXuc/FPT.csv', 'Train/FPT.csv'],
#         # 'APPLE': ['Train/Apple.csv'],  # Temporarily disabled - no sentiment data
#         'MBB': ['CamXuc/MBB.csv', 'Train/MBB.csv']
#     }

#     trainer=MultiModelTrainer()
#     all_results={}

#     # Train models for each stock
#     for stock_name, data_paths in stocks_config.items():
#         print(f"\nProcessing {stock_name}...")

#         # Check if data files exist
#         existing_paths=[path for path in data_paths if os.path.exists(path)]
#         if not existing_paths:
#             print(f"No data files found for {stock_name}, skipping...")
#             continue

#         results=trainer.train_all_models_for_stock(
#             stock_name, existing_paths, epochs=150)
#         if results:
#             all_results[stock_name]=results

#     # Find matched files
#     price_files={}
#     if os.path.isdir(base_price):
#         for f in os.listdir(base_price):
#             if f.endswith('_stock_data.csv'):
#                 stem=f.replace('_stock_data.csv', '').upper()
#                 price_files[stem]=os.path.join(base_price, f)

#     sentiment_files={}
#     if os.path.isdir(base_sentiment):
#         for f in os.listdir(base_sentiment):
#             if f.lower().endswith('_sentiment.csv'):
#                 stem=os.path.splitext(f)[0].replace('_sentiment', '').upper()
#                 sentiment_files[stem]=os.path.join(base_sentiment, f)

#     # Chỉ train cho các mã có cả file giá và file sentiment
#     stocks=sorted(set(sentiment_files.keys()) & set(price_files.keys()))

#     # --- SỬA Ở ĐÂY: Thêm save_dir='LOGS' ---
#     trainer=MultiModelTrainer(save_dir='LOGS')

#     all_results={}

#     for stock in stocks:
#         data_paths=[price_files[stock], sentiment_files[stock]]
#         print(f"\nProcessing {stock} with paths: {data_paths}")
#         results=trainer.train_all_models_for_stock(
#             stock, data_paths, epochs=150)
#         if results:
#             all_results[stock]=results

#     # Print final summary
#     print(f"\n{'='*80}")
#     print("FINAL TRAINING SUMMARY")
#     print(f"{'='*80}")

#     for stock_name, stock_results in all_results.items():
#         print(f"\n{stock_name}:")
#         for model_type, model_result in stock_results.items():
#             r2_score=model_result['r2']
#             print(f"  {model_type:12s}: R² = {r2_score:.4f} ({r2_score:.2%})")

#     print(f"\n✅ Training completed! Models saved in {trainer.save_dir}/")
#     print("📊 Individual model files: [STOCK]_[MODEL]_model.pt")
#     print("📋 Training summaries: [STOCK]_training_summary.json")
#     # In thông báo đã lưu vào LOGS
#     print(f"\n Training completed! Models saved in {trainer.save_dir}/")
#     print(" Individual model files: LOGS/[STOCK]_[MODEL]_model.pt")
#     print(" Training summaries: LOGS/[STOCK]_training_summary.json")

#     return all_results


# if __name__ == "__main__":
#     results=main()
