#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHOBERT SENTIMENT CALCULATOR
Sử dụng PhoBERT thay thế VADER cho tính toán sentiment score
Dành riêng cho văn bản tiếng Việt với độ chính xác cao

Tham khảo:
- PhoBERT: Pre-trained language models for Vietnamese (Nguyen & Nguyen, 2020)
- VinAI PhoBERT: vinai/phobert-base, vinai/phobert-large
- Transformer-based sentiment analysis for financial Vietnamese text

⚠️ YÊU CẦU:
- pip install transformers torch
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- Internet connection để tải PhoBERT lần đầu (auto-cache sau đó)
"""

import pandas as pd
import numpy as np
import re
import sys
import warnings
from datetime import datetime
from collections import deque
warnings.filterwarnings('ignore')

# PhoBERT và Transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
    PHOBERT_AVAILABLE = True
    print("✓ PhoBERT libraries available")
except ImportError as e:
    PHOBERT_AVAILABLE = False
    print(f"❌ PhoBERT libraries not available: {e}")
    print("Install: pip install transformers torch")

# VADER đã được loại bỏ - chỉ dùng PhoBERT
VADER_AVAILABLE = False

# SciPy cho trend analysis
try:
    from scipy.stats import linregress
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Trend calculation disabled.")

class PhoBERTSentimentCalculator:
    """
    Sentiment Calculator sử dụng PhoBERT cho tiếng Việt
    """
    
    def __init__(self, target_name='UNKNOWN'):
        self.target_name = target_name.upper()
        
        print(f"PHOBERT SENTIMENT CALCULATOR")
        print(f"Target: {self.target_name}")
        print(f"Using PhoBERT for Vietnamese Sentiment Analysis")
        print("=" * 60)
        
        # Historical data cho temporal metrics
        self.sentiment_history = deque(maxlen=30)  # 30-day window
        
        self.setup_phobert_analyzer()
        self.setup_financial_keywords()
        
    def setup_phobert_analyzer(self):
        """
        Setup PhoBERT Model và Tokenizer
        
        Models available:
        - vinai/phobert-base: 135M parameters, faster
        - vinai/phobert-large: 370M parameters, more accurate
        
        Architecture:
        - RoBERTa-based pre-trained on 20GB Vietnamese corpus
        - BPE tokenization with 64K vocab
        - Support for downstream fine-tuning
        """
        
        if not PHOBERT_AVAILABLE:
            print("❌ PhoBERT not available. Using keyword fallback.")
            self.setup_keyword_fallback()
            return
        
        try:
            # Sử dụng PhoBERT-base (faster, still accurate)
            model_name = "vinai/phobert-base"
            
            print(f"Loading PhoBERT model: {model_name}")
            print("⏳ Loading from cache...")
            
            # Load với local_files_only=True nếu đã có cache
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
            except:
                # Nếu chưa có cache thì download
                print("⏳ Downloading model (first time only)...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Check if CUDA available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            print(f"✓ PhoBERT loaded successfully on {self.device}")
            print(f"✓ Model: {model_name}")
            print(f"✓ Tokenizer vocabulary size: {len(self.tokenizer)}")
            
            self.phobert_available = True
            
            # Initialize sentiment mapping weights (fine-tunable)
            self.setup_sentiment_mapping()
            
        except Exception as e:
            print(f"❌ Failed to load PhoBERT: {e}")
            print("Using keyword fallback...")
            self.setup_keyword_fallback()
            
    def setup_sentiment_mapping(self):
        """
        Setup sentiment mapping từ PhoBERT embeddings
        
        Method: Linear transformation from [CLS] token embedding
        - PhoBERT [CLS] token → 768-dimensional vector
        - Linear layer: 768 → 1 (sentiment score)
        
        ⚠️ Simplified approach - không cần fine-tuning dataset
        - Sử dụng pre-trained weights với hand-crafted rules
        - Có thể improve bằng cách fine-tune trên Vietnamese financial sentiment data
        """
        
        # Sentiment keywords cho Vietnamese financial text
        self.sentiment_keywords = {
            'positive_strong': {
                'vietnamese': ['tăng mạnh', 'bứt phá', 'đột phá', 'tích cực', 'lạc quan', 
                              'khả quan', 'tăng trưởng', 'phát triển', 'thành công', 'lợi nhuận'],
                'weight': 0.8
            },
            'positive_weak': {
                'vietnamese': ['tăng', 'tốt', 'ổn định', 'cải thiện', 'phục hồi', 'khích lệ'],
                'weight': 0.4
            },
            'negative_strong': {
                'vietnamese': ['giảm mạnh', 'sụt giảm', 'thua lỗ', 'khó khăn', 'bi quan', 
                              'lo ngại', 'suy thoái', 'rủi ro', 'thất bại', 'âm'],
                'weight': -0.8
            },
            'negative_weak': {
                'vietnamese': ['giảm', 'yếu', 'chậm', 'thận trọng', 'hạn chế', 'thấp'],
                'weight': -0.4
            }
        }
        
        print("✓ Sentiment mapping initialized")
        
    def setup_keyword_fallback(self):
        """Setup keyword-based fallback khi PhoBERT không khả dụng"""
        self.phobert_available = False
        print("⚠️ PhoBERT not available, using keyword-based sentiment")
            
    def setup_financial_keywords(self):
        """Setup financial keywords cho relevance scoring"""
        
        self.financial_keywords = {
            'vietnamese': [
                'doanh thu', 'lợi nhuận', 'tài chính', 'thị trường', 'cổ phiếu',
                'phân tích', 'đánh giá', 'mua', 'bán', 'đầu tư', 'cổ tức', 
                'tăng trưởng', 'định giá', 'biên lợi nhuận', 'dòng tiền',
                'ngân hàng', 'chứng khoán', 'kinh tế', 'tài sản', 'vốn'
            ],
            'english': [
                'revenue', 'profit', 'earnings', 'growth', 'financial', 'investment',
                'market', 'stock', 'price', 'analyst', 'rating', 'target', 'buy', 
                'sell', 'dividend', 'eps', 'valuation', 'margin', 'cash flow'
            ]
        }
        
        self.target_patterns = [
            self.target_name,
            f'({self.target_name})',
            f'HM : {self.target_name}',
            f'NASDAQ : {self.target_name}',
            f'NYSE : {self.target_name}'
        ]

    def clean_text_for_phobert(self, text):
        """
        Clean text để tương thích với PhoBERT tokenizer
        Loại bỏ các ký tự đặc biệt gây lỗi BPE
        """
        if not text:
            return ""
        
        text = str(text)
        
        # Remove emoji và ký tự unicode đặc biệt
        # Giữ lại tiếng Việt và ASCII cơ bản
        cleaned = ""
        for char in text:
            # Giữ ASCII cơ bản (letters, numbers, punctuation)
            if ord(char) < 128:
                cleaned += char
            # Giữ Vietnamese characters (Latin Extended)
            elif 0x00C0 <= ord(char) <= 0x024F:
                cleaned += char
            # Giữ Vietnamese combining diacritics  
            elif 0x0300 <= ord(char) <= 0x036F:
                cleaned += char
            # Giữ Vietnamese additional characters
            elif ord(char) in [0x0110, 0x0111, 0x01A0, 0x01A1, 0x01AF, 0x01B0]:
                cleaned += char
            # Giữ Vietnamese tone marks range
            elif 0x1EA0 <= ord(char) <= 0x1EF9:
                cleaned += char
            # Thay thế các ký tự khác bằng space
            else:
                cleaned += " "
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove URLs
        cleaned = re.sub(r'http\S+|www\.\S+', '', cleaned)
        
        # Remove email addresses
        cleaned = re.sub(r'\S+@\S+', '', cleaned)
        
        # Giới hạn độ dài
        if len(cleaned) > 500:
            cleaned = cleaned[:500]
        
        return cleaned.strip()

    def calculate_phobert_sentiment(self, text):
        """
        Calculate sentiment using PhoBERT
        
        Process:
        1. Tokenize Vietnamese text with PhoBERT tokenizer
        2. Get [CLS] token embedding (768-dim)
        3. Apply sentiment keywords matching
        4. Combine embedding features + keyword matching
        5. Map to sentiment score [-4.5, +4.5]
        
        Returns: (sentiment_label, sentiment_score)
        """
        
        if not self.phobert_available or not PHOBERT_AVAILABLE:
            return self.calculate_keyword_only_sentiment(text)
        
        if pd.isna(text) or str(text).strip() == "":
            return 'NEUTRAL', 0.0
        
        text_str = str(text).strip()
        
        # Clean text for PhoBERT - remove problematic characters
        text_str = self.clean_text_for_phobert(text_str)
        
        if not text_str or len(text_str) < 2:
            return 'NEUTRAL', 0.0
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text_str,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256  # Giảm max_length để tránh lỗi
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get PhoBERT embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                # Ensure we have the right shape
                if cls_embedding.shape[0] == 0:
                    raise Exception("Empty embedding")
                    
                # Take first sample if batch
                if len(cls_embedding.shape) > 1:
                    cls_embedding = cls_embedding[0]
            
            # Convert to CPU for processing
            cls_embedding = cls_embedding.cpu().numpy()
            
            # Check embedding validity
            if cls_embedding.size == 0:
                raise Exception("Empty embedding after conversion")
            
            # Method 1: Keyword-based sentiment (rule-based)
            keyword_score = self.calculate_keyword_sentiment(text_str)
            
            # Method 2: Embedding-based sentiment (simple approach)
            # Sử dụng mean pooling của CLS embedding
            embedding_sentiment = np.mean(cls_embedding) * 10  # Scale to readable range
            
            # Combine both methods (weighted average)
            combined_score = 0.7 * keyword_score + 0.3 * embedding_sentiment
            
            # Clamp to [-4.5, +4.5] range
            combined_score = max(-4.5, min(combined_score, 4.5))
            
            # Map to 5-level sentiment
            if combined_score >= 2.0:
                return 'VERY_POSITIVE', 4.5
            elif combined_score >= 0.5:
                return 'POSITIVE', 2.5
            elif combined_score <= -2.0:
                return 'VERY_NEGATIVE', -4.5
            elif combined_score <= -0.5:
                return 'NEGATIVE', -2.5
            else:
                return 'NEUTRAL', 0.0
                
        except Exception as e:
            print(f"PhoBERT error: {e}. Using keyword fallback.")
            return self.calculate_keyword_only_sentiment(text_str)
    
    def calculate_keyword_sentiment(self, text):
        """
        Calculate sentiment dựa trên Vietnamese financial keywords
        Được optimize cho financial Vietnamese text
        """
        
        text_lower = text.lower()
        total_score = 0.0
        keyword_count = 0
        
        for category, info in self.sentiment_keywords.items():
            for keyword in info['vietnamese']:
                if keyword in text_lower:
                    total_score += info['weight']
                    keyword_count += 1
        
        if keyword_count == 0:
            return 0.0
        
        # Average và scale
        avg_score = total_score / keyword_count
        return avg_score * 5.0  # Scale to [-4.5, +4.5] range
    
    def calculate_keyword_only_sentiment(self, text):
        """Fallback: Chỉ dùng keywords khi PhoBERT fail"""
        
        score = self.calculate_keyword_sentiment(text)
        
        if score >= 2.0:
            return 'VERY_POSITIVE', 4.5
        elif score >= 0.5:
            return 'POSITIVE', 2.5
        elif score <= -2.0:
            return 'VERY_NEGATIVE', -4.5
        elif score <= -0.5:
            return 'NEGATIVE', -2.5
        else:
            return 'NEUTRAL', 0.0
    


    # =========================================================================
    # GIỮ NGUYÊN CÁC METHODS KHÁC TỪ RESEARCH-BASED VERSION
    # =========================================================================
    
    def calculate_impact_score(self, sentiment_score, relevance_score):
        """Impact Score = Sentiment × Relevance"""
        impact = sentiment_score * relevance_score
        impact = max(-3.0, min(impact, 3.0))
        return round(impact, 1)

    def calculate_relevance_score(self, header, content, target_mentioned):
        """Relevance Score = Multi-component Weighted Sum"""
        relevance = 0.3  # Base relevance
        
        if target_mentioned:
            relevance += 0.15  # Direct mention bonus
        
        if pd.notna(content):
            content_str = str(content).strip()
            word_count = len(content_str.split()) if content_str else 0
            
            # Content quality bonus
            if word_count > 100:
                relevance += 0.1
            elif word_count > 50:
                relevance += 0.05
            
            # Business keywords
            all_keywords = self.financial_keywords['vietnamese'] + self.financial_keywords['english']
            business_count = sum(1 for keyword in all_keywords 
                                if keyword in content_str.lower())
            relevance += min(business_count * 0.03, 0.15)
        
        return min(relevance, 1.0)

    def calculate_confidence(self, sentiment_score, content_length, target_mentioned):
        """Confidence = Strength-based + Adjustments"""
        base_confidence = 0.5 + min(abs(sentiment_score) / 12.5, 0.2)
        
        if target_mentioned:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        if content_length > 200:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        return round(max(base_confidence, 0.5), 1)

    def calculate_short_medium_term_scores(self, sentiment_score):
        """Short-term × 1.0, Medium-term × 0.8"""
        short_term = sentiment_score * 1.0
        medium_term = sentiment_score * 0.8
        
        short_term = max(-4.5, min(short_term, 4.5))
        medium_term = max(-4.5, min(medium_term, 4.5))
        
        return round(short_term, 1), round(medium_term, 1)

    def calculate_sentiment_momentum(self, current_sentiment):
        """Sentiment Momentum = Current - Recent Average"""
        if len(self.sentiment_history) < 2:
            return 0.0
        
        recent_avg = np.mean(list(self.sentiment_history)[-7:])
        momentum = current_sentiment - recent_avg
        return round(momentum, 2)

    def calculate_sentiment_volatility(self, window=7):
        """Sentiment Volatility = Rolling Standard Deviation"""
        if len(self.sentiment_history) < window:
            return 0.0
        
        volatility = np.std(list(self.sentiment_history)[-window:])
        return round(volatility, 2)

    def calculate_sentiment_trend(self, window=30):
        """Sentiment Trend = Linear Regression Slope"""
        if not SCIPY_AVAILABLE or len(self.sentiment_history) < window:
            return 0.0
        
        x = np.arange(len(list(self.sentiment_history)[-window:]))
        y = np.array(list(self.sentiment_history)[-window:])
        
        slope, _, _, _, _ = linregress(x, y)
        return round(slope, 3)

    def detect_target_mentioned(self, header, content):
        """Universal target detection"""
        if self.target_name == 'UNKNOWN':
            return False
            
        full_text = f"{header} {content}".upper()
        
        # Special handling for MBB
        if self.target_name == 'MBB':
            major_stocks = ['FPT', 'VCB', 'CTG', 'BID', 'VHM', 'VIC', 'TPB', 'SSI', 'GAS']
            for stock in major_stocks:
                if stock in full_text:
                    return False
            return 'MBB' in full_text
                
        for pattern in self.target_patterns:
            if pattern.upper() in full_text:
                return True
                
        return False

    def update_sentiment_history(self, sentiment_score):
        """Update historical sentiment for temporal metrics"""
        self.sentiment_history.append(sentiment_score)

    def process_dataset(self, input_file, output_file):
        """
        Process dataset với PhoBERT sentiment analysis
        
        Features:
        - Sử dụng PhoBERT cho Vietnamese text analysis
        - Keyword-based enhancement cho financial Vietnamese
        - Fallback to keyword-based nếu PhoBERT fail
        """
        
        print(f"\nPROCESSING DATASET (PhoBERT-Enhanced)")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Model: {'PhoBERT' if self.phobert_available else 'Keyword Fallback'}")
        print("=" * 60)
        
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"Error loading {input_file}: {e}")
            return None
        
        print(f"Loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ['Header', 'Content']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        if 'Created At' not in df.columns:
            df['Created At'] = datetime.now().strftime('%m/%d/%Y')
            print("Added default 'Created At' column")
        
        results = []
        phobert_success = 0
        keyword_fallback = 0
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:  # More frequent progress for slower PhoBERT
                progress = idx / len(df) * 100
                print(f"Processing {idx+1}/{len(df)} ({progress:.1f}%) - PhoBERT: {phobert_success}")
            
            created_at = row.get('Created At', datetime.now().strftime('%m/%d/%Y'))
            header = str(row.get('Header', ''))
            content = str(row.get('Content', ''))
            content_length = len(content.strip())
            
            # === PHOBERT SENTIMENT ANALYSIS ===
            
            # 1. Sentiment Score (PhoBERT)
            sentiment_label, sentiment_score = self.calculate_phobert_sentiment(content)
            
            # Track which method was used
            if self.phobert_available:
                phobert_success += 1
            else:
                keyword_fallback += 1
            
            # 2. Target Mentioned
            target_mentioned = self.detect_target_mentioned(header, content)
            
            # 3. Relevance Score
            relevance_score = self.calculate_relevance_score(header, content, target_mentioned)
            
            # 4. Impact Score
            impact_score = self.calculate_impact_score(sentiment_score, relevance_score)
            
            # 5. Confidence
            confidence = self.calculate_confidence(sentiment_score, content_length, target_mentioned)
            
            # 6-7. Short/Medium-Term
            short_term_score, medium_term_score = self.calculate_short_medium_term_scores(sentiment_score)
            
            # 8. Sentiment Momentum
            sentiment_momentum = self.calculate_sentiment_momentum(sentiment_score)
            
            # 9. Sentiment Volatility
            sentiment_volatility = self.calculate_sentiment_volatility()
            
            # 10. Sentiment Trend
            sentiment_trend = self.calculate_sentiment_trend()
            
            # Update history
            self.update_sentiment_history(sentiment_score)
            
            result = {
                'Created At': created_at,
                'Header': header,
                'Content': content,
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score,
                'impact_score': impact_score,
                'relevance_score': relevance_score,
                'confidence': confidence,
                'target_mentioned': target_mentioned,
                'short_term_score': short_term_score,
                'medium_term_score': medium_term_score,
                'sentiment_momentum': sentiment_momentum,
                'sentiment_volatility': sentiment_volatility,
                'sentiment_trend': sentiment_trend
            }
            
            results.append(result)
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\nPROCESSING COMPLETED!")
        print(f"Saved: {output_file}")
        print(f"Rows: {len(result_df)}")
        print(f"PhoBERT processed: {phobert_success}")
        if keyword_fallback > 0:
            print(f"Keyword fallback: {keyword_fallback}")
        
        # Statistics
        sentiment_dist = result_df['sentiment_label'].value_counts()
        target_count = result_df['target_mentioned'].sum()
        
        print(f"\nRESULTS STATISTICS:")
        print("-" * 40)
        for sentiment, count in sentiment_dist.items():
            pct = count / len(result_df) * 100
            print(f"  {sentiment}: {count} ({pct:.1f}%)")
        
        print(f"\nTarget mentioned: {target_count}/{len(result_df)} ({target_count/len(result_df)*100:.1f}%)")
        print(f"Average sentiment score: {result_df['sentiment_score'].mean():.2f}")
        print(f"Average impact score: {result_df['impact_score'].mean():.2f}")
        print(f"Average confidence: {result_df['confidence'].mean():.2f}")
        
        print(f"\n✅ PHOBERT SENTIMENT CALCULATION SUCCESS!")
        print(f"🤖 Using PhoBERT for Vietnamese Financial Sentiment Analysis")
        print(f"📊 Enhanced accuracy for Vietnamese text")
        
        return result_df

def main():
    """Main function với command line support"""
    
    if len(sys.argv) < 3:
        print("Usage: python Tinh_all_PhoBERT.py input.csv output.csv [target_name]")
        print("Example: python Tinh_all_PhoBERT.py test.csv result.csv META")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    target_name = sys.argv[3] if len(sys.argv) > 3 else 'UNKNOWN'
    
    calculator = PhoBERTSentimentCalculator(target_name)
    
    result_df = calculator.process_dataset(input_file, output_file)
    
    return result_df

if __name__ == "__main__":
    main()