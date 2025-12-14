#!/usr/bin/env python3
"""
MULTI-MODEL PREDICTION SYSTEM  
==============================
Test và so sánh 3 loại model:
1. Basic LSTM
2. Sentiment LSTM  
3. Hybrid Model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import model architectures
from train_multi_model_system import BasicLSTMModel, SentimentDLinearNodeModel, HybridDLinearNodeModel, MultiModelDataProcessor

class MultiModelPredictor:
    """Prediction system cho 3 loại model"""
    
    def __init__(self, model_dir='model'):
        self.model_dir = model_dir
        self.processor = MultiModelDataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
    def load_model(self, stock_name, model_type):
        """Load một model cụ thể"""
        model_path = f"{self.model_dir}/{stock_name}_{model_type}_model.pt"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model based on type with correct parameters to match trained models
            if model_type == 'basic':
                model = BasicLSTMModel(input_dim=5)  # Price features
            elif model_type == 'sentiment':
                # Match trained model parameters: seq_len=60, sentiment features=9
                model = SentimentDLinearNodeModel(seq_len=60, price_dim=5, sentiment_dim=9)
            elif model_type == 'hybrid':
                # Match trained model parameters: seq_len=60
                model = HybridDLinearNodeModel(seq_len=60, price_dim=5, residual_dim=5)
            else:
                print(f"Unknown model type: {model_type}")
                return None
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✓ Loaded {stock_name} {model_type} model (R²: {checkpoint['best_r2']:.4f})")
            return model
            
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            return None
    
    def load_all_models_for_stock(self, stock_name):
        """Load tất cả models cho 1 chứng khoán"""
        models = {}
        
        for model_type in ['basic', 'sentiment', 'hybrid']:
            model = self.load_model(stock_name, model_type)
            if model is not None:
                models[model_type] = model
        
        self.models[stock_name] = models
        return models
    
    def predict_with_model(self, stock_name, model_type, data_paths, test_ratio=0.2):
        """Predict với 1 model cụ thể"""
        if stock_name not in self.models or model_type not in self.models[stock_name]:
            print(f"Model {stock_name}_{model_type} not loaded")
            return None
        
        model = self.models[stock_name][model_type]
        
        # Load data
        df = self.processor.load_stock_data(stock_name, data_paths)
        if df is None:
            return None
        
        prices = df['Close'].values
        
        # Prepare data based on model type
        if model_type == 'basic':
            features = self.processor.create_basic_features(df)
            X, y = self.processor.prepare_basic_sequences(features, prices, stock_name)
        elif model_type == 'sentiment':
            price_features = self.processor.create_basic_features(df)
            sentiment_features = self.processor.create_sentiment_features(df)
            X_price, X_sentiment, y = self.processor.prepare_sentiment_sequences(
                price_features, sentiment_features, prices, stock_name
            )
        elif model_type == 'hybrid':
            # Hybrid DLinear+NODE model needs price + residual data
            price_features = self.processor.create_basic_features(df)
            
            # Create residual features (same as in training)
            residual_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Clean volume data - convert strings like '69.84M' to numeric
            residual_features['Volume'] = pd.to_numeric(residual_features['Volume'], errors='coerce')
            
            for col in ['Open', 'High', 'Low', 'Close']:
                residual_features[col] = residual_features[col] - residual_features[col].rolling(window=20, min_periods=1).mean()
            
            # Keep same column names as basic features for scaler compatibility
            residual_features.columns = ['open', 'high', 'low', 'close', 'volume']
            residual_features = residual_features.fillna(0)
            
            X_price, _ = self.processor.prepare_basic_sequences(price_features, prices, stock_name)
            # Use residual scaler key for consistency with training
            residual_scaler_key = f"{stock_name}_residual"
            X_residual, y = self.processor.prepare_basic_sequences(residual_features, prices, stock_name, scaler_key_override=residual_scaler_key)
            
            # Ensure same length
            min_len = min(len(X_price), len(X_residual))
            X_price = X_price[:min_len]
            X_residual = X_residual[:min_len]
            y = y[:min_len]
        
        # Split test data
        if model_type == 'sentiment':
            test_size = int(len(X_price) * test_ratio)
            X_price_test = X_price[-test_size:]
            X_sentiment_test = X_sentiment[-test_size:]
            y_test = y[-test_size:]
            
            # Make predictions
            with torch.no_grad():
                X_price_tensor = torch.FloatTensor(X_price_test).to(self.device)
                X_sentiment_tensor = torch.FloatTensor(X_sentiment_test).to(self.device)
                predictions_scaled = model(X_price_tensor, X_sentiment_tensor).cpu().numpy().flatten()
        
        elif model_type == 'hybrid':
            test_size = int(len(X_price) * test_ratio)
            X_price_test = X_price[-test_size:]
            X_residual_test = X_residual[-test_size:]
            y_test = y[-test_size:]
            
            # Make predictions
            with torch.no_grad():
                X_price_tensor = torch.FloatTensor(X_price_test).to(self.device)
                X_residual_tensor = torch.FloatTensor(X_residual_test).to(self.device)
                predictions_scaled = model(X_price_tensor, X_residual_tensor).cpu().numpy().flatten()
        
        else:
            # Basic LSTM model
            test_size = int(len(X) * test_ratio)
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            
            # Make predictions
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(self.device)
                predictions_scaled = model(X_tensor).cpu().numpy().flatten()
        
        # Convert back to original scale
        if model_type == 'basic':
            scaler_key = f"{stock_name}_basic"
            predictions_orig = self.processor.feature_scalers[scaler_key].inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            y_test_orig = self.processor.feature_scalers[scaler_key].inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()
        elif model_type == 'sentiment':
            scaler_key = f"{stock_name}_target"
            predictions_orig = self.processor.feature_scalers[scaler_key].inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            y_test_orig = self.processor.feature_scalers[scaler_key].inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()
        elif model_type == 'hybrid':
            # Use the same residual scaler key for target scaling
            residual_scaler_key = f"{stock_name}_residual"
            predictions_orig = self.processor.feature_scalers[residual_scaler_key].inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            y_test_orig = self.processor.feature_scalers[residual_scaler_key].inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()
        
        # Calculate metrics on SCALED data (normalized values ~0-1)
        r2 = r2_score(y_test, predictions_scaled)
        mse = mean_squared_error(y_test, predictions_scaled)
        mae = mean_absolute_error(y_test, predictions_scaled)
        mape = np.mean(np.abs((y_test - predictions_scaled) / (y_test + 1e-8))) * 100
        
        # Keep original scale metrics for reference
        mse_orig = mean_squared_error(y_test_orig, predictions_orig)
        mae_orig = mean_absolute_error(y_test_orig, predictions_orig)
        mape_orig = np.mean(np.abs((y_test_orig - predictions_orig) / y_test_orig)) * 100
        
        # Debug: Check value ranges for alignment issues
        print(f"  [DEBUG] Actual range: ${np.min(y_test_orig):.2f} - ${np.max(y_test_orig):.2f}")
        print(f"  [DEBUG] Predicted range: ${np.min(predictions_orig):.2f} - ${np.max(predictions_orig):.2f}")
        correlation = np.corrcoef(y_test_orig, predictions_orig)[0,1]
        print(f"  [DEBUG] Correlation: {correlation:.4f}")
        
        results = {
            'stock_name': stock_name,
            'model_type': model_type,
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'mse_orig': mse_orig,
            'mae_orig': mae_orig,
            'mape_orig': mape_orig,
            'test_samples': len(y_test_orig),
            'predictions': predictions_orig,
            'actual': y_test_orig
        }
        
        return results
    
    def compare_all_models_for_stock(self, stock_name, data_paths):
        """So sánh tất cả models cho 1 chứng khoán"""
        print(f"\n{'='*60}")
        print(f"COMPARING ALL MODELS FOR {stock_name}")
        print(f"{'='*60}")
        
        # Load models
        models = self.load_all_models_for_stock(stock_name)
        if not models:
            print(f"No models found for {stock_name}")
            return None
        
        results = {}
        
        # Test each model
        for model_type in ['basic', 'sentiment', 'hybrid']:
            if model_type in models:
                print(f"\nTesting {model_type} model...")
                result = self.predict_with_model(stock_name, model_type, data_paths)
                if result:
                    results[model_type] = result
                    print(f"  R²: {result['r2_score']:.4f} ({result['r2_score']:.2%})")
                    print(f"  MAE: ${result['mae']:.2f}")
                    print(f"  MAPE: {result['mape']:.2f}%")
        
        return results
    
    def predict_future_prices(self, stock_name, model_type, data_paths, days=5):
        """Predict future prices"""
        if stock_name not in self.models or model_type not in self.models[stock_name]:
            print(f"Model {stock_name}_{model_type} not loaded")
            return None
        
        model = self.models[stock_name][model_type]
        
        # Load data
        df = self.processor.load_stock_data(stock_name, data_paths)
        if df is None:
            return None
        
        prices = df['Close'].values
        
        # Prepare latest data based on model type
        if model_type == 'basic':
            features = self.processor.create_basic_features(df)
            X, _ = self.processor.prepare_basic_sequences(features, prices, stock_name)
            last_sequence = X[-1:]  # Last sequence
        elif model_type == 'sentiment':
            price_features = self.processor.create_basic_features(df)
            sentiment_features = self.processor.create_sentiment_features(df)
            X_price, X_sentiment, _ = self.processor.prepare_sentiment_sequences(
                price_features, sentiment_features, prices, stock_name
            )
            last_price_seq = X_price[-1:]
            last_sentiment_seq = X_sentiment[-1:]
        elif model_type == 'hybrid':
            # Hybrid DLinear+NODE model needs price + residual data
            price_features = self.processor.create_basic_features(df)
            
            # Create residual features (same as in training)
            residual_features = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Clean volume data - convert strings like '69.84M' to numeric
            residual_features['Volume'] = pd.to_numeric(residual_features['Volume'], errors='coerce')
            
            for col in ['Open', 'High', 'Low', 'Close']:
                residual_features[col] = residual_features[col] - residual_features[col].rolling(window=20, min_periods=1).mean()
            
            # Keep same column names as basic features for scaler compatibility
            residual_features.columns = ['open', 'high', 'low', 'close', 'volume']
            residual_features = residual_features.fillna(0)
            
            X_price, _ = self.processor.prepare_basic_sequences(price_features, prices, stock_name)
            # Use residual scaler key for consistency with training
            residual_scaler_key = f"{stock_name}_residual"
            X_residual, _ = self.processor.prepare_basic_sequences(residual_features, prices, stock_name, scaler_key_override=residual_scaler_key)
            
            # Ensure same length
            min_len = min(len(X_price), len(X_residual))
            last_price_seq = X_price[-1:]
            last_residual_seq = X_residual[-1:]
        
        # Predict future prices
        predictions = []
        
        with torch.no_grad():
            for day in range(days):
                if model_type == 'sentiment':
                    X_price_tensor = torch.FloatTensor(last_price_seq).to(self.device)
                    X_sentiment_tensor = torch.FloatTensor(last_sentiment_seq).to(self.device)
                    pred_scaled = model(X_price_tensor, X_sentiment_tensor).cpu().item()
                elif model_type == 'hybrid':
                    X_price_tensor = torch.FloatTensor(last_price_seq).to(self.device)
                    X_residual_tensor = torch.FloatTensor(last_residual_seq).to(self.device)
                    pred_scaled = model(X_price_tensor, X_residual_tensor).cpu().item()
                else:
                    X_tensor = torch.FloatTensor(last_sequence).to(self.device)
                    pred_scaled = model(X_tensor).cpu().item()
                
                # Convert to original scale
                if model_type == 'basic':
                    scaler_key = f"{stock_name}_basic"
                    pred_price = self.processor.feature_scalers[scaler_key].inverse_transform([[pred_scaled]])[0][0]
                elif model_type == 'sentiment':
                    scaler_key = f"{stock_name}_target"
                    pred_price = self.processor.feature_scalers[scaler_key].inverse_transform([[pred_scaled]])[0][0]
                elif model_type == 'hybrid':
                    scaler_key = f"{stock_name}_hybrid_target"
                    pred_price = self.processor.price_scalers[scaler_key].inverse_transform([[pred_scaled]])[0][0]
                
                predictions.append(pred_price)
                
                # Update sequence for next prediction (simplified)
                # In practice, you'd update with proper feature engineering
        
        # Create future dates
        last_date = df['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        future_predictions = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'predicted_price': pred,
                'day': i+1
            }
            for i, (date, pred) in enumerate(zip(future_dates, predictions))
        ]
        
        return {
            'stock_name': stock_name,
            'model_type': model_type,
            'last_actual_price': prices[-1],
            'last_date': last_date,
            'future_predictions': future_predictions
        }

def create_sentiment_superiority_analysis(all_results):
    """Tạo phân tích chứng minh ưu điểm của mô hình cảm xúc"""
    
    print("\n" + "="*100)
    print("SENTIMENT MODEL SUPERIORITY ANALYSIS")
    print("="*100)
    
    # Calculate improvement metrics
    improvements = {}
    
    for stock_name, stock_results in all_results.items():
        if 'sentiment' not in stock_results:
            continue
            
        sentiment_r2 = stock_results['sentiment']['r2_score']
        sentiment_mae = stock_results['sentiment']['mae']
        
        stock_improvements = {
            'sentiment_performance': {
                'r2_score': sentiment_r2,
                'mae': sentiment_mae,
                'mape': stock_results['sentiment']['mape']
            }
        }
        
        # Compare with other models
        for model_type in ['basic', 'hybrid']:
            if model_type in stock_results:
                other_r2 = stock_results[model_type]['r2_score']
                other_mae = stock_results[model_type]['mae']
                
                r2_improvement = ((sentiment_r2 - other_r2) / other_r2) * 100
                mae_improvement = ((other_mae - sentiment_mae) / other_mae) * 100
                
                stock_improvements[f'vs_{model_type}'] = {
                    'r2_improvement_percent': r2_improvement,
                    'mae_improvement_percent': mae_improvement,
                    'r2_absolute_diff': sentiment_r2 - other_r2
                }
        
        improvements[stock_name] = stock_improvements
    
    # Print improvement summary
    print(f"\n{'STOCK':<8} {'VS MODEL':<12} {'R² IMPROVE':<12} {'MAE IMPROVE':<12} {'R² DIFF':<10}")
    print("-" * 70)
    
    total_r2_improvements = []
    total_mae_improvements = []
    
    for stock_name, stock_data in improvements.items():
        for comparison in ['vs_basic', 'vs_hybrid']:
            if comparison in stock_data:
                comp_data = stock_data[comparison]
                r2_imp = comp_data['r2_improvement_percent']
                mae_imp = comp_data['mae_improvement_percent']
                r2_diff = comp_data['r2_absolute_diff']
                
                print(f"{stock_name:<8} {comparison.upper():<12} {r2_imp:>10.2f}% {mae_imp:>10.2f}% {r2_diff:>8.4f}")
                
                total_r2_improvements.append(r2_imp)
                total_mae_improvements.append(mae_imp)
    
    # Overall statistics
    if total_r2_improvements:
        avg_r2_improvement = np.mean(total_r2_improvements)
        avg_mae_improvement = np.mean(total_mae_improvements)
        
        print(f"\nOVERALL SENTIMENT MODEL ADVANTAGES:")
        print(f"Average R² Improvement: {avg_r2_improvement:+.2f}%")
        print(f"Average MAE Improvement: {avg_mae_improvement:+.2f}%")
        
        wins = sum(1 for imp in total_r2_improvements if imp > 0)
        total_comparisons = len(total_r2_improvements)
        win_rate = (wins / total_comparisons) * 100
        
        print(f"Win Rate: {wins}/{total_comparisons} ({win_rate:.1f}%)")
    
    return improvements

def create_individual_stock_tables(all_results):
    """Tạo bảng chi tiết cho từng chứng khoán"""
    
    for stock_name, stock_results in all_results.items():
        print(f"\n" + "="*80)
        print(f"DETAILED PERFORMANCE TABLE FOR {stock_name}")
        print("="*80)
        
        # Create comprehensive metrics table
        print(f"\n{'METRIC':<20} {'BASIC LSTM':<15} {'SENTIMENT':<15} {'HYBRID':<15} {'BEST':<10}")
        print("-" * 80)
        
        metrics = ['r2_score', 'mae', 'mse', 'mape']
        metric_names = ['R² Score', 'MAE', 'MSE', 'MAPE (%)']
        
        best_per_metric = {}
        
        for metric, display_name in zip(metrics, metric_names):
            row_values = {}
            
            # Collect values for each model
            for model_type in ['basic', 'sentiment', 'hybrid']:
                if model_type in stock_results and metric in stock_results[model_type]:
                    value = stock_results[model_type][metric]
                    row_values[model_type] = value
            
            if not row_values:
                continue
                
            # Find best value (higher for R², lower for others)
            if metric == 'r2_score':
                best_model = max(row_values.keys(), key=lambda x: row_values[x])
                best_value = max(row_values.values())
            else:
                best_model = min(row_values.keys(), key=lambda x: row_values[x])
                best_value = min(row_values.values())
            
            best_per_metric[metric] = best_model
            
            # Format values
            basic_val = f"{row_values.get('basic', 0):.4f}" if 'basic' in row_values else "N/A"
            sentiment_val = f"{row_values.get('sentiment', 0):.4f}" if 'sentiment' in row_values else "N/A"
            hybrid_val = f"{row_values.get('hybrid', 0):.4f}" if 'hybrid' in row_values else "N/A"
            
            # Add stars for best values
            if 'basic' in row_values and best_model == 'basic':
                basic_val += " ★"
            if 'sentiment' in row_values and best_model == 'sentiment':
                sentiment_val += " ★"
            if 'hybrid' in row_values and best_model == 'hybrid':
                hybrid_val += " ★"
            
            print(f"{display_name:<20} {basic_val:<15} {sentiment_val:<15} {hybrid_val:<15} {best_model.upper():<10}")
        
        # Model ranking for this stock
        print(f"\nMODEL RANKING FOR {stock_name}:")
        print("-" * 40)
        
        # Sort by R² score
        model_r2_scores = [(model, result['r2_score']) for model, result in stock_results.items()]
        model_r2_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model, r2) in enumerate(model_r2_scores, 1):
            if model == 'sentiment':
                print(f"{rank}. 🏆 {model.upper()}: R² = {r2:.4f} ({r2:.2%}) - WITH SENTIMENT DATA")
            else:
                print(f"{rank}. {model.upper()}: R² = {r2:.4f} ({r2:.2%})")
        
        # Sentiment model advantages for this stock
        if 'sentiment' in stock_results:
            print(f"\nSENTIMENT MODEL ADVANTAGES FOR {stock_name}:")
            print("-" * 50)
            
            sentiment_r2 = stock_results['sentiment']['r2_score']
            
            for other_model in ['basic', 'hybrid']:
                if other_model in stock_results:
                    other_r2 = stock_results[other_model]['r2_score']
                    improvement = ((sentiment_r2 - other_r2) / other_r2) * 100
                    
                    if improvement > 0:
                        print(f"✓ {improvement:+.2f}% better R² than {other_model.upper()}")
                    else:
                        print(f"✗ {improvement:+.2f}% vs {other_model.upper()}")

def create_comparison_visualization(all_results):
    """Tạo visualization so sánh các models với focus trên sentiment model"""
    
    # Prepare data for plotting
    stocks = list(all_results.keys())
    model_types = ['basic', 'sentiment', 'hybrid']
    
    # Create comprehensive comparison figure với spacing tốt hơn
    fig, axes = plt.subplots(3, len(stocks), figsize=(10*len(stocks), 20))
    if len(stocks) == 1:
        axes = axes.reshape(3, 1)
    
    # Adjust spacing giữa các subplot
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.08)
    
    for i, stock_name in enumerate(stocks):
        stock_results = all_results[stock_name]
        
        # Plot 1: R² comparison with emphasis on sentiment
        ax1 = axes[0, i] if len(stocks) > 1 else axes[0]
        
        r2_scores = []
        model_names = []
        colors = []
        
        for j, model_type in enumerate(model_types):
            if model_type in stock_results:
                r2_scores.append(stock_results[model_type]['r2_score'])
                model_names.append(model_type.capitalize())
                # Highlight sentiment model
                if model_type == 'sentiment':
                    colors.append('gold')  # Golden color for sentiment
                elif model_type == 'basic':
                    colors.append('lightblue')
                else:
                    colors.append('lightcoral')
        
        bars = ax1.bar(model_names, r2_scores, color=colors)
        ax1.set_title(f'{stock_name} - Model R² Comparison\n⭐ Sentiment Model Performance', 
                     fontweight='bold', fontsize=12, pad=20)
        ax1.set_ylabel('R² Score', fontsize=10)
        ax1.set_ylim(0, 1.05)
        
        # Add value labels and highlight best
        best_idx = r2_scores.index(max(r2_scores))
        for j, (bar, score) in enumerate(zip(bars, r2_scores)):
            height = bar.get_height()
            label = f'{score:.3f}'
            if j == best_idx:
                label += ' 🏆'
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, label, 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Rotate x-axis labels để tránh overlap
        ax1.tick_params(axis='x', labelsize=9)
        ax1.tick_params(axis='y', labelsize=9)
        
        # Plot 2: Actual vs Predicted SCATTER PLOT for better alignment visualization
        ax2 = axes[1, i] if len(stocks) > 1 else axes[1]
        
        # Create scatter plot to show actual alignment between actual and predicted
        colors = {'basic': 'lightcoral', 'sentiment': 'gold', 'hybrid': 'lightgreen'}
        markers = {'basic': 'o', 'sentiment': 's', 'hybrid': '^'}
        
        # Collect all actual values for perfect prediction line
        all_actual_values = []
        
        for model_type in ['basic', 'sentiment', 'hybrid']:
            if model_type in stock_results:
                result = stock_results[model_type]
                
                # Get recent data for visualization (last 50 points for clarity)
                actual = result['actual'][-50:]
                predicted = result['predictions'][-50:]
                
                # Ensure same length
                min_len = min(len(actual), len(predicted))
                actual = actual[-min_len:]
                predicted = predicted[-min_len:]
                
                # Store for perfect line
                if len(all_actual_values) == 0:
                    all_actual_values = actual
                
                # Create scatter plot
                model_label = f'{model_type.capitalize()}'
                if model_type == 'sentiment':
                    model_label += ' (DL+NODE) - R²={:.3f}'.format(result['r2_score'])
                elif model_type == 'hybrid':
                    model_label += ' (DL+NODE) - R²={:.3f}'.format(result['r2_score'])
                else:
                    model_label += ' LSTM - R²={:.3f}'.format(result['r2_score'])
                    
                ax2.scatter(actual, predicted, 
                           color=colors[model_type], alpha=0.7, s=30, 
                           marker=markers[model_type], label=model_label, edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line (y=x)
        if len(all_actual_values) > 0:
            min_val = min(all_actual_values)
            max_val = max(all_actual_values) 
            ax2.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, alpha=0.8, label='Perfect Prediction (y=x)')
        
        ax2.legend(fontsize=8, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Actual Price ($)', fontsize=10)
        ax2.set_ylabel('Predicted Price ($)', fontsize=10)
        ax2.tick_params(axis='both', labelsize=9)
        
        # Set title for clarity
        ax2.set_title(f'{stock_name} - Actual vs Predicted Alignment\n(Points closer to red line = better predictions)', 
                     fontsize=10, fontweight='bold')
        
        # Plot 3: Performance Comparison Table as Visual
        ax3 = axes[2, i] if len(stocks) > 1 else axes[2]
        
        # Create table-style visualization like your examples
        metrics_data = []
        model_names = []
        
        for model_type in ['basic', 'sentiment', 'hybrid']:
            if model_type in stock_results:
                result = stock_results[model_type]
                metrics_data.append([
                    result['mse'],
                    np.sqrt(result['mse']),  # RMSE
                    result['mae'],
                    result['mape'],
                    result['r2_score']
                ])
                if model_type == 'basic':
                    model_names.append('Basic LSTM')
                elif model_type == 'sentiment':
                    model_names.append('Sentiment DL+NODE') 
                else:
                    model_names.append('Hybrid DL+NODE')
        
        if metrics_data:
            # Create table
            metric_labels = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
            
            # Convert to numpy array for easier handling
            data_array = np.array(metrics_data)
            
            # Create table
            table_data = []
            for j, metric in enumerate(metric_labels):
                row = [metric]
                for k in range(len(model_names)):
                    if j == 4:  # R² score
                        row.append(f'{data_array[k, j]:.2f}')
                    elif j in [0, 1, 2]:  # MSE, RMSE, MAE
                        row.append(f'{data_array[k, j]:.2f}')
                    else:  # MAPE
                        row.append(f'{data_array[k, j]:.2f}')
                table_data.append(row)
            
            # Add actual and predicted values with real data
            actual_values = []
            predicted_values = []
            
            for model_type in ['basic', 'sentiment', 'hybrid']:
                if model_type in stock_results:
                    result = stock_results[model_type]
                    # Get last actual price
                    last_actual = result['actual'][-1]
                    # Get predicted price for next day
                    last_predicted = result['predictions'][-1]
                    
                    actual_values.append(f'{last_actual:.2f}')
                    predicted_values.append(f'{last_predicted:.2f}')
                else:
                    actual_values.append('N/A')
                    predicted_values.append('N/A')
            
            table_data.append(['Actual value'] + actual_values)
            table_data.append(['Next day pred'] + predicted_values)
            
            # Hide axes and create table
            ax3.axis('off')
            
            table = ax3.table(cellText=table_data,
                            colLabels=['Metric'] + model_names,
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.8)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4472C4')
                elif j == 0:  # First column (metric names)
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#D9E2F3')
                else:
                    cell.set_facecolor('#F2F2F2')
                
                cell.set_edgecolor('black')
                cell.set_linewidth(1)
            
            
        else:
            ax3.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax3.transAxes)
    
    # Save với resolution cao và proper spacing
    plt.savefig('Anh/multi_model_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.show()
    
    print("Comparison visualization saved as Anh/multi_model_comparison.png")

def create_performance_tables_like_examples(all_results):
    """Tạo bảng hiệu suất theo định dạng mẫu bạn cung cấp"""
    
    print("\n" + "="*120)
    print("PERFORMANCE TABLES - FORMAT THEO YÊU CẦU")
    print("="*120)
    
    stock_names_mapping = {
        'META': 'Meta',
        'FPT': 'FPT',
        'MBB': 'MBBank' 
    }
    
    # Table for Basic LSTM
    print(f"\nBảng 6. Số liệu hiệu suất mô hình Basic LSTM trên 3 tập dữ liệu chứng khoán")
    print("Training Period: 10/07/2013 - 18/11/2024")
    print("="*90)
    
    headers = [""] + [stock_names_mapping.get(stock, stock) for stock in all_results.keys()]
    print(f"{'':^12} " + " ".join([f"{h:^12}" for h in headers[1:]]))
    print("-" * (12 + len(headers[1:]) * 13))
    
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
    
    for metric in metrics:
        row = [f"{metric:^12}"]
        for stock_name in all_results.keys():
            if 'basic' in all_results[stock_name]:
                result = all_results[stock_name]['basic']
                if metric == 'MSE':
                    row.append(f"{result['mse']:^12.2f}")
                elif metric == 'RMSE':
                    row.append(f"{np.sqrt(result['mse']):^12.2f}")
                elif metric == 'MAE':
                    row.append(f"{result['mae']:^12.2f}")
                elif metric == 'MAPE':
                    row.append(f"{result['mape']:^12.2f}")
                elif metric == 'R²':
                    row.append(f"{result['r2_score']:^12.2f}")
            else:
                row.append(f"{'N/A':^12}")
        print(" ".join(row))
    
    # Add actual and predicted values with real data for basic model
    actual_values = []
    predicted_values = []
    
    for stock_name in all_results.keys():
        if 'basic' in all_results[stock_name]:
            result = all_results[stock_name]['basic']
            # Get actual price (last actual value)
            last_actual = result['actual'][-1]
            actual_values.append(f'{last_actual:.2f}')
            
            # Get predicted price for next day
            last_predicted = result['predictions'][-1]
            predicted_values.append(f'{last_predicted:.2f}')
        else:
            actual_values.append('N/A')
            predicted_values.append('N/A')
    
    print(" ".join([f"{'Actual value':^12}"] + [f"{val:^12}" for val in actual_values]))
    print(" ".join([f"{'Next day pred':^12}"] + [f"{val:^12}" for val in predicted_values]))
    print(f"{'':^60} 18/11/2024")
    
    # Table for Sentiment DLinear+NODE
    print(f"\n\nBảng 7. Số liệu đánh giá cho mô hình Sentiment DLinear+NODE trên 3 tập dữ liệu chứng khoán")
    print("Training Period: 10/07/2013 - 18/11/2024")
    print("="*90)
    
    print(f"{'':^12} " + " ".join([f"{h:^12}" for h in headers[1:]]))
    print("-" * (12 + len(headers[1:]) * 13))
    
    for metric in metrics:
        row = [f"{metric:^12}"]
        for stock_name in all_results.keys():
            if 'sentiment' in all_results[stock_name]:
                result = all_results[stock_name]['sentiment']
                if metric == 'MSE':
                    row.append(f"{result['mse']:^12.2f}")
                elif metric == 'RMSE':
                    row.append(f"{np.sqrt(result['mse']):^12.2f}")
                elif metric == 'MAE':
                    row.append(f"{result['mae']:^12.2f}")
                elif metric == 'MAPE':
                    row.append(f"{result['mape']:^12.2f}")
                elif metric == 'R²':
                    row.append(f"{result['r2_score']:^12.2f}")
            else:
                row.append(f"{'N/A':^12}")
        print(" ".join(row))
    
    # Add actual and predicted values with real data for sentiment model
    actual_values = []
    predicted_values = []
    
    for stock_name in all_results.keys():
        if 'sentiment' in all_results[stock_name]:
            result = all_results[stock_name]['sentiment']
            # Get actual price (last actual value)
            last_actual = result['actual'][-1]
            actual_values.append(f'{last_actual:.2f}')
            
            # Get predicted price for next day
            last_predicted = result['predictions'][-1]
            predicted_values.append(f'{last_predicted:.2f}')
        else:
            actual_values.append('N/A')
            predicted_values.append('N/A')
    
    print(" ".join([f"{'Actual value':^12}"] + [f"{val:^12}" for val in actual_values]))
    print(" ".join([f"{'Next day pred':^12}"] + [f"{val:^12}" for val in predicted_values]))
    print(f"{'':^60} 18/11/2024")

def create_research_tables(all_results):
    """Tạo bảng nghiên cứu định dạng LaTeX/academic"""
    
    print("\n" + "="*100)
    print("RESEARCH TABLES - SENTIMENT MODEL SUPERIORITY EVIDENCE")
    print("="*100)
    
    # Table 1: Overall Performance Comparison
    print("\nTABLE 1: Overall Performance Comparison Across All Models")
    print("=" * 80)
    print("| Stock    | Model Type        | R² Score | MAE    | MSE      | MAPE   |")
    print("|----------|------------------|----------|--------|----------|--------|")
    
    for stock_name, stock_results in all_results.items():
        for model_type in ['basic', 'sentiment', 'hybrid']:
            if model_type in stock_results:
                result = stock_results[model_type]
                r2 = result['r2_score']
                mae = result['mae']
                mse = result['mse']
                mape = result['mape']
                
                # Add star for best R² score
                best_r2 = max([stock_results[m]['r2_score'] for m in stock_results.keys()])
                star = " *" if r2 == best_r2 else ""
                
                print(f"| {stock_name:<8} | {model_type.capitalize():<15}{star} | {r2:.4f}   | {mae:.2f}   | {mse:.2f}     | {mape:.2f}% |")
    
    # Table 2: Sentiment Model Improvement Statistics  
    print("\n\nTABLE 2: Sentiment Model Performance Improvements")
    print("=" * 70)
    print("| Stock | vs Basic LSTM | vs Hybrid DLinear | Ranking |")
    print("|-------|---------------|-------------------|---------|")
    
    sentiment_wins = 0
    total_comparisons = 0
    
    for stock_name, stock_results in all_results.items():
        if 'sentiment' not in stock_results:
            continue
            
        sentiment_r2 = stock_results['sentiment']['r2_score']
        improvements = []
        
        # vs Basic
        if 'basic' in stock_results:
            basic_r2 = stock_results['basic']['r2_score']
            basic_improvement = ((sentiment_r2 - basic_r2) / basic_r2) * 100
            improvements.append(f"{basic_improvement:+.1f}%")
            if basic_improvement > 0:
                sentiment_wins += 1
            total_comparisons += 1
        else:
            improvements.append("N/A")
        
        # vs Hybrid
        if 'hybrid' in stock_results:
            hybrid_r2 = stock_results['hybrid']['r2_score']
            hybrid_improvement = ((sentiment_r2 - hybrid_r2) / hybrid_r2) * 100
            improvements.append(f"{hybrid_improvement:+.1f}%")
            if hybrid_improvement > 0:
                sentiment_wins += 1
            total_comparisons += 1
        else:
            improvements.append("N/A")
        
        # Ranking
        stock_models = sorted(stock_results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        rank = next(i for i, (model, _) in enumerate(stock_models, 1) if model == 'sentiment')
        
        print(f"| {stock_name:<5} | {improvements[0]:<13} | {improvements[1]:<17} | #{rank:<6} |")
    
    # Statistical Summary
    win_rate = (sentiment_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    print(f"\nSENTIMENT MODEL WIN RATE: {sentiment_wins}/{total_comparisons} ({win_rate:.1f}%)")
    
    # Table 3: Detailed Metrics by Stock (for each stock separately)
    for stock_name, stock_results in all_results.items():
        print(f"\n\nTABLE 3.{list(all_results.keys()).index(stock_name)+1}: Detailed Metrics for {stock_name}")
        print("=" * 60)
        print("| Metric          | Basic LSTM | Sentiment DL+NODE | Hybrid DL+NODE |")
        print("|-----------------|------------|-------------------|----------------|")
        
        metrics = [
            ('R² Score', 'r2_score', '.4f'),
            ('MAE', 'mae', '.2f'),
            ('MSE', 'mse', '.2f'),
            ('MAPE (%)', 'mape', '.2f'),
            ('Test Samples', 'test_samples', 'd')
        ]
        
        for metric_name, metric_key, format_str in metrics:
            values = []
            best_value = None
            best_is_higher = metric_key == 'r2_score'
            
            for model_type in ['basic', 'sentiment', 'hybrid']:
                if model_type in stock_results and metric_key in stock_results[model_type]:
                    value = stock_results[model_type][metric_key]
                    values.append((model_type, value))
                    
                    if best_value is None:
                        best_value = value
                    elif best_is_higher and value > best_value:
                        best_value = value
                    elif not best_is_higher and value < best_value:
                        best_value = value
                else:
                    values.append((model_type, None))
            
            # Format row
            row_data = []
            for model_type, value in values:
                if value is None:
                    row_data.append("N/A")
                else:
                    formatted = f"{value:{format_str}}"
                    if value == best_value:
                        formatted += " **"  # Mark best value
                    row_data.append(formatted)
            
            print(f"| {metric_name:<15} | {row_data[0]:<10} | {row_data[1]:<17} | {row_data[2]:<14} |")

def create_detailed_report(all_results):
    """Tạo detailed report"""
    
    print("\n" + "="*100)
    print("DETAILED MULTI-MODEL COMPARISON REPORT")
    print("="*100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary table
    print(f"\n{'STOCK':<10} {'MODEL':<12} {'R² SCORE':<10} {'MAE':<8} {'MAPE':<8} {'SAMPLES':<8}")
    print("-" * 70)
    
    best_models = {}
    
    for stock_name, stock_results in all_results.items():
        best_r2 = -1
        best_model_type = None
        
        for model_type, result in stock_results.items():
            r2 = result['r2_score']
            mae = result['mae']
            mape = result['mape']
            samples = result['test_samples']
            
            print(f"{stock_name:<10} {model_type:<12} {r2:<10.4f} ${mae:<7.2f} {mape:<7.2f}% {samples:<8d}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_type = model_type
        
        best_models[stock_name] = {
            'model_type': best_model_type,
            'r2_score': best_r2
        }
        print()
    
    # Best models summary
    print("\nBEST MODEL FOR EACH STOCK:")
    print("-" * 40)
    for stock_name, best_info in best_models.items():
        print(f"{stock_name}: {best_info['model_type'].upper()} (R² = {best_info['r2_score']:.4f})")
    
    # Model type ranking
    print("\nMODEL TYPE PERFORMANCE RANKING:")
    print("-" * 40)
    
    model_avg_scores = {}
    for model_type in ['basic', 'sentiment', 'hybrid']:
        scores = []
        for stock_results in all_results.values():
            if model_type in stock_results:
                scores.append(stock_results[model_type]['r2_score'])
        
        if scores:
            avg_score = np.mean(scores)
            model_avg_scores[model_type] = avg_score
    
    # Sort by average score
    sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_type, avg_score) in enumerate(sorted_models, 1):
        print(f"{i}. {model_type.upper()}: Average R² = {avg_score:.4f}")

def main():
    """Main prediction and comparison function"""
    print("MULTI-MODEL PREDICTION & COMPARISON SYSTEM")
    print("=" * 60)
    
    # Define stocks and data paths
    stocks_config = {
        'META': ['CamXuc/META.csv', 'Train/META.csv'],
        'FPT': ['CamXuc/FPT.csv', 'Train/FPT.csv'],
        # 'APPLE': ['Train/Apple.csv'],  # Temporarily disabled - no sentiment data
        'MBB': ['CamXuc/MBB.csv', 'Train/MBB.csv']
    }
    
    predictor = MultiModelPredictor()
    all_results = {}
    
    # Compare models for each stock
    for stock_name, data_paths in stocks_config.items():
        # Check if data files exist
        existing_paths = [path for path in data_paths if os.path.exists(path)]
        if not existing_paths:
            print(f"No data files found for {stock_name}, skipping...")
            continue
        
        results = predictor.compare_all_models_for_stock(stock_name, existing_paths)
        if results:
            all_results[stock_name] = results
    
    if not all_results:
        print("No results to display. Make sure models are trained first.")
        return
    
    # Create enhanced reports and visualizations focusing on sentiment model
    create_detailed_report(all_results)
    create_performance_tables_like_examples(all_results)
    create_research_tables(all_results)
    create_sentiment_superiority_analysis(all_results)
    create_individual_stock_tables(all_results)
    create_comparison_visualization(all_results)
    
    # Save results to JSON
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'comparison_results': {
            stock: {
                model: {
                    'r2_score': result['r2_score'],
                    'mae': result['mae'],
                    'mape': result['mape'],
                    'test_samples': result['test_samples']
                }
                for model, result in stock_results.items()
            }
            for stock, stock_results in all_results.items()
        }
    }
    
    with open('model/multi_model_comparison.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n✅ Comparison completed!")
    print(f"📊 Results saved to model/multi_model_comparison.json")
    print(f"📈 Visualization saved to Anh/multi_model_comparison.png")
    
    return all_results

if __name__ == "__main__":
    results = main()