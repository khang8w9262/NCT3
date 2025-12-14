#!/usr/bin/env python3
"""
ADVANCED YAHOO FINANCE FETCHER
==============================
Fetch stock data from Yahoo Finance and save to CSV
Supports multiple stocks, date ranges, and data validation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedYahooFetcher:
    """Advanced Yahoo Finance data fetcher"""
    
    def __init__(self, output_dir='../../DATASET/PRICE'):
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # Stock symbol suggestions database
        self.stock_suggestions = {
            # US Tech Giants
            'apple': ['AAPL'],
            'microsoft': ['MSFT'], 
            'google': ['GOOGL', 'GOOG'],
            'alphabet': ['GOOGL', 'GOOG'],
            'amazon': ['AMZN'],
            'meta': ['META'],
            'facebook': ['META'],
            'tesla': ['TSLA'],
            'nvidia': ['NVDA'],
            'netflix': ['NFLX'],
            
            # Popular US Stocks
            'nike': ['NKE'],
            'coca cola': ['KO'],
            'pepsi': ['PEP'],
            'walmart': ['WMT'],
            'disney': ['DIS'],
            'boeing': ['BA'],
            'ford': ['F'],
            'general motors': ['GM'],
            'jpmorgan': ['JPM'],
            'goldman sachs': ['GS'],
            'visa': ['V'],
            'mastercard': ['MA'],
            'paypal': ['PYPL'],
            'salesforce': ['CRM'],
            'oracle': ['ORCL'],
            'intel': ['INTC'],
            'amd': ['AMD'],
            'qualcomm': ['QCOM'],
            'adobe': ['ADBE'],
            'uber': ['UBER'],
            'airbnb': ['ABNB'],
            'zoom': ['ZM'],
            'slack': ['WORK'],
            'spotify': ['SPOT'],
            'twitter': ['TWTR'],
            'snapchat': ['SNAP'],
            
            # Vietnamese Stocks
            'fpt': ['FPT.VN'],
            'vietcombank': ['VCB.VN'],
            'vinamilk': ['VNM.VN'],
            'sabeco': ['SAB.VN'],
            'vingroup': ['VIC.VN'],
            'vinhomes': ['VHM.VN'],
            'techcombank': ['TCB.VN'],
            'bidv': ['BID.VN'],
            'military bank': ['MBB.VN'],
            'mb bank': ['MBB.VN'],
            'agribank': ['AGR.VN'],
            'petrolimex': ['PLX.VN'],
            'vincom retail': ['VRE.VN'],
            'hoa phat': ['HPG.VN'],
            'masan': ['MSN.VN'],
        }
    
    def ensure_output_dir(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def fetch_stock_data(self, symbol, period='5y', interval='1d', start_date="2013-07-10", end_date="2024-11-16"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'META', 'AAPL')
            period: Period to download (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
            interval: Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)
            start_date: Start date (YYYY-MM-DD), default: 2013-07-10
            end_date: End date (YYYY-MM-DD), default: 2024-11-16
        """
        
        print(f"Fetching {symbol} data...")
        
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Download data
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
            else:
                data = ticker.history(period=period, interval=interval, auto_adjust=False)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return None
            
            # Clean and process data
            data = self.clean_data(data, symbol)
            
            print(f"Fetched {len(data)} records for {symbol}")
            print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def clean_data(self, data, symbol):
        """Clean and validate stock data"""
        
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill missing values
        data = data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Ensure positive prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Validate OHLC relationships
        if all(col in data.columns for col in price_cols):
            # High should be >= Open, Close
            valid_high = (data['High'] >= data[['Open', 'Close']].max(axis=1))
            # Low should be <= Open, Close
            valid_low = (data['Low'] <= data[['Open', 'Close']].min(axis=1))
            
            data = data[valid_high & valid_low]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def save_to_csv(self, data, symbol, format='standard'):
        """
        Save data to CSV file
        
        Args:
            data: DataFrame with stock data
            symbol: Stock symbol
            format: 'standard' (English format) or 'train' (Vietnamese format)
        """
        
        if data is None or data.empty:
            print(f"No data to save for {symbol}")
            return None
        
        # Debug: Print data info
        print(f"Debug {symbol} data:")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {data.columns.tolist()}")
        print(f"   Index type: {type(data.index)}")
        print(f"   Sample values:")
        print(data.head(2))
        
        # Create filename
        filename = f"{symbol}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to desired format
        if format == 'standard':
            # Standard English format (Date, Open, High, Low, Close, Adj Close, Volume)
            df_output = pd.DataFrame()
            
            # Handle timezone-aware dates
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                # Convert to date only (remove timezone)
                dates = [d.date() for d in data.index]
                df_output['Date'] = [d.strftime('%Y-%m-%d') for d in dates]
            else:
                df_output['Date'] = data.index.strftime('%Y-%m-%d')
            
            # Copy data values with standard Yahoo Finance column names
            df_output['Open'] = data['Open'].values.round(2)
            df_output['High'] = data['High'].values.round(2)
            df_output['Low'] = data['Low'].values.round(2)
            df_output['Close'] = data['Close'].values.round(2)
            df_output['Adj Close'] = data['Volume'].values.astype(int)
            
            print(f"Output DataFrame:")
            print(df_output.head(2))
            
        else:
            # Vietnamese format like existing files
            df_output = pd.DataFrame()
            
            # Handle timezone-aware dates
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                # Convert to date only (remove timezone)
                dates = [d.date() for d in data.index]
                df_output['Ngày'] = [d.strftime('%Y-%m-%d') for d in dates]
            else:
                df_output['Ngày'] = data.index.strftime('%Y-%m-%d')
            
            # Copy data values (not accessing by column names that might not exist)
            df_output['Lần cuối'] = data['Close'].values.round(2)
            df_output['Mở'] = data['Open'].values.round(2) 
            df_output['Cao'] = data['High'].values.round(2)
            df_output['Thấp'] = data['Low'].values.round(2)
            df_output['KL'] = data['Volume'].values.astype(int)
            df_output['% Thay đổi'] = data['Close'].pct_change().fillna(0).values.round(4)
            
            print(f"Output DataFrame:")
            print(df_output.head(2))
        
        # Save to CSV with UTF-8 encoding
        df_output.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Saved {symbol} data to: {filepath}")
        print(f"   Records: {len(df_output)}")
        
        return filepath
    
    def fetch_multiple_stocks(self, symbols, format='standard', **kwargs):
        """Fetch data for multiple stocks"""
        
        results = {}
        
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            
            data = self.fetch_stock_data(symbol, **kwargs)
            
            if data is not None:
                filepath = self.save_to_csv(data, symbol, format)
                results[symbol] = {
                    'data': data,
                    'filepath': filepath,
                    'records': len(data),
                    'date_range': (data.index[0].date(), data.index[-1].date()),
                    'price_range': (data['Close'].min(), data['Close'].max())
                }
            else:
                results[symbol] = None
        
        return results
    
    def get_stock_info(self, symbol):
        """Get additional stock information"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0)
            }
        
        except:
            return {'name': symbol, 'error': 'Could not fetch info'}
    
    def suggest_symbols(self, query):
        """Suggest stock symbols based on company name or keyword"""
        
        query = query.lower().strip()
        suggestions = []
        
        # Direct match
        if query in self.stock_suggestions:
            suggestions.extend(self.stock_suggestions[query])
        
        # Partial match
        for company_name, symbols in self.stock_suggestions.items():
            if query in company_name or company_name in query:
                suggestions.extend(symbols)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for symbol in suggestions:
            if symbol not in seen:
                seen.add(symbol)
                unique_suggestions.append(symbol)
        
        return unique_suggestions[:10]  # Limit to 10 suggestions
    
    def interactive_fetch(self):
        """Interactive mode to fetch stock data from user input"""
        
        print("\nINTERACTIVE YAHOO FINANCE FETCHER")
        print("=" * 50)
        print("Enter stock symbols or company names to fetch data")
        print("Examples: META, AAPL, nike, apple, tesla, FPT, vietcombank")
        print("Smart suggestions will be provided for company names")
        print("Type 'quit', 'exit', or 'q' to stop\n")
        
        fetched_stocks = []
        
        while True:
            try:
                # Get stock symbol from user
                user_input = input("Enter stock symbol or company name (or 'quit'): ").strip()
                
                if user_input.upper() in ['QUIT', 'EXIT', 'Q', '']:
                    if not user_input:
                        continue
                    print("Goodbye!")
                    break
                
                # Validate input format
                if not user_input or len(user_input) < 1:
                    print("Please enter a valid stock symbol or company name")
                    continue
                
                # Check for symbol suggestions
                suggestions = self.suggest_symbols(user_input)
                
                if suggestions and user_input.upper() not in suggestions:
                    print(f"\nFound suggestions for '{user_input}':")
                    for i, suggestion in enumerate(suggestions, 1):
                        # Get company info for display
                        info = self.get_stock_info(suggestion)
                        company_name = info.get('name', suggestion) if 'error' not in info else suggestion
                        print(f"   {i}. {suggestion} - {company_name}")
                    
                    print(f"   0. Use '{user_input.upper()}' as entered")
                    
                    while True:
                        choice = input(f"\nSelect option (0-{len(suggestions)}, default=1): ").strip() or '1'
                        try:
                            choice_num = int(choice)
                            if choice_num == 0:
                                symbol = user_input.upper()
                                break
                            elif 1 <= choice_num <= len(suggestions):
                                symbol = suggestions[choice_num - 1]
                                break
                            else:
                                print(f"Please select a valid option (0-{len(suggestions)})")
                        except ValueError:
                            print(f"Please enter a number (0-{len(suggestions)})")
                else:
                    symbol = user_input.upper()
                
                # Check if already fetched
                if symbol in fetched_stocks:
                    print(f"Warning: {symbol} already fetched. Skipping...")
                    continue
                
                print(f"\nFetching {symbol} data (2013-07-10 to 2024-11-16)...")
                
                # Fetch the data with default date range
                data = self.fetch_stock_data(symbol)
                
                if data is not None:
                    # Save to CSV
                    filepath = self.save_to_csv(data, symbol, format='standard')
                    
                    if filepath:
                        fetched_stocks.append(symbol)
                        
                        # Show summary
                        print(f"\nSUCCESS - {symbol} data saved!")
                        print(f"   File: {filepath}")
                        print(f"   Records: {len(data)}")
                        print(f"   Period: {data.index[0].date()} to {data.index[-1].date()}")
                        print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
                        
                        # Ask to continue
                        continue_choice = input("\nFetch another stock? (y/n, default=y): ").strip().lower()
                        if continue_choice in ['n', 'no']:
                            print("Goodbye!")
                            break
                    else:
                        print(f"Failed to save {symbol} data")
                else:
                    print(f"Failed to fetch data for {symbol}")
                    print("Check if the symbol is correct and try again")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                print("Try again with a different symbol")
        
        # Final summary
        if fetched_stocks:
            print(f"\nFINAL SUMMARY")
            print("=" * 50)
            print(f"Successfully fetched {len(fetched_stocks)} stocks:")
            for stock in fetched_stocks:
                print(f"   {stock}")
        else:
            print("\nNo stocks were fetched.")
            
        return fetched_stocks

def main():
    """Main function with interactive and demo modes"""
    
    print("ADVANCED YAHOO FINANCE FETCHER")
    print("=" * 50)
    print("Choose operation mode:")
    print("   1. Interactive mode (enter symbols manually)")
    print("   2. Demo mode (fetch predefined stocks)")
    print("   3. Quit")
    
    while True:
        choice = input("\nSelect mode (1-3): ").strip()
        
        if choice == '1':
            # Interactive mode
            fetcher = AdvancedYahooFetcher()
            fetcher.interactive_fetch()
            break
            
        elif choice == '2':
            # Demo mode (original functionality)
            print("\nDEMO MODE - Fetching predefined stocks")
            print("=" * 50)
            
            # Initialize fetcher
            fetcher = AdvancedYahooFetcher()
            
            # Define stocks to fetch (matching your existing data)
            stocks = ['META', 'FPT.VN', 'MBB.VN']  # .VN for Vietnamese stocks
            
            # Fetch data for last 5 years
            results = fetcher.fetch_multiple_stocks(
                symbols=stocks,
                period='5y',
                interval='1d',
                format='standard'  # Standard Yahoo Finance format
            )
            
            # Print summary
            print(f"\nFETCH SUMMARY")
            print("=" * 50)
            
            for symbol, result in results.items():
                if result:
                    print(f"{symbol}: SUCCESS")
                    print(f"   Records: {result['records']}")
                    print(f"   Date range: {result['date_range'][0]} to {result['date_range'][1]}")
                    print(f"   Price range: ${result['price_range'][0]:.2f} - ${result['price_range'][1]:.2f}")
                    print(f"   File: {result['filepath']}")
                else:
                    print(f"{symbol}: Failed to fetch")
            
            # Get stock info
            print(f"\nSTOCK INFORMATION")
            print("=" * 50)
            
            for symbol in stocks:
                info = fetcher.get_stock_info(symbol)
                if 'error' not in info:
                    print(f"{symbol}: {info['name']} ({info['sector']})")
                else:
                    print(f"{symbol}: {info['error']}")
            break
            
        elif choice == '3' or choice.upper() in ['Q', 'QUIT', 'EXIT']:
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()