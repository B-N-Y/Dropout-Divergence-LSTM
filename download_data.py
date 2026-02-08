#!/usr/bin/env python
"""
download_data.py
----------------
Downloads the necessary financial data from Yahoo Finance for the Gold Price experiments.
No existing CSV files are required.

Data Features:
- Target: GLD (Gold Price)
- Macro: DXY (Dollar Index), TNX (10-Yr Treasury), VIX (Volatility)

Output:
- data/gold_data.csv
"""
import yfinance as yf
import pandas as pd
import os
import numpy as np

# Configuration
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"  # Or current date
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def download_gold_data():
    """Download and merge data for Gold Price Experiment."""
    print(f"Downloading data from {START_DATE} to {END_DATE}...")
    
    # 1. Define Tickers
    tickers = {
        'price': 'GLD',
        'dxy': 'DX-Y.NYB',
        'tnx': '^TNX',
        'vix': '^VIX'
    }
    
    merged_df = None
    
    # 2. Download loop
    for col_name, ticker_symbol in tickers.items():
        print(f"   Fetching {ticker_symbol} ({col_name})...")
        try:
            df = yf.download(ticker_symbol, start=START_DATE, end=END_DATE, progress=False)
            
            if len(df) == 0:
                print(f"   ⚠️ Warning: No data found for {ticker_symbol}")
                continue
                
            # Formatting
            df = df.reset_index()
            # Handle MultiIndex columns (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Select Price (Adj Close or Close)
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            
            # Create temp dataframe
            temp_df = df[['Date']].copy()
            temp_df['Date'] = pd.to_datetime(temp_df['Date'])
            temp_df[col_name] = df[price_col]
            
            # Add Volume for the target asset (GLD)
            if col_name == 'price' and 'Volume' in df.columns:
                temp_df['volume'] = df['Volume']
            
            # Merge
            if merged_df is None:
                merged_df = temp_df
            else:
                merged_df = pd.merge(merged_df, temp_df, on='Date', how='inner')
                
        except Exception as e:
            print(f"   ❌ Error downloading {ticker_symbol}: {e}")
            
    # 3. Feature Engineering (moving averages, etc.)
    if merged_df is not None:
        print("   Calculating technical indicators...")
        merged_df = merged_df.sort_values('Date')
        
        # Simple Moving Average (20 days)
        merged_df['MA_20'] = merged_df['price'].rolling(window=20).mean()
        
        # Drop initial NaNs from rolling window
        merged_df = merged_df.dropna()
        
        # 4. Save
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            
        output_path = os.path.join(DATA_DIR, 'gold_data.csv')
        merged_df.to_csv(output_path, index=False)
        print(f"✅ Success! Data saved to: {output_path}")
        print(f"   Rows: {len(merged_df)}")
        print(f"   Date Range: {merged_df['Date'].min().date()} to {merged_df['Date'].max().date()}")
        
    else:
        print("❌ Failed to create dataset.")

if __name__ == "__main__":
    download_gold_data()
