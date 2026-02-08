"""
data_loader.py
--------------
Functions for fetching financial data from Yahoo Finance and preparing
the dataset for LSTM training.

Variable configuration is done via config.py - see VARIABLE_CONFIG.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

from .config import VARIABLE_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def download_ticker(ticker: str, new_name: str, start_date: str, end_date: str,
                    interval: str = "1d") -> pd.DataFrame:
    """
    Generic function to download any ticker and rename the price column.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    df = df.reset_index()
    price_column = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Date'] = pd.to_datetime(df['Date'])
    df[new_name] = df[price_column]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[['Date', new_name]]


def load_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load target asset and all input variables.
    
    Uses data_file from current experiment config (e.g., 'gold_data.csv' or 'sp500_ff5_data.csv').
    Falls back to Yahoo Finance if local file not found.
    
    Returns:
        DataFrame with Date, price (target), volume, and all extra features
    """
    from .config import START_DATE, END_DATE, VARIABLE_CONFIG, CURRENT_EXPERIMENT, EXPERIMENTS
    import os
    
    start_date = start_date or START_DATE
    end_date = end_date or END_DATE
    
    # Get data file from current experiment config
    exp_config = EXPERIMENTS.get(CURRENT_EXPERIMENT, EXPERIMENTS.get('gold_price_p1', list(EXPERIMENTS.values())[0]))
    data_file = exp_config.get('data_file', 'gold_data.csv')
    
    # Check for local data file first (for TRUBA/offline use)
    local_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', data_file)
    if os.path.exists(local_file):
        print(f"📁 Loading data from local file: {local_file}")
        df = pd.read_csv(local_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Apply experiment period filter (from experiment config)
        if 'period' in exp_config:
            period_start, period_end = exp_config['period']
            period_start = pd.to_datetime(period_start)
            period_end = pd.to_datetime(period_end)
            df = df[(df['Date'] >= period_start) & (df['Date'] < period_end)]
            df = df.reset_index(drop=True)
            print(f"📅 Period: {period_start.date()} to {period_end.date()}")
        
        print(f"✅ Loaded {len(df)} observations")
        return df
    
    # Fallback to Yahoo Finance
    config = VARIABLE_CONFIG
    
    # Download target asset
    print(f"📊 Downloading {config['target_name']} ({config['target_ticker']})...")
    main_df = yf.download(config['target_ticker'], start=start_date, end=end_date, interval="1d")
    main_df = main_df.reset_index()
    price_col = 'Adj Close' if 'Adj Close' in main_df.columns else 'Close'
    
    if isinstance(main_df.columns, pd.MultiIndex):
        main_df.columns = main_df.columns.get_level_values(0)
    
    main_df['Date'] = pd.to_datetime(main_df['Date'])
    main_df['price'] = main_df[price_col]
    main_df['volume'] = main_df['Volume']
    
    # Technical indicators
    main_df['MA_20'] = main_df['price'].rolling(20).mean()
    main_df['MA_50'] = main_df['price'].rolling(50).mean()
    
    result = main_df[['Date', 'price', 'volume', 'MA_20', 'MA_50']].copy()
    
    # Download input variables (extra features)
    print(f"📊 Downloading {len(config['input_tickers'])} input variables...")
    for name, ticker in config['input_tickers'].items():
        extra_df = download_ticker(ticker, name, start_date, end_date)
        # MERGE: Inner join on Date
        result = result.merge(extra_df, on='Date', how='inner')
    
    # ─────────────────────────────────────────────────────────────────────────────
    # CRITICAL: PREVENT DATA LEAKAGE
    # Shift all extra features by 1 so we use (t-1) data to predict (t)
    # The target 'price' remains at (t).
    # ─────────────────────────────────────────────────────────────────────────────
    feature_cols = [c for c in result.columns if c not in ['Date', 'price', 'volume']]
    
    # Note: 'volume' is often used as a feature, so we should decide if we shift it.
    # Usually volume(t) is not known at start of day t. So we shift volume too.
    # Adding volume to features to be shifted if it exists
    if 'volume' in result.columns:
        feature_cols.append('volume')
    
    print(f"🔒 Shifting {len(feature_cols)} features by 1 to prevent leakage...")
    for col in feature_cols:
        result[col] = result[col].shift(1)
        
    # Remove rows with NaNs created by shifting
    result = result.dropna().reset_index(drop=True)
    print(f"✅ Loaded {len(result)} observations")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

def add_regime_labels(df: pd.DataFrame, vix_threshold: float = 25.0) -> pd.DataFrame:
    """
    Add regime labels for conditional analysis.
    
    Regimes:
        - 'low_vol': VIX < threshold
        - 'high_vol': VIX >= threshold
        - 'covid': 2020-03-01 to 2020-06-30
    """
    df = df.copy()
    
    # VIX-based volatility regime
    if 'vix' in df.columns:
        df['regime_vol'] = np.where(df['vix'] >= vix_threshold, 'high_vol', 'low_vol')
    
    # COVID crash period
    df['regime_covid'] = 'normal'
    if 'Date' in df.columns:
        covid_start = pd.to_datetime('2020-03-01')
        covid_end = pd.to_datetime('2020-06-30')
        mask = (df['Date'] >= covid_start) & (df['Date'] <= covid_end)
        df.loc[mask, 'regime_covid'] = 'covid_crash'
    
    return df


def split_by_regime(df: pd.DataFrame, indices: np.ndarray,
                    regime_col: str = 'regime_vol') -> dict:
    """
    Split data indices by regime for conditional evaluation.
    """
    regime_indices = {}
    for regime in df[regime_col].unique():
        mask = df.iloc[indices][regime_col] == regime
        regime_indices[regime] = indices[mask.values]
    return regime_indices


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target_col: str = None,
                     n_lags: int = None) -> pd.DataFrame:
    """
    Add lagged features for the target column.
    
    For price experiments: lags of 'price'
    For returns experiments: lags of 'target' (which contains returns after transformation)
    
    Uses N_LAGS from config if not specified.
    """
    from .config import N_LAGS
    n_lags = n_lags or N_LAGS
    
    df = df.copy()
    
    # Determine which column to create lags from
    # If 'target' column exists (after transformation), use it for returns experiments
    # Otherwise use 'price' for price experiments
    if target_col is None:
        if 'target' in df.columns and '_transformation' in df.columns:
            transformation = df['_transformation'].iloc[0] if len(df) > 0 else 'price'
            if transformation in ['returns', 'log_returns', 'difference']:
                target_col = 'target'
            else:
                target_col = 'price'
        else:
            target_col = 'price'
    
    # Create lag features
    for i in range(1, n_lags + 1):
        df[f"{target_col}_Lag{i}"] = df[target_col].shift(i)
    
    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Train/Val/Test Split & Scaling
# ─────────────────────────────────────────────────────────────────────────────

def prepare_train_val_test(df: pd.DataFrame, n_splits: int = 6):
    """
    Use TimeSeriesSplit to create train/val/test indices.
    Uses variable definitions from VARIABLE_CONFIG.
    
    Returns:
        Dict with scaled arrays ready for PyTorch DataLoader
    """
    from .config import VARIABLE_CONFIG, N_LAGS, TARGET_COL
    
    # Determine which column was used for lags based on transformation
    if 'target' in df.columns and '_transformation' in df.columns:
        transformation = df['_transformation'].iloc[0] if len(df) > 0 else 'price'
        if transformation in ['returns', 'log_returns', 'difference']:
            lag_base = 'target'
            target_col = 'target'
        else:
            lag_base = 'price'
            target_col = 'price'
    else:
        lag_base = 'price'
        target_col = TARGET_COL
    
    # Get lag column names
    lag_cols = [f"{lag_base}_Lag{i}" for i in range(1, N_LAGS + 1)]
    
    # Verify lag columns exist
    existing_lag_cols = [c for c in lag_cols if c in df.columns]
    if len(existing_lag_cols) == 0:
        raise ValueError(f"No lag columns found. Expected columns like {lag_cols}. Available: {df.columns.tolist()}")
    
    extra_cols = VARIABLE_CONFIG['extra_feature_cols']
    
    # Filter to only existing columns
    extra_cols = [c for c in extra_cols if c in df.columns]
    
    X_seq   = df[existing_lag_cols].values.reshape(-1, len(existing_lag_cols), 1)
    X_extra = df[extra_cols].values
    y_raw   = df[target_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X_seq))
    train_idx, val_idx = splits[-2]
    _, test_idx = splits[-1]

    # Scalers
    scaler_seq   = RobustScaler()
    scaler_extra = RobustScaler()
    scaler_y     = RobustScaler()

    # Fit on train, transform all
    X_train_seq = scaler_seq.fit_transform(X_seq[train_idx].reshape(-1, 1)).reshape(-1, len(lag_cols), 1)
    X_val_seq   = scaler_seq.transform(X_seq[val_idx].reshape(-1, 1)).reshape(-1, len(lag_cols), 1)
    X_test_seq  = scaler_seq.transform(X_seq[test_idx].reshape(-1, 1)).reshape(-1, len(lag_cols), 1)

    X_train_extra = scaler_extra.fit_transform(X_extra[train_idx])
    X_val_extra   = scaler_extra.transform(X_extra[val_idx])
    X_test_extra  = scaler_extra.transform(X_extra[test_idx])

    y_train_s = scaler_y.fit_transform(y_raw[train_idx].reshape(-1, 1)).ravel()
    y_val_s   = scaler_y.transform(y_raw[val_idx].reshape(-1, 1)).ravel()
    y_test_s  = scaler_y.transform(y_raw[test_idx].reshape(-1, 1)).ravel()

    return {
        'X_train_seq': X_train_seq, 'X_val_seq': X_val_seq, 'X_test_seq': X_test_seq,
        'X_train_extra': X_train_extra, 'X_val_extra': X_val_extra, 'X_test_extra': X_test_extra,
        'y_train_s': y_train_s, 'y_val_s': y_val_s, 'y_test_s': y_test_s,
        'scaler_y': scaler_y,
        'scaler_seq': scaler_seq,
        'scaler_extra': scaler_extra,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'dates': df['Date'].values,
        'lag_cols': lag_cols,
        'extra_cols': extra_cols
    }


# ─────────────────────────────────────────────────────────────────────────────
# Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptive_stats(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """
    Compute mean, std, skewness, kurtosis, min, max for given columns.
    Uses VARIABLE_CONFIG if cols not specified.
    """
    from .config import VARIABLE_CONFIG, TARGET_COL
    
    if cols is None:
        cols = [TARGET_COL, 'volume'] + list(VARIABLE_CONFIG['input_tickers'].keys())
        cols = [c for c in cols if c in df.columns]
    
    summary = []
    for c in cols:
        s = df[c].dropna()
        summary.append({
            'Series':   c,
            'μ (Mean)': round(s.mean(), 6),
            'σ (Std)':  round(s.std(), 6),
            'Skewness': round(skew(s, bias=False), 6),
            'Kurtosis': round(kurtosis(s, fisher=True, bias=False), 6),
            'Min':      round(s.min(), 6),
            'Max':      round(s.max(), 6)
        })
    return pd.DataFrame(summary)
