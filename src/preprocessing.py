"""
preprocessing.py
----------------
Pre-estimation tests (stationarity) and data transformations.

This module handles:
1. Stationarity tests (ADF, KPSS)
2. Data transformations (returns, log, differencing)
3. Inverse transformations for prediction interpretation

Usage:
    # In config.py, set:
    PREPROCESSING_CONFIG = {
        'transformation': 'price',  # or 'returns', 'log_returns', 'difference'
    }
    
    # Then in main.py:
    df = apply_transformation(df)  # Applies based on config
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from statsmodels.tsa.stattools import adfuller, kpss

from .config import PREPROCESSING_CONFIG, TARGET_COL


# ─────────────────────────────────────────────────────────────────────────────
# Stationarity Tests
# ─────────────────────────────────────────────────────────────────────────────

def adf_test(series: np.ndarray, verbose: bool = False) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    H0: Series has a unit root (non-stationary)
    Reject H0 if p-value < 0.05
    
    Returns:
        Dict with statistic, pvalue, is_stationary
    """
    result = adfuller(series, autolag='AIC')
    output = {
        'test': 'ADF',
        'statistic': result[0],
        'pvalue': result[1],
        'critical_1%': result[4]['1%'],
        'critical_5%': result[4]['5%'],
        'critical_10%': result[4]['10%'],
        'is_stationary': result[1] < 0.05
    }
    
    if verbose:
        print(f"ADF Test: statistic={output['statistic']:.4f}, p-value={output['pvalue']:.4f}")
        print(f"   → {'Stationary' if output['is_stationary'] else 'Non-stationary'} at 5% level")
    
    return output


def kpss_test(series: np.ndarray, regression: str = 'c', verbose: bool = False) -> Dict:
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
    
    H0: Series is stationary
    Reject H0 if p-value < 0.05
    
    Note: KPSS is the opposite of ADF!
    """
    result = kpss(series, regression=regression, nlags='auto')
    output = {
        'test': 'KPSS',
        'statistic': result[0],
        'pvalue': result[1],
        'critical_1%': result[3]['1%'],
        'critical_5%': result[3]['5%'],
        'critical_10%': result[3]['10%'],
        'is_stationary': result[1] >= 0.05  # Note: opposite of ADF
    }
    
    if verbose:
        print(f"KPSS Test: statistic={output['statistic']:.4f}, p-value={output['pvalue']:.4f}")
        print(f"   → {'Stationary' if output['is_stationary'] else 'Non-stationary'} at 5% level")
    
    return output


def run_stationarity_tests(series: np.ndarray, name: str = 'Series') -> pd.DataFrame:
    """
    Run both ADF and KPSS tests and return summary table.
    
    Interpretation:
        - ADF: rejects unit root → stationary
        - KPSS: fails to reject stationarity → stationary
        - Both agree → confident conclusion
    """
    adf = adf_test(series)
    kpss_result = kpss_test(series)
    
    results = pd.DataFrame([
        {
            'Series': name,
            'Test': 'ADF',
            'Statistic': adf['statistic'],
            'P-Value': adf['pvalue'],
            'Stationary': adf['is_stationary']
        },
        {
            'Series': name,
            'Test': 'KPSS',
            'Statistic': kpss_result['statistic'],
            'P-Value': kpss_result['pvalue'],
            'Stationary': kpss_result['is_stationary']
        }
    ])
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Automatic Stationarity
# ─────────────────────────────────────────────────────────────────────────────

def ensure_stationarity(df: pd.DataFrame, columns: list = None, 
                        method: str = 'log_returns',
                        verbose: bool = True) -> pd.DataFrame:
    """
    Test all specified columns for stationarity and transform non-stationary ones.
    
    Args:
        df: DataFrame with data
        columns: List of columns to test (default: all numeric except Date)
        method: Transformation method ('log_returns', 'returns', 'difference')
        verbose: Print results
    
    Returns:
        DataFrame with non-stationary columns transformed
    """
    df = df.copy()
    
    if columns is None:
        columns = [c for c in df.columns if c not in ['Date', '_transformation'] 
                   and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if verbose:
        print("\n📊 Stationarity Check (ADF Test)")
        print("=" * 50)
    
    transformed_cols = []
    stationary_cols = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Skip columns that are already returns-like (small values, centered near 0)
        col_mean = df[col].abs().mean()
        col_std = df[col].std()
        
        try:
            result = adf_test(df[col].dropna().values, verbose=False)
            is_stationary = result['is_stationary']
            
            if verbose:
                status = "✅ Stationary" if is_stationary else "❌ Non-stationary"
                print(f"  {col:15} ADF={result['statistic']:7.4f}, p={result['pvalue']:.4f} → {status}")
            
            if not is_stationary:
                # Transform to make stationary
                original_col = df[col].copy()
                
                if method == 'log_returns':
                    # Check if values are positive (required for log)
                    if (df[col] > 0).all():
                        df[col] = compute_log_returns(df[col].values)
                    else:
                        df[col] = compute_returns(df[col].values)
                elif method == 'returns':
                    df[col] = compute_returns(df[col].values)
                elif method == 'difference':
                    df[col] = compute_difference(df[col].values)
                
                transformed_cols.append(col)
                
                # Verify transformation worked
                new_result = adf_test(df[col].dropna().values, verbose=False)
                if verbose and new_result['is_stationary']:
                    print(f"    → Transformed to {method}, now stationary ✓")
            else:
                stationary_cols.append(col)
                
        except Exception as e:
            if verbose:
                print(f"  {col:15} Error: {e}")
    
    if verbose:
        print("=" * 50)
        print(f"Stationary: {len(stationary_cols)}, Transformed: {len(transformed_cols)}")
    
    # Drop NaN rows created by transformation
    df = df.dropna().reset_index(drop=True)
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Transformations
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}"""
    returns = np.diff(prices) / prices[:-1]
    return np.concatenate([[np.nan], returns])


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns: r_t = log(P_t / P_{t-1})"""
    log_returns = np.diff(np.log(prices))
    return np.concatenate([[np.nan], log_returns])


def compute_difference(series: np.ndarray, order: int = 1) -> np.ndarray:
    """Compute first difference: Δy_t = y_t - y_{t-1}"""
    diff = np.diff(series, n=order)
    return np.concatenate([[np.nan] * order, diff])


def apply_transformation(df: pd.DataFrame, transformation: str = None,
                          transform_inputs: bool = None,
                          input_cols: list = None) -> pd.DataFrame:
    """
    Apply transformation to target column and optionally input columns.
    
    Transformations:
        - 'price': No transformation (raw prices)
        - 'returns': Simple returns
        - 'log_returns': Log returns  
        - 'difference': First difference
    
    Args:
        df: DataFrame with price data
        transformation: Type of transformation ('price', 'returns', etc.)
        transform_inputs: If True, also transform input columns to returns
        input_cols: List of input column names to transform (if transform_inputs=True)
    
    Adds new column 'target' which is used by the model.
    Keeps original 'price' column for inverse transformation.
    """
    transformation = transformation or PREPROCESSING_CONFIG.get('transformation', 'price')
    transform_inputs = transform_inputs if transform_inputs is not None else PREPROCESSING_CONFIG.get('transform_inputs', False)
    
    df = df.copy()
    
    # Transform target column
    if transformation == 'price':
        df['target'] = df[TARGET_COL]
        df['_transformation'] = 'price'
        
    elif transformation == 'returns':
        df['target'] = compute_returns(df[TARGET_COL].values)
        df['_transformation'] = 'returns'
        
    elif transformation == 'log_returns':
        df['target'] = compute_log_returns(df[TARGET_COL].values)
        df['_transformation'] = 'log_returns'
        
    elif transformation == 'difference':
        df['target'] = compute_difference(df[TARGET_COL].values)
        df['_transformation'] = 'difference'
        
    else:
        raise ValueError(f"Unknown transformation: {transformation}")
    
    # Transform input columns (for stationarity)
    if transform_inputs and transformation != 'price':
        # Get input columns from config if not provided
        if input_cols is None:
            from .config import VARIABLE_CONFIG
            input_cols = VARIABLE_CONFIG.get('extra_feature_cols', [])
        
        print(f"📐 Transforming {len(input_cols)} input columns to {transformation}...")
        
        for col in input_cols:
            if col in df.columns and col != TARGET_COL:
                # Skip derived columns like MA_20, volume (they need different treatment)
                if col.startswith('MA_') or col == 'volume':
                    continue
                    
                if transformation == 'returns':
                    df[col] = compute_returns(df[col].values)
                elif transformation == 'log_returns':
                    df[col] = compute_log_returns(df[col].values)
                elif transformation == 'difference':
                    df[col] = compute_difference(df[col].values)
    
    # Drop NaN rows created by transformation
    df = df.dropna().reset_index(drop=True)
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Inverse Transformations (for predictions)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_transform_predictions(predictions: np.ndarray, 
                                   prev_prices: np.ndarray,
                                   transformation: str) -> np.ndarray:
    """
    Convert transformed predictions back to price levels.
    
    Args:
        predictions: Model predictions (in transformed space)
        prev_prices: Previous prices (P_{t-1})
        transformation: Type of transformation used
    
    Returns:
        Predictions in original price space
    """
    if transformation == 'price':
        return predictions
    
    elif transformation == 'returns':
        # P_t = P_{t-1} * (1 + r_t)
        return prev_prices * (1 + predictions)
    
    elif transformation == 'log_returns':
        # P_t = P_{t-1} * exp(r_t)
        return prev_prices * np.exp(predictions)
    
    elif transformation == 'difference':
        # P_t = P_{t-1} + diff_t
        return prev_prices + predictions
    
    else:
        raise ValueError(f"Unknown transformation: {transformation}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Estimation Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def run_pre_estimation_diagnostics(df: pd.DataFrame, 
                                    target_col: str = None,
                                    verbose: bool = True) -> Dict:
    """
    Run all pre-estimation tests on the target series.
    
    Returns:
        Dict with test results for both raw prices and transformed target
    """
    target_col = target_col or TARGET_COL
    
    results = {}
    
    # Test raw prices
    if verbose:
        print("📊 Pre-Estimation Diagnostics")
        print("=" * 50)
        print("\n1. Raw Prices:")
    
    results['price_adf'] = adf_test(df[target_col].values, verbose=verbose)
    results['price_kpss'] = kpss_test(df[target_col].values, verbose=verbose)
    
    # Test transformed target (if different)
    if 'target' in df.columns and df['_transformation'].iloc[0] != 'price':
        transformation = df['_transformation'].iloc[0]
        if verbose:
            print(f"\n2. Transformed ({transformation}):")
        
        results['target_adf'] = adf_test(df['target'].values, verbose=verbose)
        results['target_kpss'] = kpss_test(df['target'].values, verbose=verbose)
    
    # Summary
    if verbose:
        print("\n" + "=" * 50)
        price_stationary = results['price_adf']['is_stationary'] and results['price_kpss']['is_stationary']
        print(f"Price is {'stationary' if price_stationary else '⚠️ NON-STATIONARY'}")
        
        if 'target_adf' in results:
            target_stationary = results['target_adf']['is_stationary'] and results['target_kpss']['is_stationary']
            print(f"Target is {'stationary' if target_stationary else '⚠️ NON-STATIONARY'}")
    
    return results
