"""
evaluation.py
-------------
Model evaluation, metrics computation, and statistical tests.
Includes blocked cross-validation and multi-seed experiment utilities.
"""
import numpy as np
import pandas as pd
import torch
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import acf, adfuller
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Model Inference
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_seq: np.ndarray, X_extra: np.ndarray,
                   scaler_y=None, mc_samples: int = 50, device: str = 'cpu') -> np.ndarray:
    """
    Get model predictions. For MC Dropout, average over multiple samples.
    
    Args:
        model: Trained MultiFeatureLSTM
        X_seq, X_extra: Input arrays
        scaler_y: RobustScaler for inverse transforming predictions
        mc_samples: Number of MC samples (for monte_carlo dropout)
        device: 'cpu' or 'cuda'
    
    Returns:
        Predictions in original scale (if scaler_y provided)
    """
    if model.dropout_type == 'monte_carlo':
        model.train()  # Keep dropout active
        preds_mc = []
        for _ in range(mc_samples):
            with torch.no_grad():
                out, _ = model(
                    torch.tensor(X_seq, dtype=torch.float32).to(device),
                    torch.tensor(X_extra, dtype=torch.float32).to(device)
                )
            preds_mc.append(out.cpu().numpy().ravel())
        mean_preds = np.mean(preds_mc, axis=0)
        model.eval()
        if scaler_y:
            return scaler_y.inverse_transform(mean_preds.reshape(-1, 1)).ravel()
        return mean_preds

    model.eval()
    with torch.no_grad():
        out, _ = model(
            torch.tensor(X_seq, dtype=torch.float32).to(device),
            torch.tensor(X_extra, dtype=torch.float32).to(device)
        )
    preds = out.cpu().numpy().ravel()
    if scaler_y:
        return scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, MAPE, and R² between true and predicted values.
    """
    # Use np.sqrt for compatibility with older scikit-learn versions
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def compute_implied_return_metrics(y_true_prices: np.ndarray,
                                    y_pred_prices: np.ndarray,
                                    prev_prices: np.ndarray) -> dict:
    """
    Compute R² on implied returns: r_t = (P_t - P_{t-1}) / P_{t-1}.
    This is the 'honest' metric that addresses reviewer concerns.
    """
    true_returns = (y_true_prices - prev_prices) / (prev_prices + 1e-10)
    pred_returns = (y_pred_prices - prev_prices) / (prev_prices + 1e-10)
    r2_returns = r2_score(true_returns, pred_returns)
    return {'R2_Returns': r2_returns}


# ─────────────────────────────────────────────────────────────────────────────
# Random Walk Baseline
# ─────────────────────────────────────────────────────────────────────────────

def random_walk_baseline(y_true: np.ndarray, prev_prices: np.ndarray) -> dict:
    """
    Compute metrics for the naive random walk predictor: P̂_t = P_{t-1}.
    """
    return compute_metrics(y_true, prev_prices)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Tests
# Note: For stationarity tests (ADF, KPSS), use preprocessing.py module
# ─────────────────────────────────────────────────────────────────────────────


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray,
                         h: int = 1, power: int = 2) -> dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Args:
        e1: Errors from model 1
        e2: Errors from model 2
        h: Forecast horizon
        power: Loss function power (2 for MSE)
    
    Returns:
        DM statistic and p-value
    """
    d = np.abs(e1)**power - np.abs(e2)**power
    acov = acf(d, nlags=h-1, fft=False)
    gamma0 = acov[0]
    gamma_sum = acov[1:h].sum() if h > 1 else 0.0
    var_d = gamma0 + 2 * gamma_sum
    dm_stat = d.mean() / np.sqrt(var_d / len(d))
    p_value = 2 * sps.t.sf(abs(dm_stat), df=len(d)-1)
    return {'DM_statistic': dm_stat, 'pvalue': p_value}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                         prev_prices: np.ndarray) -> float:
    """
    Compute directional accuracy: % of times the model correctly
    predicts whether price goes up or down.
    """
    true_direction = np.sign(y_true - prev_prices)
    pred_direction = np.sign(y_pred - prev_prices)
    return np.mean(true_direction == pred_direction)


# ─────────────────────────────────────────────────────────────────────────────
# Blocked Cross-Validation (Reviewer Comment 1.3)
# ─────────────────────────────────────────────────────────────────────────────

class BlockedTimeSeriesSplit:
    """
    Time series cross-validation with a gap between train and validation sets
    to prevent data leakage from temporal autocorrelation.
    
    Unlike standard TimeSeriesSplit, this adds a 'gap' of observations
    between the training and validation sets.
    """
    def __init__(self, n_splits: int = 5, gap: int = 10):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X):
        """
        Generate indices for blocked time series splits.
        
        Yields:
            train_idx, val_idx tuples
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.gap
            val_end = val_start + fold_size
            
            if val_end > n_samples:
                val_end = n_samples
            
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            
            if len(val_idx) > 0:
                yield train_idx, val_idx
    
    def get_n_splits(self):
        return self.n_splits


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Seed Experiments (Reviewer Comment 1.6)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_multi_seed_results(results_list: List[Dict]) -> Dict:
    """
    Aggregate results from multiple seed runs.
    
    Args:
        results_list: List of metric dictionaries from different seeds
    
    Returns:
        Dict with mean ± std for each metric
    """
    if not results_list:
        return {}
    
    metrics = results_list[0].keys()
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in results_list if metric in r]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_values'] = values
    
    return aggregated


def format_multi_seed_table(aggregated: Dict) -> pd.DataFrame:
    """
    Format aggregated multi-seed results as a table.
    """
    rows = []
    for key in aggregated:
        if key.endswith('_mean'):
            metric = key.replace('_mean', '')
            mean_val = aggregated[f'{metric}_mean']
            std_val = aggregated.get(f'{metric}_std', 0)
            rows.append({
                'Metric': metric,
                'Mean': mean_val,
                'Std': std_val,
                'Mean ± Std': f"{mean_val:.4f} ± {std_val:.4f}"
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Regime-Based Evaluation (Reviewer Comment 1.5)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_by_regime(y_true: np.ndarray, y_pred: np.ndarray,
                       regime_labels: np.ndarray) -> pd.DataFrame:
    """
    Compute metrics separately for each regime (e.g., high_vol, low_vol, covid).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        regime_labels: Array of regime labels (same length as y_true)
    
    Returns:
        DataFrame with metrics per regime
    """
    results = []
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        if mask.sum() > 0:
            metrics = compute_metrics(y_true[mask], y_pred[mask])
            metrics['Regime'] = regime
            metrics['N_samples'] = mask.sum()
            results.append(metrics)
    
    return pd.DataFrame(results)[['Regime', 'N_samples', 'RMSE', 'MAE', 'R2']]


def compute_dropout_vix_correlation(dropout_rates: np.ndarray,
                                      vix_values: np.ndarray) -> Dict:
    """
    Compute correlation between adaptive dropout rates and VIX (volatility index).
    This addresses reviewer's request to show relationship between model
    uncertainty and market conditions.
    
    Returns:
        Pearson correlation coefficient and p-value
    """
    if len(dropout_rates) != len(vix_values):
        raise ValueError("Arrays must have same length")
    
    # Remove any NaN values
    mask = ~(np.isnan(dropout_rates) | np.isnan(vix_values))
    dr_clean = dropout_rates[mask]
    vix_clean = vix_values[mask]
    
    corr, pvalue = sps.pearsonr(dr_clean, vix_clean)
    return {
        'correlation': corr,
        'pvalue': pvalue,
        'interpretation': 'positive' if corr > 0 else 'negative'
    }

