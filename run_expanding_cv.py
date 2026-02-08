#!/usr/bin/env python
"""
run_expanding_cv.py
-------------------
EXPANDING WINDOW CROSS-VALIDATION for Time Series

This implements proper K-fold cross-validation for time series data,
where each fold uses an expanding training window and tests on the next period.

Fold 0: Train[2010-2012] → Test[2012-2013]
Fold 1: Train[2010-2013] → Test[2013-2014]
Fold 2: Train[2010-2014] → Test[2014-2015]
...

This addresses Reviewer #1's request for:
"alternative validation schemes (e.g., blocked cross-validation or extended rolling horizons)"

Usage:
    python run_expanding_cv.py --config gold_price_2010_2025 --n-folds 5
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from datetime import datetime


def setup_experiment(config_name: str):
    """Setup experiment configuration before importing other modules."""
    import src.config as cfg
    cfg.CURRENT_EXPERIMENT = config_name
    exp = cfg.EXPERIMENTS.get(config_name, cfg.EXPERIMENTS.get('gold_price_2010_2025', list(cfg.EXPERIMENTS.values())[0]))
    cfg.VARIABLE_CONFIG = {
        'target_ticker': exp['target_ticker'],
        'target_name': exp['target_name'],
        'input_tickers': exp.get('input_tickers', {}),
        'extra_feature_cols': exp['extra_feature_cols'],
    }
    cfg.PREPROCESSING_CONFIG = {
        'transformation': exp.get('transformation', 'price'),
        'transform_inputs': exp.get('transform_inputs', False),
        'run_diagnostics': False,
    }
    return exp


from src.config import (
    set_seeds, DEVICE,
    SEARCH_EPOCHS, RETRAIN_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    HYPERPARAM_SPACE, VARIABLE_CONFIG, TARGET_COL, N_LAGS
)
from src.data_loader import load_data, add_lag_features
from src.preprocessing import apply_transformation
from src.models import MultiFeatureLSTM
from src.training import train_model
from src.evaluation import compute_metrics


def get_best_config_from_results(config_name: str) -> tuple:
    """
    Read best dropout/divergence configuration from main grid search results.
    """
    results_dir = f'results_{config_name}'
    best_file = os.path.join(results_dir, 'best_per_combo.csv')
    
    if os.path.exists(best_file):
        df = pd.read_csv(best_file)
        if 'Test_R2' in df.columns and len(df) > 0:
            best_row = df.loc[df['Test_R2'].idxmax()]
            dt = best_row['Dropout_Type']
            dm = best_row['Divergence']
            print(f"📖 Loaded best config: {dt} + {dm} (Test R² = {best_row['Test_R2']:.4f})")
            return (dt, dm)
    
    print(f"⚠️  No results found, using default: dynamic + hellinger")
    return ('dynamic', 'hellinger')


def get_best_hyperparams_from_results(config_name: str, dropout_type: str, divergence: str) -> dict:
    """
    Read best hyperparameters for a specific dropout/divergence combo.
    """
    results_dir = f'results_{config_name}'
    best_file = os.path.join(results_dir, 'best_per_combo.csv')
    
    default_params = {
        'num_layers': 3,
        'hidden_size': 256,
        'base_rate': 0.3,
        'alpha': 5.0
    }
    
    if os.path.exists(best_file):
        df = pd.read_csv(best_file)
        mask = (df['Dropout_Type'] == dropout_type) & (df['Divergence'] == divergence)
        if mask.any():
            row = df[mask].iloc[0]
            return {
                'num_layers': int(row.get('Num_Layers', 3)),
                'hidden_size': int(row.get('Hidden_Size', 256)),
                'base_rate': float(row.get('Base_Rate', 0.3)),
                'alpha': float(row.get('Alpha', 5.0))
            }
    
    return default_params


def prepare_fold_data(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    """
    Prepare data for a single fold of expanding window CV.
    """
    from src.config import VARIABLE_CONFIG, N_LAGS
    
    # Determine target column based on transformation
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
        target_col = 'price'
    
    lag_cols = [f"{lag_base}_Lag{i}" for i in range(1, N_LAGS + 1)]
    lag_cols = [c for c in lag_cols if c in df.columns]
    
    extra_cols = VARIABLE_CONFIG['extra_feature_cols']
    extra_cols = [c for c in extra_cols if c in df.columns]
    
    if len(lag_cols) == 0:
        raise ValueError(f"No lag columns found in dataframe")
    
    X_seq = df[lag_cols].values.reshape(-1, len(lag_cols), 1)
    X_extra = df[extra_cols].values
    y_raw = df[target_col].values
    
    # Fit scalers on train data only
    scaler_seq = RobustScaler()
    scaler_extra = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_seq = scaler_seq.fit_transform(X_seq[train_idx].reshape(-1, 1)).reshape(-1, len(lag_cols), 1)
    X_test_seq = scaler_seq.transform(X_seq[test_idx].reshape(-1, 1)).reshape(-1, len(lag_cols), 1)
    
    X_train_extra = scaler_extra.fit_transform(X_extra[train_idx])
    X_test_extra = scaler_extra.transform(X_extra[test_idx])
    
    y_train_s = scaler_y.fit_transform(y_raw[train_idx].reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_raw[test_idx].reshape(-1, 1)).ravel()
    
    return {
        'X_train_seq': X_train_seq, 'X_test_seq': X_test_seq,
        'X_train_extra': X_train_extra, 'X_test_extra': X_test_extra,
        'y_train_s': y_train_s, 'y_test_s': y_test_s,
        'scaler_y': scaler_y,
        'train_idx': train_idx, 'test_idx': test_idx,
        'dates': df['Date'].values,
        'n_lag_features': len(lag_cols),
        'n_extra_features': len(extra_cols)
    }


def train_and_evaluate_fold(fold_data: dict, dropout_type: str, divergence: str, 
                            hyperparams: dict, epochs: int = 30, seed: int = 42):
    """
    Train model on one fold and return test metrics.
    """
    set_seeds(seed)
    
    n_lag = fold_data['n_lag_features']
    n_extra = fold_data['n_extra_features']
    
    model = MultiFeatureLSTM(
        input_size=1,  # Each lag feature has size 1
        hidden_size=hyperparams['hidden_size'],
        num_layers=hyperparams['num_layers'],
        extra_features=n_extra,
        dropout_type=dropout_type,
        base_rate=hyperparams['base_rate'],
        alpha=hyperparams['alpha'],
        divergence_method=divergence
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    history = train_model(
        model=model,
        optimizer=optimizer,
        X_train_seq=fold_data['X_train_seq'],
        X_train_extra=fold_data['X_train_extra'],
        y_train=fold_data['y_train_s'],
        X_val_seq=fold_data['X_test_seq'],
        X_val_extra=fold_data['X_test_extra'],
        y_val=fold_data['y_test_s'],
        epochs=epochs,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    
    # Evaluate on test
    model.eval()
    with torch.no_grad():
        X_test_seq_t = torch.tensor(fold_data['X_test_seq'], dtype=torch.float32).to(DEVICE)
        X_test_extra_t = torch.tensor(fold_data['X_test_extra'], dtype=torch.float32).to(DEVICE)
        output, _ = model(X_test_seq_t, X_test_extra_t)  # model returns (output, divergence)
        y_pred_scaled = output.cpu().numpy().ravel()
    
    y_pred = fold_data['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = fold_data['scaler_y'].inverse_transform(fold_data['y_test_s'].reshape(-1, 1)).ravel()
    
    metrics = compute_metrics(y_true, y_pred)
    
    return {
        'r2': metrics['R2'],
        'rmse': metrics['RMSE'],
        'mae': metrics['MAE'],
        'mape': metrics.get('MAPE', np.nan),
        'train_size': len(fold_data['train_idx']),
        'test_size': len(fold_data['test_idx'])
    }


def run_expanding_cv(config_name: str, n_folds: int = 5, epochs: int = 30):
    """
    Run expanding window cross-validation.
    """
    print("=" * 70)
    print("EXPANDING WINDOW CROSS-VALIDATION")
    print("=" * 70)
    print(f"Config: {config_name}")
    print(f"Folds: {n_folds}")
    print(f"Epochs per fold: {epochs}")
    print("=" * 70)
    
    # Setup and load data
    exp_config = setup_experiment(config_name)
    print(f"\n📊 Loading data for {exp_config['target_name']}...")
    
    df = load_data()
    df = apply_transformation(df)
    df = add_lag_features(df)
    
    print(f"✅ Data loaded: {len(df)} observations")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Get best config from main experiment
    dropout_type, divergence = get_best_config_from_results(config_name)
    hyperparams = get_best_hyperparams_from_results(config_name, dropout_type, divergence)
    
    print(f"\n🔧 Model: {dropout_type} + {divergence}")
    print(f"   Layers: {hyperparams['num_layers']}, Hidden: {hyperparams['hidden_size']}")
    print(f"   Base Rate: {hyperparams['base_rate']:.2f}, Alpha: {hyperparams['alpha']:.1f}")
    
    # Setup TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_folds)
    
    # Prepare data arrays for splitting
    n_samples = len(df)
    X_dummy = np.zeros((n_samples, 1))  # Just for index splitting
    
    # Run each fold
    fold_results = []
    
    print(f"\n{'='*70}")
    print("RUNNING CROSS-VALIDATION FOLDS")
    print(f"{'='*70}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_dummy)):
        train_start = df.iloc[train_idx[0]]['Date'].date()
        train_end = df.iloc[train_idx[-1]]['Date'].date()
        test_start = df.iloc[test_idx[0]]['Date'].date()
        test_end = df.iloc[test_idx[-1]]['Date'].date()
        
        print(f"\n📁 Fold {fold_idx + 1}/{n_folds}")
        print(f"   Train: {train_start} → {train_end} ({len(train_idx)} samples)")
        print(f"   Test:  {test_start} → {test_end} ({len(test_idx)} samples)")
        
        # Prepare fold data
        fold_data = prepare_fold_data(df, train_idx, test_idx)
        
        # Train and evaluate
        metrics = train_and_evaluate_fold(
            fold_data, dropout_type, divergence, hyperparams, 
            epochs=epochs, seed=42 + fold_idx
        )
        
        metrics['fold'] = fold_idx + 1
        metrics['train_period'] = f"{train_start} to {train_end}"
        metrics['test_period'] = f"{test_start} to {test_end}"
        fold_results.append(metrics)
        
        print(f"   Results: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    
    mean_r2 = results_df['r2'].mean()
    std_r2 = results_df['r2'].std()
    mean_rmse = results_df['rmse'].mean()
    std_rmse = results_df['rmse'].std()
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {dropout_type} + {divergence}")
    print(f"Folds: {n_folds}")
    print(f"\nTest R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"{'='*70}")
    
    # Save results
    results_dir = f'results_expanding_cv_{config_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(results_dir, 'fold_results.csv'), index=False)
    
    summary = {
        'config': config_name,
        'dropout_type': dropout_type,
        'divergence': divergence,
        'n_folds': n_folds,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'timestamp': datetime.now().isoformat()
    }
    pd.DataFrame([summary]).to_csv(os.path.join(results_dir, 'cv_summary.csv'), index=False)
    
    print(f"\n📁 Results saved to {results_dir}/")
    
    return results_df, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expanding Window Cross-Validation')
    parser.add_argument('--config', type=str, default='gold_price_2010_2025',
                        help='Experiment config name')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per fold (default: 30)')
    
    args = parser.parse_args()
    
    run_expanding_cv(args.config, args.n_folds, args.epochs)
