#!/usr/bin/env python
"""
main.py
-------
Main entry point for running LSTM-Dropout-Divergence experiments.
Supports multi-seed experiments and regime analysis.

Variable configuration is done in src/config.py - see VARIABLE_CONFIG.

Usage:
    python main.py                           # Quick run with default settings
    python main.py --seeds 42,123,456        # Multi-seed experiment
    python main.py --regime --vix-threshold 25  # With regime analysis
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch.optim as optim

from src.config import (
    set_seeds, DEVICE, START_DATE, END_DATE,
    RETRAIN_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    VARIABLE_CONFIG, TARGET_COL, N_LAGS,
    BEST_MODEL_CONFIG, RESULTS_DIR
)
from src.data_loader import (
    load_data, add_lag_features, prepare_train_val_test,
    compute_descriptive_stats, add_regime_labels
)
from src.training import train_model, create_model
from src.preprocessing import adf_test
from src.evaluation import (
    evaluate_model, compute_metrics, random_walk_baseline,
    compute_implied_return_metrics, directional_accuracy,
    aggregate_multi_seed_results, format_multi_seed_table, evaluate_by_regime
)


def parse_seeds(seeds_str: str) -> list:
    """Parse comma-separated seed string into list of ints."""
    return [int(s.strip()) for s in seeds_str.split(',')]


def run_single_experiment(df, data, seed, results_dir):
    """Run a single training experiment with given seed."""
    set_seeds(seed)
    
    extra_cols = data['extra_cols']
    
    # Create model
    model = create_model(
        dropout_type=BEST_MODEL_CONFIG['dropout_type'],
        divergence_method=BEST_MODEL_CONFIG['divergence_method'],
        num_layers=BEST_MODEL_CONFIG['num_layers'],
        hidden_size=BEST_MODEL_CONFIG['hidden_size'],
        extra_features=len(extra_cols),
        base_rate=BEST_MODEL_CONFIG['base_rate'],
        alpha=BEST_MODEL_CONFIG['alpha'],
        device=DEVICE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train on Train+Val combined
    X_re_seq = np.vstack([data['X_train_seq'], data['X_val_seq']])
    X_re_extra = np.vstack([data['X_train_extra'], data['X_val_extra']])
    y_re = np.concatenate([data['y_train_s'], data['y_val_s']])
    
    history = train_model(
        model, optimizer,
        X_re_seq, X_re_extra, y_re,
        epochs=RETRAIN_EPOCHS, batch_size=BATCH_SIZE,
        dynamic_update=True, device=DEVICE
    )
    
    # Evaluate on test set
    y_test_true = data['scaler_y'].inverse_transform(data['y_test_s'].reshape(-1, 1)).ravel()
    prev_prices_test = df[TARGET_COL].values[data['test_idx'] - 1]
    
    preds = evaluate_model(model, data['X_test_seq'], data['X_test_extra'],
                           scaler_y=data['scaler_y'], device=DEVICE)
    
    metrics = compute_metrics(y_test_true, preds)
    ret_metrics = compute_implied_return_metrics(y_test_true, preds, prev_prices_test)
    dir_acc = directional_accuracy(y_test_true, preds, prev_prices_test)
    
    result = {
        **metrics,
        'R2_Returns': ret_metrics['R2_Returns'],
        'Dir_Accuracy': dir_acc,
        'seed': seed
    }
    
    return result, model, preds, y_test_true


def main():
    parser = argparse.ArgumentParser(
        description="LSTM Dropout-Divergence Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Quick run
  python main.py --seeds 42,123,456     # Multi-seed
  python main.py --regime               # With regime analysis
        """
    )
    parser.add_argument('--config', type=str, default='gold_price_2010_2025',
                        help="Experiment config name (e.g., gold_price_2010_2015)")
    parser.add_argument('--seeds', type=str, default='42',
                        help="Comma-separated list of seeds (e.g., '42,123,456')")
    parser.add_argument('--regime', action='store_true',
                        help="Enable regime-based analysis (VIX, COVID)")
    parser.add_argument('--vix-threshold', type=float, default=25.0,
                        help="VIX threshold for high/low volatility (default: 25)")
    args = parser.parse_args()
    
    # Update configuration based on CLI arg
    import src.config as cfg
    if args.config in cfg.EXPERIMENTS:
        cfg.CURRENT_EXPERIMENT = args.config
        # Reload variable config for the selected experiment
        exp = cfg.get_experiment_config(args.config)
        cfg.VARIABLE_CONFIG = {
            'target_ticker': exp['target_ticker'],
            'target_name': exp['target_name'],
            'input_tickers': exp.get('input_tickers', {}),
            'extra_feature_cols': exp['extra_feature_cols'],
        }
        cfg.PREPROCESSING_CONFIG['transformation'] = exp.get('transformation', 'price')
        cfg.PREPROCESSING_CONFIG['transform_inputs'] = exp.get('transform_inputs', False)
    else:
        print(f"⚠️ Warning: Config '{args.config}' not found. Using default: {cfg.CURRENT_EXPERIMENT}")
    
    seeds = parse_seeds(args.seeds)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. Load and prepare data
    # ─────────────────────────────────────────────────────────────────────
    print(f"=" * 60)
    print(f"Target: {VARIABLE_CONFIG['target_name']} ({VARIABLE_CONFIG['target_ticker']})")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Seeds: {seeds}")
    print(f"=" * 60)
    
    df = load_data()
    df = add_lag_features(df)
    
    # Add regime labels if requested
    if args.regime:
        df = add_regime_labels(df, vix_threshold=args.vix_threshold)
        print(f"📈 Regime labels added (VIX threshold: {args.vix_threshold})")
    
    # Descriptive statistics
    print("\n📊 Descriptive Statistics:")
    stats = compute_descriptive_stats(df)
    print(stats.to_string(index=False))
    
    # ADF test
    print("\n📈 Stationarity Tests:")
    adf_price = adf_test(df['price'].values)
    print(f"   Price: ADF={adf_price['statistic']:.4f}, p={adf_price['pvalue']:.4f}, stationary={adf_price['is_stationary']}")
    
    returns = df['price'].pct_change().dropna()
    adf_returns = adf_test(returns.values)
    print(f"   Returns: ADF={adf_returns['statistic']:.4f}, p={adf_returns['pvalue']:.4f}, stationary={adf_returns['is_stationary']}")
    
    # Prepare data splits
    data = prepare_train_val_test(df)
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. Random Walk Baseline
    # ─────────────────────────────────────────────────────────────────────
    print("\n🚶 Random Walk Baseline:")
    y_test_true = data['scaler_y'].inverse_transform(data['y_test_s'].reshape(-1, 1)).ravel()
    prev_prices_test = df[TARGET_COL].values[data['test_idx'] - 1]
    rw_metrics = random_walk_baseline(y_test_true, prev_prices_test)
    print(f"   RMSE: {rw_metrics['RMSE']:.4f}, R²: {rw_metrics['R2']:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. Multi-seed experiments
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n🧠 Training with {len(seeds)} seed(s)...")
    all_results = []
    last_model = None
    last_preds = None
    last_y_test = None
    
    for seed in seeds:
        print(f"   Seed {seed}...", end=" ", flush=True)
        result, model, preds, y_test = run_single_experiment(df, data, seed, RESULTS_DIR)
        all_results.append(result)
        last_model = model
        last_preds = preds
        last_y_test = y_test
        print(f"R²={result['R2']:.4f}, R²_Returns={result['R2_Returns']:.4f}")
    
    # Aggregate results
    print("\n" + "=" * 60)
    if len(seeds) > 1:
        aggregated = aggregate_multi_seed_results(all_results)
        print("📊 Multi-Seed Results (Mean ± Std):")
        table = format_multi_seed_table(aggregated)
        print(table.to_string(index=False))
    else:
        result = all_results[0]
        print("📊 Test Results:")
        print(f"   RMSE: {result['RMSE']:.4f}")
        print(f"   MAE:  {result['MAE']:.4f}")
        print(f"   R²:   {result['R2']:.4f}")
        print(f"   R² (Implied Returns): {result['R2_Returns']:.4f}")
        print(f"   Directional Accuracy: {result['Dir_Accuracy']:.2%}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. Regime Analysis (if enabled)
    # ─────────────────────────────────────────────────────────────────────
    if args.regime and last_preds is not None:
        print("\n📊 Regime-Based Analysis:")
        
        for regime_col in ['regime_vol', 'regime_covid']:
            if regime_col in df.columns:
                regime_labels = df.iloc[data['test_idx']][regime_col].values
                regime_results = evaluate_by_regime(last_y_test, last_preds, regime_labels)
                print(f"\n   {regime_col}:")
                print(regime_results.to_string(index=False))
    
    # ─────────────────────────────────────────────────────────────────────
    # 5. Save results
    # ─────────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df['Target'] = VARIABLE_CONFIG['target_name']
    results_df['RW_RMSE'] = rw_metrics['RMSE']
    results_df['RW_R2'] = rw_metrics['R2']
    
    output_file = f"{RESULTS_DIR}/results_seeds_{'-'.join(map(str, seeds))}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()
