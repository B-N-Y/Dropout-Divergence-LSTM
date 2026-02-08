# Dropout-Divergence-LSTM

This repository contains the official PyTorch implementation of the paper **"Regulating the Unregulated: A Divergence-Guided Dropout Framework for Non-Stationary Financial Time Series"** (Neural Computing and Applications).

We introduce a novel regularization framework where dropout rates are dynamically adjusted based on the distributional divergence between stochastic forward passes. This approach outperforms standard LSTM baselines on non-stationary gold price data without requiring stationarity transformations.

## 📂 Repository Structure

```
├── src/                # Source code (models, data loader, divergence measures)
├── main.py             # Main training and hyperparameter search script
├── run_expanding_cv.py # Expanding Window Cross-Validation script (Robustness)
├── download_data.py    # Script to download data from Yahoo Finance API
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
This repository does **not** include static CSV files. Instead, use the provided script to download the latest market data directly from Yahoo Finance:

```bash
python download_data.py
```
This will create `data/gold_data.csv` containing:
- **Gold Price (GLD)**
- **Macro Features:** VIX, DXY, TNX
- **Technical Features:** 20-day Moving Average

### 3. Run Experiments

#### A. Gold Price Prediction (Main Model)
To run the hyperparameter optimization using Optuna on Gold Price (2010-2025):
```bash
python main.py --config gold_price_2010_2025
```

#### B. Gold Returns Prediction (Market Efficiency Check)
To run the same model on Gold Returns (Stationary) for comparison:
```bash
python main.py --config gold_returns_2010_2025
```

#### C. Subsample Analysis (Regime Changes)
To analyze model performance across different market regimes (as in Section 6.2):
```bash
# Period 1: 2010-2015
python main.py --config gold_price_2010_2015

# Period 2: 2015-2020
python main.py --config gold_price_2015_2020

# Period 3: 2020-2025
python main.py --config gold_price_2020_2025
```

#### C. Robustness Check (Expanding Window CV)
To run the 5-fold expanding window cross-validation as detailed in Section 6.3 of the manuscript:
```bash
python run_expanding_cv.py --config gold_price_2010_2025 --n-folds 5
```

#### D. Seed Sensitivity (Stability Check)
To test the model's stability across different random seeds (as in Section 6.4):
```bash
python main.py --config gold_price_2010_2025 --seeds 42,123,456,789,101
```

## 📊 Configuration
The `src/config.py` file controls all experiment parameters. Key configurations include:
- **Date Ranges:** 2010-2025 (Full). Subperiods: 2010-2015, 2015-2020, 2020-2025.
- **Model Parameters:** LSTM layers, hidden size, dropout types.
- **Divergence Measures:** TV, Hellinger, KL, Wasserstein, etc.

## 📝 Citation
If you use this code, please cite our paper:
> [Citation specific to Neural Computing and Applications will be added here upon acceptance]

## 📜 License
MIT License
