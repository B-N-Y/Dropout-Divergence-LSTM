"""
training.py
-----------
Training loop and Optuna hyperparameter optimization utilities.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .models import MultiFeatureLSTM


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model: MultiFeatureLSTM, optimizer: optim.Optimizer,
                X_train_seq: np.ndarray, X_train_extra: np.ndarray, y_train: np.ndarray,
                X_val_seq: np.ndarray = None, X_val_extra: np.ndarray = None, y_val: np.ndarray = None,
                epochs: int = 50, batch_size: int = 16, clip_value: float = 5.0,
                dynamic_update: bool = False, device: str = 'cpu') -> dict:
    """
    Train the LSTM model with optional validation tracking.
    
    Args:
        model: MultiFeatureLSTM instance
        optimizer: PyTorch optimizer
        X_train_seq, X_train_extra, y_train: Training data
        X_val_seq, X_val_extra, y_val: Validation data (optional)
        epochs: Number of training epochs
        batch_size: Mini-batch size
        clip_value: Gradient clipping threshold
        dynamic_update: Whether to update dynamic dropout rate
        device: 'cpu' or 'cuda'
    
    Returns:
        History dict with train_loss, val_loss, train_div per epoch
    """
    # Create DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(X_train_extra, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    history = {'train_loss': [], 'val_loss': [], 'train_div': []}

    for epoch in range(epochs):
        model.train()
        
        # Reset adaptive rate at epoch start
        if model.dropout_type == 'adaptive':
            model.adaptive.reset_rate()

        epoch_losses = []
        epoch_divs = []

        for seq, extra, tar in train_loader:
            seq, extra, tar = seq.to(device), extra.to(device), tar.to(device)
            optimizer.zero_grad()
            out, div = model(seq, extra)

            # Dynamic rate update
            if dynamic_update and model.dropout_type == 'dynamic':
                new_p = float(torch.clamp(model.base_rate + model.alpha * div.mean(), 0.0, 0.5))
                model.dynamic.p = new_p

            loss = model.compute_loss(out, tar, div)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_divs.append(div.mean().item())

        scheduler.step()
        history['train_loss'].append(float(np.mean(epoch_losses)))
        history['train_div'].append(float(np.mean(epoch_divs)))

        # Validation loss
        if X_val_seq is not None:
            model.eval()
            with torch.no_grad():
                seq_v = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
                extra_v = torch.tensor(X_val_extra, dtype=torch.float32).to(device)
                y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
                out_v, _ = model(seq_v, extra_v)
                val_loss = F.mse_loss(out_v, y_v).item()
            history['val_loss'].append(val_loss)
        else:
            history['val_loss'].append(history['train_loss'][-1])

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_model(dropout_type: str, divergence_method: str,
                 num_layers: int, hidden_size: int,
                 extra_features: int, base_rate: float = 0.1, alpha: float = 1.0,
                 device: str = 'cpu') -> MultiFeatureLSTM:
    """
    Factory function to create a configured LSTM model.
    """
    model = MultiFeatureLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        extra_features=extra_features,
        dropout_type=dropout_type,
        base_rate=base_rate,
        alpha=alpha,
        divergence_method=divergence_method
    )
    return model.to(device)
