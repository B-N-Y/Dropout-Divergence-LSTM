"""
models.py
---------
LSTM architectures with various dropout strategies and divergence-based
regularization for time series forecasting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .divergences import compute_divergence


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Dropout Module
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveDropout(nn.Module):
    """
    Dropout module where the rate adapts based on divergence between
    original and dropped representations.
    
    Tracks dropout rate history for visualization (Reviewer Comment 1.4).
    """
    def __init__(self, base_rate: float, alpha: float, divergence_method: str,
                 eps: float = 1e-10, smoothing_beta: float = 0.0):
        super().__init__()
        self.base_rate = base_rate
        self.alpha = alpha
        self.divergence_method = divergence_method
        self.eps = eps
        self.smoothing_beta = smoothing_beta
        self.current_rate = base_rate
        # Track rate history for visualization
        self.rate_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First dropout pass at base_rate
        dropped = F.dropout(x, p=self.base_rate, training=True)

        # Compute probability distributions
        orig_prob = F.softmax(x, dim=-1)
        drop_prob = F.softmax(dropped, dim=-1)

        # Compute divergence
        div = compute_divergence(orig_prob, drop_prob, self.divergence_method)
        div_mean = div.mean()

        # Update rate with optional EMA smoothing
        raw_rate = float(torch.clamp(self.base_rate + self.alpha * div_mean, 0.0, 0.5))
        smoothed = self.smoothing_beta * self.current_rate + (1 - self.smoothing_beta) * raw_rate
        self.current_rate = smoothed
        
        # Record rate for visualization
        self.rate_history.append(smoothed)

        return F.dropout(x, p=smoothed, training=True)

    def reset_rate(self):
        """Reset rate at the start of each epoch."""
        self.current_rate = self.base_rate

    def get_rate_history(self):
        """Return the rate history for plotting."""
        return self.rate_history.copy()

    def clear_rate_history(self):
        """Clear the rate history."""
        self.rate_history = []


# ─────────────────────────────────────────────────────────────────────────────
# Main LSTM Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiFeatureLSTM(nn.Module):
    """
    Multi-input LSTM for time series forecasting with various dropout strategies.
    
    Dropout Types:
        - 'none': No dropout (baseline)
        - 'fixed': Standard dropout with fixed rate (0.5)
        - 'monte_carlo': MC Dropout for uncertainty estimation
        - 'dynamic': Global dropout rate updated based on divergence
        - 'adaptive': Per-layer adaptive dropout based on divergence
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 extra_features: int, dropout_type: str = "none",
                 base_rate: float = 0.1, alpha: float = 1.0,
                 divergence_method: str = "js", mc_dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_type = dropout_type
        self.base_rate = base_rate if dropout_type != "fixed" else 0.5
        self.mc_dropout = mc_dropout_rate
        self.divergence_method = divergence_method
        self.alpha = alpha

        # LSTM with inter-layer dropout
        lstm_dropout = self.base_rate if dropout_type != 'none' else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)

        # Readout heads for divergence computation
        self.readout_orig = nn.Linear(hidden_size, hidden_size)
        self.readout_drop = nn.Linear(hidden_size, hidden_size)

        # Dropout modules for dynamic/adaptive
        if dropout_type == "adaptive":
            self.adaptive = AdaptiveDropout(base_rate, alpha, divergence_method)
        elif dropout_type == "dynamic":
            self.dynamic = nn.Dropout(base_rate)

        # Final prediction head
        self.fc = nn.Linear(hidden_size + extra_features, 1)

    def forward(self, x_seq: torch.Tensor, x_extra: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x_seq: Sequential input (batch, seq_len, input_size)
            x_extra: Extra features (batch, extra_features)
        
        Returns:
            out: Predictions (batch, 1)
            div: Divergence values (batch,)
        """
        # Input-level dropout
        if self.training:
            if self.dropout_type == "fixed":
                x_seq = F.dropout(x_seq, p=self.base_rate, training=True)
            elif self.dropout_type == "monte_carlo":
                x_seq = F.dropout(x_seq, p=self.mc_dropout, training=True)
            elif self.dropout_type == "dynamic":
                x_seq = self.dynamic(x_seq)
            elif self.dropout_type == "adaptive":
                x_seq = self.adaptive(x_seq)

        # LSTM forward
        lstm_out, _ = self.lstm(x_seq)
        h_last = lstm_out[:, -1, :]

        # Hidden-level dropout
        if self.training:
            if self.dropout_type == "fixed":
                dropped = F.dropout(h_last, p=self.base_rate, training=True)
            elif self.dropout_type == "monte_carlo":
                dropped = F.dropout(h_last, p=self.mc_dropout, training=True)
            elif self.dropout_type == "dynamic":
                dropped = self.dynamic(h_last)
            elif self.dropout_type == "adaptive":
                dropped = self.adaptive(h_last)
            else:
                dropped = h_last
        else:
            # Only MC dropout stays on at test time
            if self.dropout_type == "monte_carlo":
                dropped = F.dropout(h_last, p=self.mc_dropout, training=True)
            else:
                dropped = h_last

        # Compute divergence (skip for 'none')
        if self.dropout_type == 'none':
            div = torch.zeros(x_seq.size(0), device=x_seq.device)
        else:
            orig_proj = self.readout_orig(h_last)
            drop_proj = self.readout_drop(dropped)
            orig_prob = F.softmax(orig_proj, dim=-1)
            drop_prob = F.softmax(drop_proj, dim=-1)
            div = compute_divergence(orig_prob, drop_prob, self.divergence_method)

        # Prediction
        out = self.fc(torch.cat([dropped, x_extra], dim=1))
        return out, div

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor,
                     div: torch.Tensor) -> torch.Tensor:
        """
        Compute loss: MSE + α * mean(divergence) for regularized strategies.
        """
        mse = F.mse_loss(preds, target)
        if self.dropout_type in ['fixed', 'dynamic', 'adaptive']:
            return mse + self.alpha * div.mean()
        return mse
