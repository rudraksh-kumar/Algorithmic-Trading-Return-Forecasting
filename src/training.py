"""
training.py - PyTorch Training Pipeline & Hyperparameter Experiments
======================================================================
This module handles:
  1. Training loop for the LSTM/NN model with mini-batch gradient descent.
  2. Validation loss monitoring and Early Stopping.
  3. Learning rate scheduling.
  4. Hyperparameter experiment runner (window size, hidden size, dropout, lr).
  5. Saving and loading model checkpoints.

Key ML Concepts Explained
--------------------------
Mini-batch Gradient Descent
  Instead of computing gradients on the entire dataset (batch GD) or one
  sample at a time (SGD), mini-batch GD computes gradients on a small
  random subset (a "batch").  This gives a good balance between:
  - Gradient stability (vs. pure SGD)
  - Computational efficiency (vs. full batch GD)
  - Regularisation effect from noisy gradient estimates

Early Stopping
  Monitors validation loss every epoch.  If the validation loss does not
  improve for `patience` consecutive epochs, training is stopped early.
  This prevents overfitting: the model has started memorising training
  data rather than learning generalisable patterns.

Learning Rate Scheduling (ReduceLROnPlateau)
  When validation loss stagnates, the learning rate is reduced by a
  factor (e.g., 0.5).  This allows the optimiser to take smaller steps
  and converge to a better minimum.

Loss Function
  MSELoss (Mean Squared Error): L = (1/n) * sum((y_pred - y_true)^2)
  Penalises large prediction errors more than small ones.
  HuberLoss is also available as a robust alternative (less sensitive
  to outliers than MSE).
"""

import os
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lstm_model import LSTMModel, FeedforwardNN, build_lstm, make_dataloader

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


# ===================================================================
# 1. EARLY STOPPING CALLBACK
# ===================================================================
class EarlyStopping:
    """
    Monitors validation loss and stops training when it stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement before stopping.
    min_delta : float
        Minimum change in val loss to qualify as an improvement.
    verbose : bool
        Print message when early stopping triggers.

    Attributes
    ----------
    best_loss : float
        Best validation loss seen so far.
    counter : int
        Number of epochs without improvement.
    best_model_state : dict
        Deep copy of the model state at best validation loss.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-6, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] Stopping. Best val loss: {self.best_loss:.6f}")

    def restore_best(self, model: nn.Module):
        """Restore model weights to the best checkpoint."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


# ===================================================================
# 2. SINGLE TRAINING LOOP
# ===================================================================
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training DataLoader (one epoch).

    Process per mini-batch:
    1. Move data to device (CPU / GPU).
    2. Zero out accumulated gradients (optimizer.zero_grad).
    3. Forward pass: compute predictions.
    4. Compute loss between predictions and targets.
    5. Backward pass: compute gradients (loss.backward).
    6. Gradient clipping (prevents exploding gradients in RNNs).
    7. Parameter update (optimizer.step).

    Returns
    -------
    float : Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)                    # (batch, 1)
        loss = criterion(predictions, y_batch)          # scalar
        loss.backward()

        # Gradient clipping - important for RNNs to prevent exploding gradients
        # Clips the norm of all gradients to max_norm
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate the model on a DataLoader (validation or test).

    Returns
    -------
    avg_loss : float
    preds    : np.ndarray of shape (N,)
    targets  : np.ndarray of shape (N,)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    avg_loss = total_loss / len(loader)
    return avg_loss, preds, targets


# ===================================================================
# 3. FULL TRAINING PIPELINE
# ===================================================================
def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    weight_decay: float = 1e-4,
    use_huber: bool = False,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    """
    Full training pipeline with early stopping and LR scheduling.

    Parameters
    ----------
    model        : PyTorch model to train.
    train_loader : DataLoader for training data.
    val_loader   : DataLoader for validation data.
    epochs       : Maximum number of training epochs.
    lr           : Initial learning rate for Adam optimiser.
    patience     : Early stopping patience (epochs).
    weight_decay : L2 regularisation coefficient for Adam.
                   This penalises large weights: Loss += weight_decay * ||w||²
    use_huber    : If True, use HuberLoss instead of MSELoss.
    device       : torch.device (auto-detected if None).
    verbose      : Print progress every 10 epochs.

    Returns
    -------
    dict with keys:
        'train_losses'  : list of per-epoch training loss
        'val_losses'    : list of per-epoch validation loss
        'best_val_loss' : float
        'epochs_trained': int
        'model'         : model with best weights restored
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device}")
    model = model.to(device)

    # Loss function
    criterion = nn.HuberLoss() if use_huber else nn.MSELoss()

    # Optimiser: Adam with weight decay (L2 regularisation)
    # weight_decay adds L2 penalty to help prevent overfitting
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler: reduce LR when val loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                   patience=5)

    early_stop = EarlyStopping(patience=patience, verbose=verbose)

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stop(val_loss, model)

        if verbose and (epoch % 10 == 0 or epoch == 1 or early_stop.stop):
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch [{epoch:4d}/{epochs}]  "
                  f"Train Loss: {train_loss:.6f}  |  "
                  f"Val Loss: {val_loss:.6f}  |  "
                  f"LR: {current_lr:.2e}  |  "
                  f"Elapsed: {elapsed:.1f}s")

        if early_stop.stop:
            print(f"[Training] Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    model = early_stop.restore_best(model)
    model.eval()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": early_stop.best_loss,
        "epochs_trained": len(train_losses),
        "model": model,
    }


# ===================================================================
# 4. GET PREDICTIONS ON TEST SET
# ===================================================================
def get_predictions(
    model: nn.Module,
    loader,
    device: torch.device = None,
) -> tuple:
    """
    Run inference on a DataLoader and return predictions and targets.

    Parameters
    ----------
    model  : Trained PyTorch model.
    loader : DataLoader (test or validation).
    device : torch.device.

    Returns
    -------
    preds   : np.ndarray of shape (N,)
    targets : np.ndarray of shape (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    _, preds, targets = evaluate_epoch(model, loader, criterion, device)
    return preds, targets


# ===================================================================
# 5. HYPERPARAMETER EXPERIMENT RUNNER
# ===================================================================
def run_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    experiment_grid: list = None,
    epochs: int = 60,
    patience: int = 10,
) -> pd.DataFrame:
    """
    Run a grid of LSTM hyperparameter experiments and return results.

    Experiments vary:
    - seq_len    : lookback window (10, 30, 60 days)
    - hidden_size: LSTM hidden units (32, 64, 128)
    - dropout    : regularisation rate (0.1, 0.3)
    - lr         : learning rate (1e-3, 5e-4)
    - batch_size : mini-batch size (32, 64)

    Parameters
    ----------
    X_train, y_train : Training arrays.
    X_val, y_val     : Validation arrays.
    input_size       : Number of input features.
    experiment_grid  : List of config dicts. Uses default grid if None.
    epochs           : Max epochs per experiment.
    patience         : Early stopping patience per experiment.

    Returns
    -------
    pd.DataFrame with one row per experiment configuration and its val_loss.
    """
    if experiment_grid is None:
        experiment_grid = [
            # Baseline
            {"seq_len": 30, "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
            # Vary window size
            {"seq_len": 10, "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
            {"seq_len": 60, "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
            # Vary hidden size
            {"seq_len": 30, "hidden_size": 32,  "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
            {"seq_len": 30, "hidden_size": 128, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 32},
            # Vary dropout
            {"seq_len": 30, "hidden_size": 64,  "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "batch_size": 32},
            {"seq_len": 30, "hidden_size": 64,  "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "batch_size": 32},
            # Vary learning rate
            {"seq_len": 30, "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "lr": 5e-4, "batch_size": 32},
            # Vary batch size
            {"seq_len": 30, "hidden_size": 64,  "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64},
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []

    for i, cfg in enumerate(experiment_grid, 1):
        print(f"\n{'='*65}")
        print(f"  Experiment {i}/{len(experiment_grid)}: {cfg}")
        print(f"{'='*65}")

        seq_len = cfg["seq_len"]

        # Build dataloaders for this window size
        train_loader = make_dataloader(X_train, y_train, seq_len=seq_len,
                                       batch_size=cfg["batch_size"], shuffle=False)
        val_loader = make_dataloader(X_val, y_val, seq_len=seq_len,
                                     batch_size=cfg["batch_size"], shuffle=False)

        model = build_lstm(
            input_size=input_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )

        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=cfg["lr"],
            patience=patience,
            weight_decay=1e-4,
            device=device,
            verbose=True,
        )

        record = {**cfg, "best_val_loss": result["best_val_loss"],
                  "epochs_trained": result["epochs_trained"]}
        records.append(record)
        print(f"  -> Best Val Loss: {result['best_val_loss']:.6f}  "
              f"after {result['epochs_trained']} epochs")

    results_df = pd.DataFrame(records).sort_values("best_val_loss")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "experiment_summary.csv"), index=False)
    print(f"\n[Experiments] Results saved to results/experiment_summary.csv")
    print("\nExperiment Summary (sorted by val loss):")
    print(results_df.to_string(index=False))
    return results_df


# ===================================================================
# 6. MODEL SAVE / LOAD
# ===================================================================
def save_model(model: nn.Module, path: str):
    """Save model state dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Training] Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device = None):
    """Load model state dict from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[Training] Model loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    T, F = 600, 15
    X = np.random.randn(T, F).astype(np.float32)
    y = np.random.randn(T).astype(np.float32)

    split = int(0.8 * T)
    train_loader = make_dataloader(X[:split], y[:split], seq_len=30, batch_size=32)
    val_loader = make_dataloader(X[split:], y[split:], seq_len=30, batch_size=32)

    model = build_lstm(input_size=F, hidden_size=64, num_layers=2, dropout=0.2)
    result = train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, patience=5)
    print(f"\nTraining complete. Epochs: {result['epochs_trained']}  "
          f"Best val loss: {result['best_val_loss']:.6f}")
