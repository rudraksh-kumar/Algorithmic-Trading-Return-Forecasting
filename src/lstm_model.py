"""
lstm_model.py - PyTorch LSTM Model & Dataset
=============================================
This module implements:
  1. A sliding-window PyTorch Dataset for time-series data.
  2. A configurable LSTM neural network using PyTorch.

Why LSTM for Financial Time Series?
-------------------------------------
Standard feedforward networks treat each input independently. In financial
markets, the ordering and recency of data matters - yesterday's return
influences today's behaviour. Long Short-Term Memory (LSTM) networks are
a type of Recurrent Neural Network (RNN) that maintain a hidden state
across time steps, allowing them to learn temporal dependencies.

LSTM Key Equations (per time step t):
  Forget gate : f_t = sigma(W_f · [h_{t-1}, x_t] + b_f)
  Input gate  : i_t = sigma(W_i · [h_{t-1}, x_t] + b_i)
  Cell update : C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
  Cell state  : C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
  Output gate : o_t = sigma(W_o · [h_{t-1}, x_t] + b_o)
  Hidden state: h_t = o_t ⊙ tanh(C_t)

The cell state C_t acts as the "memory" that the network can read from,
write to, or erase via the gating mechanisms.

Input/Output Shape
------------------
  Input  : (batch_size, sequence_length, num_features)   e.g. (32, 30, 20)
  Output : (batch_size, 1)                               next-day prediction
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===================================================================
# 1. PYTORCH DATASET - Sliding Window
# ===================================================================
class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset that creates fixed-length sliding windows over a
    multivariate time series.

    Given a feature matrix X of shape (T, num_features) and a target
    vector y of shape (T,), this dataset produces samples:
        X[i : i+seq_len]  ->  y[i+seq_len-1]

    where each X sample contains seq_len consecutive feature rows and
    the corresponding target is the value immediately after that window.

    Parameters
    ----------
    X : np.ndarray, shape (T, num_features)
        Normalised feature matrix (already scaled).
    y : np.ndarray, shape (T,)
        Target vector.
    seq_len : int
        Number of consecutive trading days in each input window.

    Example
    -------
    With T=1000, seq_len=30:
        Sample 0: X[0:30]  -> y[29]
        Sample 1: X[1:31]  -> y[30]
        ...
        Sample 970: X[970:1000] -> y[999]
    Total samples = T - seq_len = 970
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30):
        assert len(X) == len(y), "X and y must have the same length."
        assert len(X) > seq_len, f"Data length ({len(X)}) must exceed seq_len ({seq_len})."
        self.seq_len = seq_len
        # Convert to float32 tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        x_window : torch.Tensor, shape (seq_len, num_features)
        y_target : torch.Tensor, shape (1,)
        """
        x_window = self.X[idx: idx + self.seq_len]       # (seq_len, features)
        y_target = self.y[idx + self.seq_len - 1].unsqueeze(0)  # (1,)
        return x_window, y_target


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 30,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """
    Convenience function to create a DataLoader from arrays.

    NOTE: shuffle=False for time-series to preserve temporal order.

    Parameters
    ----------
    X : np.ndarray of shape (T, num_features)
    y : np.ndarray of shape (T,)
    seq_len : int
        Window size (number of past trading days).
    batch_size : int
        Mini-batch size for gradient updates.
    shuffle : bool
        Should be False for time-series data.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = TimeSeriesDataset(X, y, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ===================================================================
# 2. PYTORCH LSTM MODEL
# ===================================================================
class LSTMModel(nn.Module):
    """
    Multi-layer LSTM for next-day return / volatility prediction.

    Architecture
    ------------
    Input  -> LSTM layers (with Dropout between layers)
           -> Final hidden state h_T from last layer
           -> Fully Connected (Linear) layer
           -> Scalar output (regression)

    Parameters
    ----------
    input_size : int
        Number of features per time step (num_features).
    hidden_size : int
        Number of units in each LSTM hidden layer.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers (0 = no dropout).
        Dropout is a regularisation technique: during training, random
        units are zeroed out with probability `dropout`, forcing the
        network to learn redundant representations and reducing overfitting.
    output_size : int
        Dimensionality of the output (default 1 for scalar regression).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer(s)
        # batch_first=True means input/output shape is (batch, seq, features)
        # dropout is applied between stacked LSTM layers (not after the last)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Additional dropout before the output layer (regularisation)
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, seq_len, input_size)

        Returns
        -------
        out : torch.Tensor, shape (batch_size, output_size)
        """
        batch_size = x.size(0)

        # Initialise hidden and cell states to zeros on the same device as input
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # lstm_out: (batch_size, seq_len, hidden_size)
        # We only need the last time step's hidden state for prediction
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take only the last time step output -> (batch_size, hidden_size)
        last_hidden = lstm_out[:, -1, :]

        # Apply dropout for regularisation
        last_hidden = self.dropout(last_hidden)

        # Project to output dimension -> (batch_size, output_size)
        out = self.fc(last_hidden)
        return out


class RNNModel(nn.Module):
    """
    Multi-layer Vanilla RNN for next-day return / volatility prediction.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, output_size: int = 1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        last_hidden = self.dropout(out[:, -1, :])
        return self.fc(last_hidden)


class GRUModel(nn.Module):
    """
    Multi-layer GRU for next-day return / volatility prediction.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, output_size: int = 1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        last_hidden = self.dropout(out[:, -1, :])
        return self.fc(last_hidden)


# ===================================================================
# 3. SIMPLE FEEDFORWARD NN (for comparison)
# ===================================================================
class FeedforwardNN(nn.Module):
    """
    Simple feedforward (MLP) neural network for comparison with LSTM.

    This flattens the sequence window and passes it through dense layers.
    Unlike LSTM, it has no notion of temporal order within the window.

    Architecture: Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear -> Output
    """

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.2):
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, seq_len, num_features) OR (batch_size, flat_features)
        """
        if x.dim() == 3:
            # Flatten the sequence dimension for feedforward network
            x = x.reshape(x.size(0), -1)
        return self.net(x)


# ===================================================================
# 4. MODEL FACTORY
# ===================================================================
def build_lstm(input_size: int, hidden_size: int = 64, num_layers: int = 2,
               dropout: float = 0.2) -> LSTMModel:
    """Build and return an LSTMModel instance."""
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[LSTM] Model built  |  input_size={input_size}  hidden_size={hidden_size}  "
          f"num_layers={num_layers}  dropout={dropout}  params={total_params:,}")
    return model


def build_rnn(input_size: int, hidden_size: int = 64, num_layers: int = 2,
              dropout: float = 0.2) -> RNNModel:
    """Build and return an RNNModel instance."""
    model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                     num_layers=num_layers, dropout=dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[RNN] Model built   |  input_size={input_size}  hidden_size={hidden_size}  "
          f"num_layers={num_layers}  dropout={dropout}  params={total_params:,}")
    return model


def build_gru(input_size: int, hidden_size: int = 64, num_layers: int = 2,
              dropout: float = 0.2) -> GRUModel:
    """Build and return a GRUModel instance."""
    model = GRUModel(input_size=input_size, hidden_size=hidden_size,
                     num_layers=num_layers, dropout=dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GRU] Model built   |  input_size={input_size}  hidden_size={hidden_size}  "
          f"num_layers={num_layers}  dropout={dropout}  params={total_params:,}")
    return model


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate batch: 32 samples, each 30 time steps, each with 20 features
    batch_size, seq_len, num_features = 32, 30, 20
    dummy_X = np.random.randn(500, num_features).astype(np.float32)
    dummy_y = np.random.randn(500).astype(np.float32)

    loader = make_dataloader(dummy_X, dummy_y, seq_len=seq_len, batch_size=batch_size)
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")

    # Test forward pass
    model = build_lstm(input_size=num_features, hidden_size=64, num_layers=2, dropout=0.2)
    for x_batch, y_batch in loader:
        print(f"\nInput shape  : {x_batch.shape}")   # (32, 30, 20)
        print(f"Target shape : {y_batch.shape}")      # (32, 1)
        out = model(x_batch)
        print(f"Output shape : {out.shape}")           # (32, 1)
        break

    print("\nLSTM Model Architecture:")
    print(model)
