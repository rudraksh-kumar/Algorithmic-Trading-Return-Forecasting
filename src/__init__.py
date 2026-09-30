# src package initializer
# Stock Market Volatility Forecasting & Algorithmic Trading
# ---------------------------------------------------------
# This package contains the modular components of the ML pipeline:
#   data_loader   - Data ingestion (yfinance + synthetic fallback)
#   preprocessing - Cleaning, returns, volatility, chronological splits
#   features      - Technical indicator feature engineering
#   models        - Classical ML baselines (Linear, RF, XGBoost)
#   lstm_model    - PyTorch Dataset, DataLoader, LSTM architecture
#   training      - Training loops, early stopping, hyperparameter experiments
#   backtesting   - Vectorized backtesting engine with transaction costs
#   evaluation    - Statistical & financial performance metrics
