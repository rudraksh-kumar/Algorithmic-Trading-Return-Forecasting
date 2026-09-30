"""
main.py - End-to-End Pipeline Orchestrator
==========================================
Stock Market Volatility Forecasting & Algorithmic Trading using ML

This script runs the complete pipeline:
  1. Data loading (Yahoo Finance / synthetic fallback)
  2. Preprocessing (cleaning, returns, volatility, targets)
  3. Feature engineering (technical indicators)
  4. Classical ML baselines (Ridge, Random Forest, XGBoost)
  5. PyTorch LSTM training with early stopping
  6. Hyperparameter experiments (window size, hidden size, dropout, lr)
  7. Model comparison (MAE, RMSE, R²)
  8. Trading signal generation
  9. Backtesting with transaction costs
 10. Financial performance analysis (Sharpe, MDD, Sortino, etc.)
 11. Comprehensive visualisation suite (11 plots saved to plots/)
 12. Results export to results/

Run:
    python main.py                      # default: S&P 500, real data
    python main.py --ticker ^NSEI       # NIFTY 50
    python main.py --synthetic          # use synthetic GBM data (no internet)
    python main.py --skip-experiments   # skip hyperparameter sweep (faster)

Reproducibility
---------------
All random seeds are fixed.  Data is cached after first download.
Results are saved to results/ and plots/ directories.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch

warnings.filterwarnings("ignore")

# ---- Resolve src/ directory on path ----
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

from data_loader import load_data
from preprocessing import preprocess_pipeline, split_time_series, scale_features
from features import build_features, get_feature_columns
from models import train_and_evaluate_all, compute_regression_metrics as classical_metrics
from lstm_model import build_lstm, build_rnn, build_gru, make_dataloader
from training import train_model, get_predictions, run_experiments, save_model
from evaluation import (compute_regression_metrics, compute_financial_metrics,
                         build_model_comparison_table, print_metrics_table,
                         get_drawdown_series)
from backtesting import (generate_signals, run_backtest, run_buy_and_hold,
                          backtest_summary, compute_adaptive_threshold)
from tuning import run_hyperparameter_tuning, update_lstm_tuning_results
from feature_selection import run_feature_selection

# ===================================================================
# GLOBAL CONFIGURATION
# ===================================================================
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RANDOM_SEED = 42
SEQ_LEN = 30          # Default LSTM lookback window
BATCH_SIZE = 32
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
EPOCHS = 100
PATIENCE = 15
BUY_THRESHOLD = 0.002
SELL_THRESHOLD = 0.002
INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST = 0.001


def set_seeds(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================================================================
# VISUALISATION HELPERS
# ===================================================================
def save_fig(fig, name: str):
    """Save figure to plots/ directory."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved -> {path}")


def set_style():
    """Apply a clean, professional plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    })


# ===================================================================
# PLOT 1: Stock Price Over Time
# ===================================================================
def plot_price(df: pd.DataFrame, ticker: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Close price
    axes[0].plot(df.index, df["Close"], color="#1f77b4", linewidth=0.8, label="Close Price")
    axes[0].fill_between(df.index, df["Low"], df["High"], alpha=0.15, color="#1f77b4", label="High-Low Range")
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"{ticker} - Historical Price")
    axes[0].legend(loc="upper left")

    # Daily returns
    axes[1].plot(df.index, df["Daily_Return"] * 100, color="#2ca02c", linewidth=0.5, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Daily Return (%)")
    axes[1].set_title("Daily Returns")

    # Volume
    axes[2].bar(df.index, df["Volume"] / 1e9, color="#ff7f0e", alpha=0.6, width=1)
    axes[2].set_ylabel("Volume (Billions)")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Trading Volume")

    fig.suptitle(f"{ticker} - Historical Market Data Overview", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "01_price_overview.png")


# ===================================================================
# PLOT 2: Returns Distribution & Volatility
# ===================================================================
def plot_returns_and_volatility(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Returns distribution
    axes[0, 0].hist(df["Daily_Return"].dropna() * 100, bins=80, color="#1f77b4",
                    alpha=0.75, edgecolor="white", linewidth=0.3)
    axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("Daily Return (%)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Daily Return Distribution")

    # Rolling volatility over time
    axes[0, 1].plot(df.index, df["Volatility_21d"] * 100, color="#d62728", linewidth=0.8)
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Annualised Volatility (%)")
    axes[0, 1].set_title("21-Day Rolling Historical Volatility")

    # QQ plot (returns vs normal)
    from scipy import stats
    returns_clean = df["Daily_Return"].dropna() * 100
    (osm, osr), (slope, intercept, r) = stats.probplot(returns_clean, dist="norm", fit=True)
    axes[1, 0].plot(osm, osr, ".", color="#1f77b4", markersize=2, alpha=0.5)
    axes[1, 0].plot([osm[0], osm[-1]],
                    [slope * osm[0] + intercept, slope * osm[-1] + intercept],
                    "r-", linewidth=1.5, label=f"Normal fit (R²={r**2:.3f})")
    axes[1, 0].set_xlabel("Theoretical Quantiles")
    axes[1, 0].set_ylabel("Sample Quantiles")
    axes[1, 0].set_title("Q-Q Plot of Daily Returns vs Normal Distribution")
    axes[1, 0].legend()

    # Autocorrelation of returns (lag 1-30)
    returns_clean_arr = df["Daily_Return"].dropna().values
    lags = range(1, 31)
    acf_vals = [pd.Series(returns_clean_arr).autocorr(lag=l) for l in lags]
    axes[1, 1].bar(list(lags), acf_vals, color="#9467bd", alpha=0.7)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    ci = 1.96 / np.sqrt(len(returns_clean_arr))
    axes[1, 1].axhline(ci, color="red", linestyle="--", linewidth=0.8, label=f"95% CI (±{ci:.3f})")
    axes[1, 1].axhline(-ci, color="red", linestyle="--", linewidth=0.8)
    axes[1, 1].set_xlabel("Lag (days)")
    axes[1, 1].set_ylabel("Autocorrelation")
    axes[1, 1].set_title("Return Autocorrelation (Lags 1-30)")
    axes[1, 1].legend()

    fig.suptitle("Returns Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "02_returns_volatility.png")


# ===================================================================
# PLOT 3: Feature Correlation Heatmap
# ===================================================================
def plot_feature_correlation(featured_df: pd.DataFrame, feature_cols: list):
    # Use a subset of features to keep heatmap readable
    cols_to_plot = [c for c in feature_cols if c in featured_df.columns][:20]
    corr = featured_df[cols_to_plot].dropna().corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.4,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Feature Correlation Matrix (Top 20 Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "03_feature_correlation.png")


# ===================================================================
# PLOT 4: Actual vs Predicted Returns (all models)
# ===================================================================
def plot_actual_vs_predicted(
    test_dates: pd.DatetimeIndex,
    y_test: np.ndarray,
    classical_results: dict,
    nn_results: dict,
    nn_dates: pd.DatetimeIndex,
):
    models = {k: v["test_preds"] for k, v in classical_results.items()}
    if nn_results is not None:
        for k, v in nn_results.items():
            models[k] = v["preds"]

    n_models = len(models)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(14, 4 * (n_models + 1)), sharex=False)

    # Top panel: actual returns
    axes[0].plot(test_dates, y_test * 100, color="black", linewidth=0.7, label="Actual Return")
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title("Actual Next-Day Returns (Test Set)")
    axes[0].legend()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, preds) in enumerate(models.items()):
        ax = axes[i + 1]
        # For NN models, use nn_dates (slightly shorter due to seq_len)
        dates_to_use = nn_dates if (nn_results and name in nn_results) else test_dates
        y_to_use = y_test[-len(preds):] if (nn_results and name in nn_results) else y_test

        ax.plot(dates_to_use, y_to_use * 100, color="black", linewidth=0.5,
                alpha=0.5, label="Actual")
        ax.plot(dates_to_use, preds * 100, color=colors[i % len(colors)],
                linewidth=0.8, alpha=0.8, label=f"{name} Predicted")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.set_ylabel("Return (%)")
        ax.set_title(f"{name} - Actual vs Predicted")
        ax.legend(loc="upper left")

    plt.tight_layout()
    save_fig(fig, "04_actual_vs_predicted.png")


# ===================================================================
# PLOT 5: Deep Learning Training / Validation Loss Curves
# ===================================================================
def plot_loss_curves(nn_results: dict):
    n_models = len(nn_results)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, nn_results.items()):
        train_losses = data["train_losses"]
        val_losses = data["val_losses"]
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label="Training Loss", color="#1f77b4", linewidth=1.5)
        ax.plot(epochs, val_losses, label="Validation Loss", color="#d62728", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{name} Training & Validation Loss")
        
        best_epoch = int(np.argmin(val_losses)) + 1
        best_loss = min(val_losses)
        ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5,
                   label=f"Best epoch: {best_epoch}\n(val loss={best_loss:.5f})")
        ax.legend(fontsize=9)
        
    plt.tight_layout()
    save_fig(fig, "05_loss_curves.png")



# ===================================================================
# PLOT 6: Model Comparison Bar Chart
# ===================================================================
def plot_model_comparison(comparison_df: pd.DataFrame):
    metrics = ["MAE", "RMSE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, metric in enumerate(["MAE", "RMSE", "R²"]):
        col = metric
        if col not in comparison_df.columns:
            col = metric.replace("²", "2")
        vals = comparison_df[col] if col in comparison_df.columns else comparison_df.get("R2", [])
        axes[i].bar(comparison_df["Model"], vals,
                    color=colors[:len(comparison_df)], alpha=0.85, edgecolor="white")
        axes[i].set_title(f"Model Comparison - {metric}")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis="x", rotation=25)
        for j, v in enumerate(vals):
            axes[i].text(j, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Classical ML vs LSTM - Prediction Metrics (Test Set)", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "06_model_comparison.png")


# ===================================================================
# PLOT 7: Portfolio Value Over Time
# ===================================================================
def plot_portfolio(backtest_dfs: dict, title: str = "Portfolio Value Over Time"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for i, (name, df) in enumerate(backtest_dfs.items()):
        color = colors[i % len(colors)]
        axes[0].plot(df.index, df["Portfolio_Value"],
                     color=color, linewidth=1.2, label=name)

    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title(title)
    axes[0].legend()
    initial = list(backtest_dfs.values())[0].attrs.get("initial_capital", 100_000)
    axes[0].axhline(initial, color="gray", linestyle="--", linewidth=0.8, label="Initial Capital")

    # Drawdown panel
    for i, (name, df) in enumerate(backtest_dfs.items()):
        color = colors[i % len(colors)]
        axes[1].fill_between(df.index, df["Drawdown"] * 100, 0,
                             alpha=0.4, color=color, label=f"{name} Drawdown")
        axes[1].plot(df.index, df["Drawdown"] * 100, color=color, linewidth=0.6)

    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Drawdown from Peak")
    axes[1].legend()

    plt.tight_layout()
    save_fig(fig, "07_portfolio_performance.png")


# ===================================================================
# PLOT 8: Signal Distribution Over Time
# ===================================================================
def plot_signals(backtest_df: pd.DataFrame, actual_returns: np.ndarray):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Actual returns coloured by signal
    signals = backtest_df["Signal"].values
    returns_pct = actual_returns * 100

    for sig, color, label in [(1, "#2ca02c", "Long"), (0, "#aec7e8", "Flat"), (-1, "#d62728", "Short/Sell")]:
        mask = signals == sig
        if mask.sum() > 0:
            axes[0].bar(backtest_df.index[mask], returns_pct[mask],
                        color=color, alpha=0.6, label=label, width=1.5)

    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Actual Return (%)")
    axes[0].set_title("Actual Returns Coloured by Trading Signal")
    axes[0].legend()

    # Position over time
    axes[1].step(backtest_df.index, backtest_df["Position"],
                 color="#1f77b4", linewidth=0.8, where="post")
    axes[1].fill_between(backtest_df.index, backtest_df["Position"],
                         step="post", alpha=0.3, color="#1f77b4")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["Short", "Flat", "Long"])
    axes[1].set_ylabel("Position")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Position Over Time")

    plt.tight_layout()
    save_fig(fig, "08_trading_signals.png")


# ===================================================================
# PLOT 9: Hyperparameter Experiment Results
# ===================================================================
def plot_experiments(exp_df: pd.DataFrame):
    if exp_df is None or exp_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    params = ["seq_len", "hidden_size", "dropout", "lr"]
    titles = ["Lookback Window (seq_len)", "Hidden Size", "Dropout Rate", "Learning Rate"]

    for i, (param, title) in enumerate(zip(params, titles)):
        ax = axes[i // 2][i % 2]
        if param in exp_df.columns:
            grouped = exp_df.groupby(param)["best_val_loss"].min().reset_index()
            ax.bar(grouped[param].astype(str), grouped["best_val_loss"],
                   color="#1f77b4", alpha=0.8, edgecolor="white")
            ax.set_xlabel(param)
            ax.set_ylabel("Best Val Loss (MSE)")
            ax.set_title(f"Effect of {title}")
            for j, row in grouped.iterrows():
                ax.text(j, row["best_val_loss"],
                        f"{row['best_val_loss']:.5f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Hyperparameter Experiments - LSTM Validation Loss", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "09_hyperparameter_experiments.png")


# ===================================================================
# PLOT 10: Scatter - Actual vs Predicted (per model)
# ===================================================================
def plot_scatter_actual_vs_pred(y_true: np.ndarray, predictions_dict: dict):
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        y_t = y_true[-len(preds):]
        ax = axes[i]
        ax.scatter(y_t, preds, alpha=0.3, s=8, color=colors[i % len(colors)])
        lim = max(abs(y_t).max(), abs(preds).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Perfect fit")
        ax.set_xlabel("Actual Return")
        ax.set_ylabel("Predicted Return")
        ax.set_title(f"{name}\nActual vs Predicted")
        ax.legend()

    fig.suptitle("Scatter: Actual vs Predicted Returns", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "10_scatter_actual_vs_predicted.png")


# ===================================================================
# PLOT 11: Feature Importances (Random Forest & XGBoost)
# ===================================================================
def plot_feature_importances(classical_results: dict, feature_cols: list):
    fig, axes = plt.subplots(1, len(classical_results), figsize=(7 * len(classical_results), 7))
    if len(classical_results) == 1:
        axes = [axes]

    for i, (name, res) in enumerate(classical_results.items()):
        model = res["model"]
        if not hasattr(model, "feature_importances_"):
            continue
        importances = model.feature_importances_
        # Top 15
        top_idx = np.argsort(importances)[::-1][:15]
        top_features = [feature_cols[j] for j in top_idx]
        top_vals = importances[top_idx]

        axes[i].barh(top_features[::-1], top_vals[::-1], color="#1f77b4", alpha=0.8)
        axes[i].set_xlabel("Feature Importance")
        axes[i].set_title(f"{name} - Top 15 Feature Importances")

    fig.suptitle("Feature Importances", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "11_feature_importances.png")


# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main(args):
    set_seeds()
    set_style()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STOCK MARKET VOLATILITY FORECASTING & ALGORITHMIC TRADING")
    print("  Educational ML Coursework Project")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load Data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading data...")
    df_raw = load_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        use_cache=not args.no_cache,
        force_synthetic=args.synthetic,
    )

    # ------------------------------------------------------------------
    # STEP 2: Preprocessing
    # ------------------------------------------------------------------
    print("\n[Step 2] Preprocessing...")
    proc_result = preprocess_pipeline(df_raw, volatility_window=21)
    df_proc = proc_result["processed_df"]

    # Merge raw OHLCV back for feature computation
    raw_ohlcv = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw_ohlcv.index = pd.to_datetime(raw_ohlcv.index)
    df_merged = df_proc.join(raw_ohlcv, how="left", rsuffix="_raw")

    # ------------------------------------------------------------------
    # STEP 3: Feature Engineering
    # ------------------------------------------------------------------
    print("\n[Step 3] Engineering features...")
    df_featured = build_features(df_merged)
    df_featured = df_featured.dropna()

    feature_cols = get_feature_columns(df_featured)
    print(f"  -> {len(feature_cols)} features: {feature_cols}")

    # ------------------------------------------------------------------
    # STEP 4: Time-Series Split
    # ------------------------------------------------------------------
    print("\n[Step 4] Chronological train/val/test split...")
    train_df, val_df, test_df = split_time_series(df_featured, train_ratio=0.70, val_ratio=0.15)

    # ------------------------------------------------------------------
    # STEP 4.5: Feature Selection (uses ONLY training data)
    # ------------------------------------------------------------------
    target_col = "Target_Return"
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[target_col].values

    feature_cols = run_feature_selection(
        df=df_featured,
        feature_cols=feature_cols,
        X_train=X_train_raw,
        y_train=y_train_raw,
        corr_threshold=0.90,
        top_k=10,
    )
    print(f"\n  -> Selected {len(feature_cols)} features: {feature_cols}")

    # ------------------------------------------------------------------
    # STEP 5: Scale Features (fit on train only -> no data leakage)
    # ------------------------------------------------------------------
    print("\n[Step 5] Scaling features (StandardScaler fit on train only)...")
    train_sc, val_sc, test_sc, scaler = scale_features(
        train_df, val_df, test_df, feature_cols=feature_cols
    )

    # Extract arrays
    X_train = train_sc[feature_cols].values
    y_train = train_sc[target_col].values
    X_val = val_sc[feature_cols].values
    y_val = val_sc[target_col].values
    X_test = test_sc[feature_cols].values
    y_test = test_sc[target_col].values

    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")

    # ------------------------------------------------------------------
    # STEP 6: Visualise Raw Data
    # ------------------------------------------------------------------
    print("\n[Step 6] Generating data visualisations...")
    plot_price(df_merged.dropna(), ticker=args.ticker)
    plot_returns_and_volatility(df_merged.dropna())
    plot_feature_correlation(df_featured, feature_cols)

    # ------------------------------------------------------------------
    # STEP 7: Classical ML Baselines
    # ------------------------------------------------------------------
    print("\n[Step 7] Training classical ML baselines...")
    classical_results = train_and_evaluate_all(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        target_name="Next-Day Return",
    )

    # ------------------------------------------------------------------
    # STEP 8: PyTorch Neural Networks (RNN, LSTM, GRU)
    # ------------------------------------------------------------------
    print(f"\n[Step 8] Training Neural Network models (seq_len={SEQ_LEN}, hidden={LSTM_HIDDEN})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(X_train, y_train, seq_len=SEQ_LEN,
                                   batch_size=BATCH_SIZE, shuffle=False)
    val_loader = make_dataloader(X_val, y_val, seq_len=SEQ_LEN,
                                 batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, seq_len=SEQ_LEN,
                                  batch_size=BATCH_SIZE, shuffle=False)

    nn_builders = {
        "Vanilla RNN": build_rnn,
        "LSTM": build_lstm,
        "GRU": build_gru
    }
    
    nn_results = {}
    dl_metrics_list = []
    
    import time
    for name, builder in nn_builders.items():
        print(f"\n  ---> Training {name} <---")
        model = builder(
            input_size=len(feature_cols),
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            dropout=DROPOUT,
        )

        start_time = time.time()
        train_result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LR,
            patience=PATIENCE,
            weight_decay=1e-4,
            device=device,
            verbose=True,
        )
        training_time = time.time() - start_time
        
        trained_model = train_result["model"]
        save_model(trained_model, os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').lower()}_best.pt"))
        
        # Test predictions
        preds, targets = get_predictions(trained_model, test_loader, device)
        metrics = compute_regression_metrics(targets, preds, model_name=name)
        metrics["Training_Time_s"] = round(training_time, 1)
        dl_metrics_list.append(metrics)
        
        print(f"  {name} Test Metrics: MAE={metrics['MAE']:.6f}  "
              f"RMSE={metrics['RMSE']:.6f}  R\u00b2={metrics['R2']:.6f}  Time={training_time:.1f}s")
        
        nn_results[name] = {
            "train_losses": train_result["train_losses"],
            "val_losses": train_result["val_losses"],
            "model": trained_model,
            "preds": preds,
            "targets": targets
        }
        
        # For compatibility with subsequent steps (which expect 'lstm'), we'll keep a reference
        # to the LSTM specifically, since the rest of the script is hardcoded to use it for backtesting.
        if name == "LSTM":
            lstm = trained_model
            lstm_preds = preds
            lstm_targets = targets
            
            # LSTM dates (shorter due to seq_len window)
            lstm_test_dates = test_df.index[SEQ_LEN - 1: SEQ_LEN - 1 + len(lstm_preds)]
            # Track which seq_len is active (may change after tuning)
            active_seq_len = SEQ_LEN

            # LSTM val predictions -- baseline 'before tuning' val score
            lstm_val_preds, lstm_val_targets = get_predictions(lstm, val_loader, device)
            lstm_val_metrics = compute_regression_metrics(
                lstm_val_targets, lstm_val_preds, model_name="LSTM_val"
            )
            print(f"  LSTM Val  Metrics: MAE={lstm_val_metrics['MAE']:.6f}  "
                  f"RMSE={lstm_val_metrics['RMSE']:.6f}  R\u00b2={lstm_val_metrics['R2']:.6f}")

    # Loss curves plot for all 3 models
    plot_loss_curves(nn_results)
    
    print("\n" + "=" * 60)
    print("  NEURAL NETWORK COMPARISON TABLE")
    print("=" * 60)
    nn_comp_df = pd.DataFrame(dl_metrics_list)[["Model", "RMSE", "MAE", "Training_Time_s"]]
    print(nn_comp_df.to_string(index=False))
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 9: Hyperparameter Experiments (existing sweep)
    # ------------------------------------------------------------------
    exp_df = None
    if not args.skip_experiments:
        print("\n[Step 9] Running hyperparameter experiments...")
        exp_df = run_experiments(
            X_train, y_train, X_val, y_val,
            input_size=len(feature_cols),
            epochs=40,
            patience=8,
        )
        plot_experiments(exp_df)
    else:
        print("\n[Step 9] Skipping hyperparameter experiments (--skip-experiments flag).")

    # ------------------------------------------------------------------
    # STEP 9.5: Hyperparameter Tuning (--tune or --fast-tune)
    # ------------------------------------------------------------------
    run_tuning = getattr(args, "tune", False) or getattr(args, "fast_tune", False)
    if run_tuning:
        fast_mode = getattr(args, "fast_tune", False)
        print(f"\n[Step 9.5] Hyperparameter tuning  "
              f"({'fast mode' if fast_mode else 'full mode'})...")

        tuning_result = run_hyperparameter_tuning(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            input_size=len(feature_cols),
            device=device,
            baseline_classical_results=classical_results,
            baseline_lstm_val_metrics=lstm_val_metrics,
            fast_mode=fast_mode,
        )

        best_lstm_p = tuning_result["best_lstm_params"]
        print(f"\n  [Tuning] Retraining LSTM with best params: {best_lstm_p}")

        # Rebuild DataLoaders for the tuned seq_len (CRITICAL)
        tuned_seq_len  = int(best_lstm_p["seq_len"])
        tuned_batch    = int(best_lstm_p["batch_size"])
        train_loader   = make_dataloader(X_train, y_train, seq_len=tuned_seq_len,
                                         batch_size=tuned_batch, shuffle=False)
        val_loader     = make_dataloader(X_val,   y_val,   seq_len=tuned_seq_len,
                                         batch_size=tuned_batch, shuffle=False)
        test_loader    = make_dataloader(X_test,  y_test,  seq_len=tuned_seq_len,
                                         batch_size=tuned_batch, shuffle=False)

        # Build and retrain LSTM with best hyperparameters
        lstm_tuned = build_lstm(
            input_size=len(feature_cols),
            hidden_size=int(best_lstm_p["hidden_size"]),
            num_layers=int(best_lstm_p["num_layers"]),
            dropout=float(best_lstm_p["dropout"]),
        )
        tuned_train_result = train_model(
            model=lstm_tuned,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=float(best_lstm_p["lr"]),
            patience=PATIENCE,
            weight_decay=1e-4,
            device=device,
            verbose=True,
        )
        lstm_tuned = tuned_train_result["model"]
        save_model(lstm_tuned, os.path.join(RESULTS_DIR, "lstm_tuned_best.pt"))

        # Evaluate tuned LSTM on val and test
        tuned_val_preds,  tuned_val_targets  = get_predictions(lstm_tuned, val_loader,  device)
        tuned_test_preds, tuned_test_targets = get_predictions(lstm_tuned, test_loader, device)

        tuned_val_m  = compute_regression_metrics(tuned_val_targets,  tuned_val_preds,
                                                  model_name="LSTM_tuned_val")
        tuned_test_m = compute_regression_metrics(tuned_test_targets, tuned_test_preds,
                                                  model_name="LSTM_tuned_test")
        print(f"  [Tuned LSTM] Val  RMSE={tuned_val_m['RMSE']:.6f}  "
              f"R\u00b2={tuned_val_m['R2']:.6f}")
        print(f"  [Tuned LSTM] Test RMSE={tuned_test_m['RMSE']:.6f}  "
              f"R\u00b2={tuned_test_m['R2']:.6f}")

        # Fill LSTM row in comparison CSV (after-val/test scores)
        update_lstm_tuning_results(
            after_val_rmse=tuned_val_m["RMSE"],
            after_val_r2=tuned_val_m["R2"],
            final_test_rmse=tuned_test_m["RMSE"],
            final_test_r2=tuned_test_m["R2"],
        )

        # Replace default LSTM with tuned LSTM for downstream steps
        lstm          = lstm_tuned
        lstm_preds    = tuned_test_preds
        lstm_targets  = tuned_test_targets
        lstm_metrics  = tuned_test_m
        active_seq_len = tuned_seq_len
        lstm_test_dates = test_df.index[
            tuned_seq_len - 1: tuned_seq_len - 1 + len(lstm_preds)
        ]

        # Tuning comparison bar-chart
        plot_tuning_comparison()
    else:
        print("\n[Step 9.5] Hyperparameter tuning skipped "
              "(use --tune or --fast-tune to enable).")

    # ------------------------------------------------------------------
    # STEP 10: Model Comparison Table
    # ------------------------------------------------------------------
    print("\n[Step 10] Comparing all models...")
    comparison_df = build_model_comparison_table(
        classical_results, dl_metrics_list, target_name="Next-Day Return", split="test"
    )
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    # Visualise
    test_dates = test_df.index
    plot_actual_vs_predicted(
        test_dates, y_test, classical_results, nn_results, lstm_test_dates
    )
    all_preds_for_scatter = {k: v["test_preds"] for k, v in classical_results.items()}
    for k, v in nn_results.items():
        all_preds_for_scatter[k] = v["preds"]
    plot_scatter_actual_vs_pred(y_test, all_preds_for_scatter)
    plot_model_comparison(comparison_df)
    plot_feature_importances(classical_results, feature_cols)

    # ------------------------------------------------------------------
    # STEP 11: Trading Strategy & Backtesting
    # ------------------------------------------------------------------
    print("\n[Step 11] Running backtests...")

    # Use LSTM predictions for main strategy
    # Fix #1: Use adaptive threshold — LSTM predictions are clustered near 0
    # and never cross the fixed tau=0.002. Adaptive tau = p60 of |preds|,
    # guaranteeing the top 40% of signals trigger trades.
    print("\n  [LSTM Backtest] Running prediction diagnostics...")
    lstm_signals = generate_signals(
        lstm_preds,
        adaptive=True,
        adaptive_percentile=60.0,
    )

    # Align actual returns to LSTM test window
    actual_ret_lstm = test_df["Daily_Return"].values[
        active_seq_len - 1: active_seq_len - 1 + len(lstm_preds)
    ]

    bt_lstm = run_backtest(
        lstm_signals, actual_ret_lstm, lstm_test_dates,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST,
        strategy_name="LSTM Strategy",
    )

    # Random Forest strategy
    rf_preds = classical_results.get("RandomForest", {}).get("test_preds", None)
    bt_rf = None
    if rf_preds is not None:
        rf_signals = generate_signals(rf_preds, buy_threshold=BUY_THRESHOLD, sell_threshold=SELL_THRESHOLD)
        bt_rf = run_backtest(
            rf_signals, y_test, test_dates,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost_pct=TRANSACTION_COST,
            strategy_name="Random Forest Strategy",
        )

    # Buy-and-Hold benchmark (use LSTM test window)
    bh = run_buy_and_hold(actual_ret_lstm, lstm_test_dates, initial_capital=INITIAL_CAPITAL)

    # Compile backtest dfs for plotting
    backtest_dfs = {"LSTM Strategy": bt_lstm, "Buy & Hold": bh}
    if bt_rf is not None:
        backtest_dfs["Random Forest Strategy"] = bt_rf

    plot_portfolio(backtest_dfs)
    plot_signals(bt_lstm, actual_ret_lstm)

    # ------------------------------------------------------------------
    # STEP 12: Financial Performance Metrics
    # ------------------------------------------------------------------
    print("\n[Step 12] Computing financial performance metrics...")
    all_financial = []

    fm_lstm = compute_financial_metrics(
        pd.Series(bt_lstm["Strategy_Return"].values, index=lstm_test_dates),
        strategy_name="LSTM Strategy",
    )
    all_financial.append(fm_lstm)

    if bt_rf is not None:
        fm_rf = compute_financial_metrics(
            pd.Series(bt_rf["Strategy_Return"].values, index=test_dates),
            strategy_name="RF Strategy",
        )
        all_financial.append(fm_rf)

    fm_bh = compute_financial_metrics(
        pd.Series(bh["Strategy_Return"].values, index=lstm_test_dates),
        strategy_name="Buy & Hold",
    )
    all_financial.append(fm_bh)

    print_metrics_table(all_financial)

    # Save financial metrics
    fin_df = pd.DataFrame(all_financial)
    fin_df.to_csv(os.path.join(RESULTS_DIR, "financial_metrics.csv"), index=False)

    # Backtest summaries
    summaries = [backtest_summary(bt_lstm), backtest_summary(bh)]
    if bt_rf is not None:
        summaries.append(backtest_summary(bt_rf))
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(os.path.join(RESULTS_DIR, "backtest_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Plots saved to  : {PLOTS_DIR}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("\n  Generated files:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        print(f"    plots/{f}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"    results/{f}")
    print()


# ===================================================================
# PLOT 12: Hyperparameter Tuning Comparison
# ===================================================================
def plot_tuning_comparison():
    """
    Read before_vs_after_tuning.csv and plot a grouped bar chart:
      - Before Tuning Val RMSE (default hyperparams)
      - After Tuning  Val RMSE (tuned hyperparams, selected by val not test)
      - Annotate Final Test RMSE above each 'after' bar

    Saved to plots/12_tuning_comparison.png.
    """
    comp_path = os.path.join(RESULTS_DIR, "before_vs_after_tuning.csv")
    if not os.path.exists(comp_path):
        print("  [Plot] Tuning comparison CSV not found -- skipping plot.")
        return

    df = pd.read_csv(comp_path)
    # Only rows with valid After_Val_RMSE
    df = df.dropna(subset=["After_Val_RMSE"])
    if df.empty:
        print("  [Plot] No complete tuning rows -- skipping tuning comparison plot.")
        return

    set_style()
    models = df["Model"].tolist()
    x      = np.arange(len(models))
    width  = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: RMSE bar chart ----
    ax = axes[0]
    before_rmse = pd.to_numeric(df["Before_Val_RMSE"], errors="coerce").tolist()
    after_rmse  = pd.to_numeric(df["After_Val_RMSE"],  errors="coerce").tolist()
    test_rmse   = pd.to_numeric(df.get("Final_Test_RMSE", [np.nan] * len(models)),
                                errors="coerce").tolist()

    bars_b = ax.bar(x - width / 2, before_rmse, width, label="Before Tuning (Val)",
                    color="#4c72b0", alpha=0.85, edgecolor="white")
    bars_a = ax.bar(x + width / 2, after_rmse,  width, label="After Tuning (Val)",
                    color="#55a868", alpha=0.85, edgecolor="white")

    # Annotate Final Test RMSE above each 'after' bar
    for bar, trmse in zip(bars_a, test_rmse):
        if not np.isnan(trmse):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                    f"Test:{trmse:.4f}", ha="center", va="bottom", fontsize=7,
                    color="#c44e52", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Val RMSE: Before vs After Tuning\n(Test RMSE annotated, selected by val only)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(filter(lambda v: not np.isnan(v), before_rmse + after_rmse)) * 1.3)

    # ---- Right: Improvement % bar chart ----
    ax2 = axes[1]
    improvements = pd.to_numeric(
        df.get("Val_RMSE_Improvement_%", [np.nan] * len(models)), errors="coerce"
    ).tolist()

    colors = ["#55a868" if (not np.isnan(v) and v >= 0) else "#c44e52"
              for v in improvements]
    bars_i = ax2.bar(x, improvements, width * 1.8, color=colors, alpha=0.85, edgecolor="white")

    for bar, imp in zip(bars_i, improvements):
        if not np.isnan(imp):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.1 if imp >= 0 else -0.3),
                     f"{imp:+.1f}%", ha="center", va="bottom" if imp >= 0 else "top",
                     fontsize=9, fontweight="bold")

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_ylabel("Val RMSE Improvement (%)")
    ax2.set_title("Hyperparameter Tuning Gain\n(positive = tuning helped)", fontsize=11)

    fig.suptitle("Hyperparameter Tuning Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "12_tuning_comparison.png")


# ===================================================================
# ARGUMENT PARSER
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Stock Market Volatility Forecasting & Algorithmic Trading - ML Coursework"
    )
    parser.add_argument("--ticker", type=str, default="^GSPC",
                        help="Yahoo Finance ticker (default: ^GSPC = S&P 500)")
    parser.add_argument("--start", type=str, default="2010-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic GBM data instead of real data")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-download data even if cache exists")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="Skip hyperparameter sweep (faster run)")
    parser.add_argument("--tune", action="store_true",
                        help="Run full hyperparameter tuning for all models (Ridge, RF, XGB, LSTM)")
    parser.add_argument("--fast-tune", action="store_true",
                        help="Run hyperparameter tuning with reduced grids for quick testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
