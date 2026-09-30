"""
run_sequence_experiment.py
==========================
ML Research Experiment: How much historical context is optimal?
Tests Sequence Lengths [5, 10, 20, 30, 60] across Vanilla RNN, LSTM, and GRU.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ---- Resolve src/ directory on path ----
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

from data_loader import load_data
from preprocessing import preprocess_pipeline, split_time_series, scale_features
from features import build_features, get_feature_columns
from lstm_model import build_lstm, build_rnn, build_gru, make_dataloader
from training import train_model, get_predictions
from evaluation import compute_regression_metrics
from feature_selection import run_feature_selection

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def set_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14
    })

def plot_sequence_experiment(df: pd.DataFrame):
    """Plot MAE vs Sequence Length for each model architecture."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"Vanilla RNN": "#d62728", "LSTM": "#1f77b4", "GRU": "#2ca02c"}
    markers = {"Vanilla RNN": "o", "LSTM": "s", "GRU": "^"}
    
    for model_name in df["Model"].unique():
        model_df = df[df["Model"] == model_name].sort_values("Sequence_Length")
        ax.plot(
            model_df["Sequence_Length"], 
            model_df["Test_MAE"], 
            marker=markers.get(model_name, "o"), 
            linewidth=2, 
            markersize=8,
            color=colors.get(model_name, "black"),
            label=model_name
        )
        
    ax.set_xlabel("Sequence Length (Days of Historical Context)")
    ax.set_ylabel("Test MAE (Lower is better)")
    ax.set_title("How Much History Do Neural Networks Need?\nTest MAE vs Sequence Length")
    ax.set_xticks([5, 10, 20, 30, 60])
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "13_sequence_experiment.png")
    fig.savefig(plot_path, dpi=300)
    print(f"  [Plot] Saved -> {plot_path}")


def main(args):
    print("=" * 60)
    print("  ML RESEARCH EXPERIMENT: SEQUENCE LENGTH MEMORY")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Step 1] Loading data...")
    df = load_data(args.ticker, args.start, args.end, force_synthetic=args.synthetic, use_cache=True)
    
    # 2. Preprocess & Feature Engineering
    print("\n[Step 2] Preprocessing...")
    proc_result = preprocess_pipeline(df, volatility_window=21)
    df_proc = proc_result["processed_df"]
    
    raw_ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw_ohlcv.index = pd.to_datetime(raw_ohlcv.index)
    df_merged = df_proc.join(raw_ohlcv, how="left", rsuffix="_raw")
    
    df_featured = build_features(df_merged)
    df_featured = df_featured.dropna()
    feature_cols = get_feature_columns(df_featured)
    target_col = "Target_Return"
    
    # 3. Train/Val/Test Split
    print("\n[Step 3] Splitting data...")
    train_df, val_df, test_df = split_time_series(df_featured, train_ratio=0.7, val_ratio=0.15)
    
    # 3.5 Feature Selection (uses ONLY training data)
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
    
    # 4. Scale Features
    print("\n[Step 4] Scaling features...")
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Hardware] Using device: {device}")
    
    seq_lengths = [5, 10, 20, 30, 60]
    
    nn_builders = {
        "Vanilla RNN": build_rnn,
        "LSTM": build_lstm,
        "GRU": build_gru
    }
    
    results = []
    
    print("\n[Step 5] Running Experiment Loop...")
    for seq_len in seq_lengths:
        print(f"\n{'='*50}")
        print(f"  Testing Sequence Length: {seq_len} Days")
        print(f"{'='*50}")
        
        # Build dataloaders for this specific sequence length
        train_loader = make_dataloader(X_train, y_train, seq_len=seq_len, batch_size=32, shuffle=False)
        val_loader   = make_dataloader(X_val,   y_val,   seq_len=seq_len, batch_size=32, shuffle=False)
        test_loader  = make_dataloader(X_test,  y_test,  seq_len=seq_len, batch_size=32, shuffle=False)
        
        for model_name, builder in nn_builders.items():
            print(f"\n  ---> Training {model_name} (seq_len={seq_len}) <---")
            
            model = builder(
                input_size=len(feature_cols),
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
            )
            
            start_time = time.time()
            # Fast training for experiment: max 40 epochs, patience 5
            train_result = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=40,
                lr=1e-3,
                patience=5,
                weight_decay=1e-4,
                device=device,
                verbose=False,
            )
            training_time = time.time() - start_time
            
            trained_model = train_result["model"]
            preds, targets = get_predictions(trained_model, test_loader, device)
            metrics = compute_regression_metrics(targets, preds, model_name=model_name)
            
            print(f"       Test MAE: {metrics['MAE']:.6f} | RMSE: {metrics['RMSE']:.6f} | Time: {training_time:.1f}s")
            
            results.append({
                "Model": model_name,
                "Sequence_Length": seq_len,
                "Test_MAE": metrics["MAE"],
                "Test_RMSE": metrics["RMSE"],
                "Training_Time_s": round(training_time, 1)
            })
            
    print("\n[Step 6] Compiling Results...")
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "sequence_experiment.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  [Saved] -> {csv_path}")
    
    print("\n" + "=" * 60)
    print("  SEQUENCE EXPERIMENT RESULTS TABLE")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("=" * 60)
    
    plot_sequence_experiment(df_results)
    print("\nExperiment Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence Length Experiment")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Yahoo Finance ticker")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()
    main(args)
