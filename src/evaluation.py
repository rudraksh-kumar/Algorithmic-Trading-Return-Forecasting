"""
evaluation.py - Statistical & Financial Performance Metrics
=============================================================
This module computes two categories of metrics:

A) ML Prediction Metrics (how well the model forecasts):
   - MAE   : Mean Absolute Error
   - RMSE  : Root Mean Squared Error
   - R²    : Coefficient of Determination
   - Directional Accuracy : % of times predicted direction matches actual

B) Financial / Portfolio Metrics (how well the trading strategy performs):
   - Total Return
   - Annualised Return (CAGR)
   - Annualised Volatility
   - Sharpe Ratio
   - Sortino Ratio
   - Maximum Drawdown (MDD)
   - Calmar Ratio
   - Win Rate
   - Number of Trades
   - Profit Factor

These two categories are deliberately kept separate.
A model with poor R² can still generate a profitable strategy (if it
correctly predicts direction), and vice versa.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===================================================================
# A. ML PREDICTION METRICS
# ===================================================================
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = "") -> dict:
    """
    Compute standard regression evaluation metrics.

    Parameters
    ----------
    y_true     : Ground-truth target values.
    y_pred     : Model predictions.
    model_name : Optional label for display.

    Returns
    -------
    dict with keys: MAE, RMSE, R2, Directional_Accuracy
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: % of cases where sign(pred) == sign(actual)
    correct_direction = np.sign(y_pred) == np.sign(y_true)
    dir_acc = correct_direction.mean() * 100.0

    metrics = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy_%": dir_acc,
    }
    return metrics


def compare_models(results_dict: dict, split: str = "test") -> pd.DataFrame:
    """
    Build a comparison table of all model metrics.

    Parameters
    ----------
    results_dict : dict
        Output of models.train_and_evaluate_all().
        Keys are model names; values contain 'val_metrics' and 'test_metrics'.
    split : str
        'val' or 'test'.

    Returns
    -------
    pd.DataFrame sorted by RMSE ascending.
    """
    rows = []
    for name, res in results_dict.items():
        m = res.get(f"{split}_metrics", {})
        rows.append({
            "Model": name,
            "MAE": m.get("MAE", np.nan),
            "RMSE": m.get("RMSE", np.nan),
            "R²": m.get("R2", np.nan),
        })
    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return df


# ===================================================================
# B. FINANCIAL PERFORMANCE METRICS
# ===================================================================
def compute_financial_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.04,
    trading_days_per_year: int = 252,
    strategy_name: str = "ML Strategy",
) -> dict:
    """
    Compute comprehensive financial performance and risk metrics.

    Parameters
    ----------
    strategy_returns   : Daily strategy returns (as fractions, e.g. 0.01 = 1%).
                         These should be the daily P&L divided by portfolio value.
    benchmark_returns  : Optional daily benchmark returns for comparison.
    risk_free_rate     : Annualised risk-free rate (default 4% = 0.04).
    trading_days_per_year : Number of trading days per year (default 252).
    strategy_name      : Label for display.

    Returns
    -------
    dict of computed metrics.
    """
    r = pd.Series(strategy_returns).dropna()

    if len(r) == 0:
        return {"Strategy": strategy_name, "Error": "No returns data"}

    # ---- Cumulative / Total Return ----
    cumulative = (1 + r).cumprod()
    total_return = cumulative.iloc[-1] - 1.0

    # ---- Annualised Return (CAGR) ----
    n_years = len(r) / trading_days_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # ---- Annualised Volatility ----
    ann_vol = r.std() * np.sqrt(trading_days_per_year)

    # ---- Sharpe Ratio ----
    # Sharpe = (Mean daily return - daily risk-free) / Std daily return
    # Annualised by multiplying by sqrt(252)
    daily_rf = risk_free_rate / trading_days_per_year
    excess_returns = r - daily_rf
    sharpe = (excess_returns.mean() / r.std()) * np.sqrt(trading_days_per_year) \
        if r.std() > 0 else 0.0

    # ---- Sortino Ratio ----
    # Like Sharpe but only penalises downside volatility
    downside = r[r < daily_rf] - daily_rf
    downside_std = downside.std() * np.sqrt(trading_days_per_year) \
        if len(downside) > 1 else np.nan
    sortino = (cagr - risk_free_rate) / downside_std \
        if (downside_std and downside_std > 0) else 0.0

    # ---- Maximum Drawdown ----
    # MDD = max(peak - trough) / peak, expressed as a percentage
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()  # most negative value

    # ---- Calmar Ratio ----
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # ---- Win Rate & Profit Factor ----
    winning_days = r[r > 0]
    losing_days = r[r < 0]
    win_rate = len(winning_days) / len(r) * 100 if len(r) > 0 else 0.0
    total_gains = winning_days.sum()
    total_losses = abs(losing_days.sum())
    profit_factor = total_gains / total_losses if total_losses > 0 else np.inf

    metrics = {
        "Strategy": strategy_name,
        "Total_Return_%": round(total_return * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Ann_Volatility_%": round(ann_vol * 100, 2),
        "Sharpe_Ratio": round(sharpe, 3),
        "Sortino_Ratio": round(sortino, 3),
        "Max_Drawdown_%": round(max_drawdown * 100, 2),
        "Calmar_Ratio": round(calmar, 3),
        "Win_Rate_%": round(win_rate, 2),
        "Profit_Factor": round(profit_factor, 3),
        "N_Trading_Days": len(r),
    }
    return metrics


def get_drawdown_series(strategy_returns: pd.Series) -> pd.Series:
    """
    Compute the drawdown series (fraction below running peak).

    Parameters
    ----------
    strategy_returns : Daily returns.

    Returns
    -------
    pd.Series of drawdown fractions (all <= 0).
    """
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown


def print_metrics_table(metrics_list: list):
    """
    Pretty-print a list of metric dictionaries as a formatted table.

    Parameters
    ----------
    metrics_list : list of dict (one dict per strategy).
    """
    df = pd.DataFrame(metrics_list)
    # Move 'Strategy' to first column
    cols = ["Strategy"] + [c for c in df.columns if c != "Strategy"]
    df = df[cols]
    print("\n" + "=" * 90)
    print("  STRATEGY PERFORMANCE COMPARISON")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)


# ===================================================================
# C. MODEL COMPARISON TABLE
# ===================================================================
def build_model_comparison_table(
    classical_results: dict,
    dl_metrics_list: list = None,
    target_name: str = "Return",
    split: str = "test",
) -> pd.DataFrame:
    """
    Build a unified model comparison table including both classical ML
    and Deep Learning results.

    Parameters
    ----------
    classical_results : dict from models.train_and_evaluate_all().
    dl_metrics_list : list of dicts from compute_regression_metrics() for DL models.
    target_name : str (for display).
    split : 'val' or 'test'.

    Returns
    -------
    pd.DataFrame sorted by RMSE.
    """
    rows = []
    for name, res in classical_results.items():
        m = res.get(f"{split}_metrics", {})
        rows.append({
            "Model": name,
            "Type": "Classical ML",
            "MAE": round(m.get("MAE", np.nan), 6),
            "RMSE": round(m.get("RMSE", np.nan), 6),
            "R²": round(m.get("R2", np.nan), 4),
        })

    if dl_metrics_list:
        for m in dl_metrics_list:
            rows.append({
                "Model": m.get("Model", "Unknown DL"),
                "Type": "Deep Learning",
                "MAE": round(m.get("MAE", np.nan), 6),
                "RMSE": round(m.get("RMSE", np.nan), 6),
                "R²": round(m.get("R2", np.nan), 4),
            })

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON TABLE  -  Target: {target_name}  |  Split: {split}")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}")
    return df


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    y_true = np.random.randn(n)
    y_pred = y_true * 0.6 + np.random.randn(n) * 0.5

    m = compute_regression_metrics(y_true, y_pred, model_name="TestModel")
    print("Regression metrics:", m)

    # Simulate daily returns
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    strat_returns = pd.Series(np.random.randn(n) * 0.01 + 0.0005, index=dates)
    bench_returns = pd.Series(np.random.randn(n) * 0.012 + 0.0004, index=dates)

    fm = compute_financial_metrics(strat_returns, bench_returns,
                                   strategy_name="TestStrategy")
    bm = compute_financial_metrics(bench_returns, strategy_name="Buy & Hold")
    print_metrics_table([fm, bm])
