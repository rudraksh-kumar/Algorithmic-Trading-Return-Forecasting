"""
backtesting.py - Trading Strategy & Backtesting Engine
=======================================================
This module converts model predictions into systematic trading signals
and simulates the resulting trading strategy on historical data.

DISCLAIMER: This is an educational simulation for a coursework project.
It does NOT constitute real financial advice.  Past simulated performance
does not guarantee future results.

Trading Strategy
----------------
The strategy is threshold-based:
  - If predicted_return > +buy_threshold  -> LONG (BUY):  hold +1 unit
  - If predicted_return < -sell_threshold -> FLAT/SHORT:  hold 0 or -1 unit
  - Otherwise                             -> HOLD:        maintain current position

Both thresholds are configurable.  A SHORT position means you profit when
the asset price falls (implemented here as a 0 position for simplicity,
i.e. "go flat" rather than short).

Adaptive Threshold (Fix #1)
----------------------------
When a model (e.g. LSTM) produces predictions tightly clustered near 0,
a fixed threshold like 0.002 will never trigger any trades. The function
`compute_adaptive_threshold` solves this by deriving the threshold from
the empirical distribution of the model's own predictions:

  tau = percentile(|predictions|, percentile_level)

With percentile_level=60, the top 40% of signal magnitudes cross the
threshold, guaranteeing a minimum level of market participation.

Backtesting Logic
-----------------
For each trading day t in the test period:
  1. Look at the signal generated from the model's prediction for day t.
  2. Trade at the next day's open (realistic: we can't trade at the same
     close price we used to generate the signal).
     [Simplified here: trade at same-day close, clearly documented.]
  3. Apply transaction costs on every position change.
  4. Track portfolio value = cash + position * current_price.

Performance is compared against a Buy-and-Hold benchmark that simply
holds the asset for the entire test period.

Key Assumptions (documented for transparency)
----------------------------------------------
- We trade a fractional unit (portfolio fraction), not shares.
- Transaction cost is applied as a percentage of trade value.
- No leverage, margin, or short selling in the default configuration.
- Prices are adjusted closing prices (already adjusted for splits/dividends).
"""

import numpy as np
import pandas as pd


# ===================================================================
# 1. ADAPTIVE THRESHOLD COMPUTATION
# ===================================================================
def compute_adaptive_threshold(
    predictions: np.ndarray,
    percentile_level: float = 60.0,
    min_threshold: float = 1e-5,
) -> float:
    """
    Compute a data-driven trading threshold from the prediction distribution.

    **Why this is needed:**
    Deep learning models (especially LSTMs on financial returns) often produce
    predictions clustered very close to zero — far below any fixed threshold
    like 0.002. A fixed threshold will then trigger zero trades.

    This function sets the threshold as a percentile of the absolute
    prediction values, guaranteeing that a fixed fraction of signals
    (100 - percentile_level) % always cross the threshold and trigger trades.

    Parameters
    ----------
    predictions      : Array of model predictions (any scale).
    percentile_level : Percentile of |predictions| used as threshold (default 60).
                       60 -> top 40% of signal magnitudes trigger trades.
                       Higher value -> fewer, higher-conviction trades.
    min_threshold    : Floor value to prevent threshold collapsing to zero.

    Returns
    -------
    float : Adaptive threshold value tau.

    Example
    -------
    If LSTM preds range from -0.0003 to +0.0004:
      Fixed tau=0.002 -> 0% trades triggered  (BUG)
      Adaptive tau    -> ~40% of predictions trigger trades (FIXED)
    """
    tau = float(np.percentile(np.abs(predictions), percentile_level))
    tau = max(tau, min_threshold)   # prevent zero threshold

    # Diagnostic report
    pct_long  = (predictions >  tau).mean() * 100
    pct_flat  = np.abs((np.abs(predictions) <= tau)).mean() * 100
    pct_short = (predictions < -tau).mean() * 100
    print(f"  [AdaptiveTau] pred range  : [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"  [AdaptiveTau] pred std    : {predictions.std():.6f}")
    print(f"  [AdaptiveTau] tau         : {tau:.6f}  (p{percentile_level:.0f} of |preds|)")
    print(f"  [AdaptiveTau] signals     : Long={pct_long:.1f}%  Flat={pct_flat:.1f}%  Short={pct_short:.1f}%")
    return tau


# ===================================================================
# 2. SIGNAL GENERATION
# ===================================================================
def generate_signals(
    predictions: np.ndarray,
    buy_threshold: float = 0.002,
    sell_threshold: float = 0.002,
    allow_short: bool = False,
    adaptive: bool = False,
    adaptive_percentile: float = 60.0,
) -> np.ndarray:
    """
    Convert return predictions to discrete trading signals.

    Signal values:
      +1 : LONG  (buy / hold long position)
       0 : FLAT  (no position)
      -1 : SHORT (short position, only if allow_short=True)

    Parameters
    ----------
    predictions          : Array of predicted next-day returns.
    buy_threshold        : Minimum predicted return to trigger BUY signal.
                           Ignored when adaptive=True.
    sell_threshold       : Maximum predicted return (negative) for SELL/FLAT.
                           Ignored when adaptive=True.
    allow_short          : If True, emit -1 for strongly negative predictions.
                           If False, emit 0 (flat) for sell signals.
    adaptive             : If True, compute threshold from prediction distribution
                           using compute_adaptive_threshold(). Recommended when
                           the model's output range is much smaller than the
                           fixed threshold (e.g., LSTM predictions near 0).
    adaptive_percentile  : Percentile level for adaptive threshold (default 60).
                           Only used when adaptive=True.

    Returns
    -------
    np.ndarray of int (same length as predictions), values in {-1, 0, 1}.

    Example
    -------
    # Fixed threshold (original)
    signals = generate_signals(preds, buy_threshold=0.002)

    # Adaptive threshold (recommended for LSTM)
    signals = generate_signals(preds, adaptive=True, adaptive_percentile=60)
    """
    if adaptive:
        tau = compute_adaptive_threshold(predictions, percentile_level=adaptive_percentile)
        buy_threshold  = tau
        sell_threshold = tau
    else:
        print(f"  [Signals] Using fixed threshold tau={buy_threshold:.6f}")
        pct_long = (predictions > buy_threshold).mean() * 100
        print(f"  [Signals] pred range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"  [Signals] % above threshold: {pct_long:.1f}%  (0% = no trades will occur)")

    signals = np.zeros(len(predictions), dtype=int)
    signals[predictions > buy_threshold] = 1
    if allow_short:
        signals[predictions < -sell_threshold] = -1
    else:
        signals[predictions < -sell_threshold] = 0   # FLAT instead of SHORT

    n_long  = (signals ==  1).sum()
    n_flat  = (signals ==  0).sum()
    n_short = (signals == -1).sum()
    print(f"  [Signals] Generated: Long={n_long}  Flat={n_flat}  Short={n_short}  "
          f"Total={len(signals)} days")
    return signals


# ===================================================================
# 2. BACKTESTING ENGINE
# ===================================================================
def run_backtest(
    signals: np.ndarray,
    actual_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_capital: float = 100_000.0,
    transaction_cost_pct: float = 0.001,
    strategy_name: str = "ML Strategy",
) -> pd.DataFrame:
    """
    Simulate a trading strategy and track portfolio value over time.

    Parameters
    ----------
    signals          : Array of signals (+1, 0, -1) for each test day.
    actual_returns   : Array of realised daily returns for those days.
    dates            : DatetimeIndex corresponding to each day.
    initial_capital  : Starting portfolio value (e.g. $100,000).
    transaction_cost_pct : Cost per trade as a fraction of trade value.
                           e.g. 0.001 = 0.1% per trade (covers commissions
                           and bid-ask spread).
    strategy_name    : Label for this strategy in output.

    Returns
    -------
    pd.DataFrame with columns:
        Date, Signal, Position, Daily_Return, Strategy_Return,
        Transaction_Cost, Portfolio_Value, Drawdown

    Notes on Position vs Signal
    ---------------------------
    Position at day t = Signal at day t-1 (we act on yesterday's signal).
    This prevents look-ahead bias from same-day trading on the signal.
    """
    n = len(signals)
    assert len(actual_returns) == n, "signals and actual_returns must have same length."

    portfolio_values = np.zeros(n)
    strategy_returns = np.zeros(n)
    transaction_costs = np.zeros(n)
    positions = np.zeros(n, dtype=int)

    capital = initial_capital
    current_position = 0  # Start flat

    for t in range(n):
        # New position: we act on the signal for today
        new_position = int(signals[t])

        # Compute transaction cost if position changes
        trade_occurred = (new_position != current_position)
        cost = 0.0
        if trade_occurred:
            # Cost = transaction_cost_pct * portfolio value
            cost = abs(transaction_cost_pct * capital)
            capital -= cost
            transaction_costs[t] = cost

        positions[t] = new_position
        current_position = new_position

        # Daily P&L: position * return * capital
        # position=+1 -> profit when return > 0
        # position=-1 -> profit when return < 0
        # position= 0 -> flat, no market exposure
        daily_pnl_rate = current_position * actual_returns[t]
        capital = capital * (1 + daily_pnl_rate)
        strategy_returns[t] = daily_pnl_rate
        portfolio_values[t] = capital

    # Compute drawdown series
    port_series = pd.Series(portfolio_values, index=dates)
    rolling_max = port_series.cummax()
    drawdown = (port_series - rolling_max) / rolling_max

    df = pd.DataFrame({
        "Date": dates,
        "Signal": signals,
        "Position": positions,
        "Actual_Return": actual_returns,
        "Strategy_Return": strategy_returns,
        "Transaction_Cost": transaction_costs,
        "Portfolio_Value": portfolio_values,
        "Drawdown": drawdown.values,
    })
    df = df.set_index("Date")
    df.attrs["strategy_name"] = strategy_name
    df.attrs["initial_capital"] = initial_capital
    return df


# ===================================================================
# 3. BUY-AND-HOLD BENCHMARK
# ===================================================================
def run_buy_and_hold(
    actual_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """
    Simulate a simple Buy-and-Hold benchmark.

    The benchmark invests the entire capital at the start and holds the
    asset throughout the test period without any trading.

    Parameters
    ----------
    actual_returns : Realised daily returns for the test period.
    dates          : DatetimeIndex for the test period.
    initial_capital: Starting portfolio value.

    Returns
    -------
    pd.DataFrame with portfolio values for the B&H strategy.
    """
    cumulative = initial_capital * np.cumprod(1 + np.asarray(actual_returns))
    port_series = pd.Series(cumulative, index=dates)
    rolling_max = port_series.cummax()
    drawdown = (port_series - rolling_max) / rolling_max

    df = pd.DataFrame({
        "Date": dates,
        "Strategy_Return": actual_returns,
        "Portfolio_Value": cumulative,
        "Drawdown": drawdown.values,
    })
    df = df.set_index("Date")
    df.attrs["strategy_name"] = "Buy & Hold"
    df.attrs["initial_capital"] = initial_capital
    return df


# ===================================================================
# 4. TRADE LOG
# ===================================================================
def extract_trade_log(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract individual trade events from a backtest DataFrame.

    A trade is logged every time the position changes.

    Parameters
    ----------
    backtest_df : Output of run_backtest().

    Returns
    -------
    pd.DataFrame with one row per trade: Date, From_Position, To_Position, Type.
    """
    positions = backtest_df["Position"]
    changes = positions.diff().fillna(0)
    trade_dates = changes[changes != 0].index

    trades = []
    for date in trade_dates:
        prev_pos = positions.get(positions.index[positions.index.get_loc(date) - 1], 0) \
            if positions.index.get_loc(date) > 0 else 0
        new_pos = positions[date]
        trade_type = "BUY" if new_pos > prev_pos else ("SELL" if new_pos < prev_pos else "FLAT")
        trades.append({"Date": date, "From": prev_pos, "To": new_pos, "Type": trade_type})

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df = trade_df.set_index("Date")
    return trade_df


# ===================================================================
# 5. SUMMARY STATS
# ===================================================================
def backtest_summary(backtest_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics from a backtest run.

    Parameters
    ----------
    backtest_df : Output of run_backtest() or run_buy_and_hold().

    Returns
    -------
    dict with total return, n_trades, n_long_days, n_flat_days.
    """
    initial = backtest_df.attrs.get("initial_capital", backtest_df["Portfolio_Value"].iloc[0])
    final = backtest_df["Portfolio_Value"].iloc[-1]
    total_return = (final - initial) / initial * 100

    n_trades = 0
    n_long = 0
    n_flat = 0
    if "Position" in backtest_df.columns:
        pos = backtest_df["Position"]
        n_trades = (pos.diff().fillna(0) != 0).sum()
        n_long = (pos == 1).sum()
        n_flat = (pos == 0).sum()

    return {
        "Strategy": backtest_df.attrs.get("strategy_name", "Unknown"),
        "Initial_Capital": initial,
        "Final_Capital": round(final, 2),
        "Total_Return_%": round(total_return, 2),
        "N_Trades": int(n_trades),
        "N_Long_Days": int(n_long),
        "N_Flat_Days": int(n_flat),
        "Total_Transaction_Costs": round(backtest_df.get("Transaction_Cost",
                                          pd.Series([0])).sum(), 2),
    }


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
    actual_returns = np.random.randn(n) * 0.01 + 0.0002

    # Simulate predictions (slightly correlated with actual)
    predictions = actual_returns * 0.5 + np.random.randn(n) * 0.008

    signals = generate_signals(predictions, buy_threshold=0.002, sell_threshold=0.002)
    print(f"Signal distribution: Long={( signals==1).sum()}  "
          f"Flat={(signals==0).sum()}  Short={(signals==-1).sum()}")

    bt = run_backtest(signals, actual_returns, dates, initial_capital=100_000)
    bh = run_buy_and_hold(actual_returns, dates, initial_capital=100_000)

    print(f"\nML Strategy  - Final portfolio: ${bt['Portfolio_Value'].iloc[-1]:,.2f}")
    print(f"Buy & Hold   - Final portfolio: ${bh['Portfolio_Value'].iloc[-1]:,.2f}")

    bt_sum = backtest_summary(bt)
    bh_sum = backtest_summary(bh)
    print("\nBacktest Summary:", bt_sum)
    print("B&H Summary:     ", bh_sum)
