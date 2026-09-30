"""
features.py - Technical Indicator & Feature Engineering Module
===============================================================
This module computes a comprehensive set of technical indicators from raw
OHLCV + return data.  Every feature is calculated using ONLY past and present
data (no look-ahead bias).

Feature Categories
------------------
1. **Lagged Returns & Volatility**
   Past values of daily return and rolling volatility at lags 1, 2, 3, 5.

2. **Moving Averages & Ratios**
   Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
   at various windows, plus price-to-SMA ratios.

3. **Price Range**
   Normalised daily high-low range as a fraction of the close price.

4. **Volume Indicators**
   Daily volume percentage change and volume-to-SMA ratio.

5. **Momentum - Rate of Change (ROC)**
   Percentage price change over the last N days.

6. **Relative Strength Index (RSI)**
   Measures the speed and magnitude of recent price changes (0-100).

7. **MACD (Moving Average Convergence Divergence)**
   Difference between 12-day and 26-day EMAs, plus signal line.

8. **Bollinger Bands**
   Upper/lower bands at ±2 standard deviations from 20-day SMA,
   %B (position within bands), and bandwidth.

All formulas are documented in docstrings.
"""

import numpy as np
import pandas as pd


# ===================================================================
# 1. LAGGED FEATURES
# ===================================================================
def add_lagged_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    Create lagged versions of Daily_Return and Volatility_21d.

    A lag-k feature at time t contains the value from time t-k.
    This lets the model see recent historical patterns without
    leaking future information.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Daily_Return' and optionally 'Volatility_21d'.
    lags : list of int
        Lag periods to create (default: [1, 2, 3, 5]).

    Returns
    -------
    pd.DataFrame
        DataFrame with lag columns appended (e.g. 'Return_Lag_1').
    """
    if lags is None:
        lags = [1, 2, 3, 5]

    df = df.copy()
    for lag in lags:
        df[f"Return_Lag_{lag}"] = df["Daily_Return"].shift(lag)
        if "Volatility_21d" in df.columns:
            df[f"Vol_Lag_{lag}"] = df["Volatility_21d"].shift(lag)
    return df


# ===================================================================
# 2. MOVING AVERAGES & RATIOS
# ===================================================================
def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Simple Moving Averages (SMA) and price-to-SMA ratios.

    SMA_n = (1/n) * sum(Close_{t-n+1}, ..., Close_t)

    Price-to-SMA ratios capture whether the current price is above or
    below its recent trend:
        Ratio = Close_t / SMA_n_t

    Also computes Exponential Moving Averages (EMA) for 12 and 26 days
    (used later in MACD calculation).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA, EMA, and ratio columns.
    """
    df = df.copy()
    for window in [5, 10, 20, 50]:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"Close_to_SMA_{window}"] = df["Close"] / df[f"SMA_{window}"]

    # Exponential Moving Averages (for MACD and general trend)
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    return df


# ===================================================================
# 3. PRICE RANGE
# ===================================================================
def add_price_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the normalised daily high-low price range.

    HL_Range_t = (High_t - Low_t) / Close_t

    This captures intraday volatility: a wider range typically indicates
    higher uncertainty or momentum.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'High', 'Low', and 'Close'.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'HL_Range' column.
    """
    df = df.copy()
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
    return df


# ===================================================================
# 4. VOLUME INDICATORS
# ===================================================================
def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume-based features.

    Volume_Change = (Volume_t - Volume_{t-1}) / Volume_{t-1}
        Measures the percentage change in trading volume.

    Volume_SMA_Ratio = Volume_t / SMA(Volume, 20)
        Values > 1 indicate above-average volume (unusual activity).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Volume'.

    Returns
    -------
    pd.DataFrame
        DataFrame with volume feature columns.
    """
    df = df.copy()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_SMA_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
    return df


# ===================================================================
# 5. MOMENTUM - RATE OF CHANGE (ROC)
# ===================================================================
def add_momentum(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """
    Compute Rate of Change (ROC) momentum indicator.

    ROC_n_t = (Close_t - Close_{t-n}) / Close_{t-n} * 100

    ROC measures the percentage change in price over the last n days.
    Positive ROC indicates upward momentum; negative indicates downward.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close'.
    periods : list of int
        Look-back periods (default: [5, 10, 20]).

    Returns
    -------
    pd.DataFrame
        DataFrame with ROC columns.
    """
    if periods is None:
        periods = [5, 10, 20]

    df = df.copy()
    for p in periods:
        df[f"ROC_{p}"] = ((df["Close"] - df["Close"].shift(p)) / df["Close"].shift(p)) * 100
    return df


# ===================================================================
# 6. RELATIVE STRENGTH INDEX (RSI)
# ===================================================================
def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute the Relative Strength Index (RSI).

    RSI is a momentum oscillator that measures the speed and change of
    price movements on a scale of 0 to 100.

    Calculation steps:
    1. Compute daily price changes:  delta_t = Close_t - Close_{t-1}
    2. Separate gains (delta > 0) and losses (delta < 0).
    3. Compute exponential moving average of gains and losses over
       the specified window:
           avg_gain = EMA(gains, window)
           avg_loss = EMA(|losses|, window)
    4. Relative Strength:  RS = avg_gain / avg_loss
    5. RSI = 100 - 100 / (1 + RS)

    Interpretation:
    - RSI > 70 -> overbought (possible reversal down)
    - RSI < 30 -> oversold (possible reversal up)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close'.
    window : int
        Look-back period (default: 14 days).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'RSI_{window}' column.
    """
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


# ===================================================================
# 7. MACD (Moving Average Convergence Divergence)
# ===================================================================
def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the MACD indicator and its signal line.

    MACD = EMA(Close, 12) - EMA(Close, 26)
    Signal Line = EMA(MACD, 9)
    MACD Histogram = MACD - Signal Line

    MACD crossing above the signal line is considered bullish;
    crossing below is bearish.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'EMA_12' and 'EMA_26' (from add_moving_averages).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'MACD', 'MACD_Signal', and 'MACD_Hist' columns.
    """
    df = df.copy()
    # Ensure EMAs exist
    if "EMA_12" not in df.columns:
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    if "EMA_26" not in df.columns:
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


# ===================================================================
# 8. BOLLINGER BANDS
# ===================================================================
def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Bollinger Bands consist of:
    - Middle Band = SMA(Close, window)
    - Upper Band  = Middle Band + num_std * std(Close, window)
    - Lower Band  = Middle Band - num_std * std(Close, window)

    Derived features:
    - %B = (Close - Lower Band) / (Upper Band - Lower Band)
      Measures where the price sits relative to the bands (0 = at lower, 1 = at upper).
    - Bandwidth = (Upper - Lower) / Middle
      Measures band width (higher = more volatile).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close'.
    window : int
        Rolling window (default: 20).
    num_std : float
        Number of standard deviations (default: 2.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with Bollinger Band columns.
    """
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()

    df["BB_Upper"] = sma + num_std * std
    df["BB_Lower"] = sma - num_std * std
    df["BB_Middle"] = sma
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    return df


# ===================================================================
# MASTER FUNCTION - Build all features
# ===================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    This is the main entry point for feature construction.  It calls each
    individual feature function and returns the enriched DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with at minimum:
        Open, High, Low, Close, Volume, Daily_Return, Volatility_21d.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame (NaN rows from rolling windows NOT yet dropped).
    """
    df = add_lagged_features(df)
    df = add_moving_averages(df)
    df = add_price_range(df)
    df = add_volume_features(df)
    df = add_momentum(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)

    print(f"[Features] Built {df.shape[1]} total columns (including targets & raw price).")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature column names (excluding raw OHLCV, targets,
    and intermediate columns not suitable as model inputs).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched DataFrame.

    Returns
    -------
    list of str
        Column names to use as model input features.
    """
    # Columns to EXCLUDE from feature matrix
    exclude = {
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "Daily_Return", "Log_Return",
        "Target_Return", "Target_Volatility",
        # Raw moving averages and EMAs (we use ratios instead)
        "SMA_5", "SMA_10", "SMA_20", "SMA_50",
        "EMA_12", "EMA_26",
        "Volume_SMA_20",
        # Bollinger raw bands (we use %B and bandwidth)
        "BB_Upper", "BB_Lower", "BB_Middle",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_pipeline

    raw = load_data()
    result = preprocess_pipeline(raw)
    df = result["processed_df"]

    # Need to re-add feature engineering on the processed data
    # (processed_df already has returns & volatility but not technical indicators)
    # Re-merge raw OHLCV needed for features
    raw_clean = raw.loc[df.index]
    merged = pd.concat([raw_clean[["Open", "High", "Low", "Close", "Volume"]], df], axis=1)
    # Remove duplicate columns if any
    merged = merged.loc[:, ~merged.columns.duplicated()]

    featured = build_features(merged)
    feature_cols = get_feature_columns(featured)

    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, c in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {c}")
    print(f"\nDataFrame shape after features: {featured.shape}")
    print("\nSample (last 3 rows):")
    print(featured[feature_cols].tail(3))
