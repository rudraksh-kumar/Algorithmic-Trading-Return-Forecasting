"""
preprocessing.py - Data Cleaning & Preparation Module
======================================================
This module handles all data cleaning, target variable construction, and
chronological train/validation/test splitting.

Key Responsibilities
--------------------
1. **Missing-value handling**: Forward-fill small gaps, drop remaining NaNs.
2. **Duplicate removal**: Drop duplicate dates if any.
3. **Chronological sorting**: Ensure data is ordered by date (oldest first).
4. **Target construction**:
   - Daily simple return:  R_t = (Close_t - Close_{t-1}) / Close_{t-1}
   - Daily log return:     r_t = ln(Close_t / Close_{t-1})
   - Rolling historical volatility (annualised):
       vol_t = std(R_{t-window+1:t}) * sqrt(252)
5. **Time-series splitting** (NO random shuffling):
   - Training set   : first 70% of observations
   - Validation set : next 15%
   - Test set       : final 15%
6. **Feature scaling**: StandardScaler fit ONLY on training data, then
   applied (transform) to validation and test sets to prevent data leakage.

Why NOT randomly split time-series data?
-----------------------------------------
Random splitting would allow future data points to leak into the training
set.  A model trained on 2023 data and tested on 2020 data has an unfair
advantage because the "test" observations chronologically precede the
training observations.  In production, we can never see future data, so
our evaluation protocol must respect temporal ordering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OHLCV data.

    Steps:
    1. Sort by date (ascending / chronological order).
    2. Remove duplicate date entries (keep last).
    3. Forward-fill missing values (up to 5 consecutive NaNs).
    4. Drop any remaining rows with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame indexed by Date.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort chronologically
    df = df.sort_index(ascending=True)

    # Remove duplicate dates
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        print(f"[Preprocessing] Removing {n_dupes} duplicate date entries.")
        df = df[~df.index.duplicated(keep="last")]

    # Forward-fill small gaps (e.g. holidays with partial data)
    df = df.ffill(limit=5)

    # Drop any remaining NaN rows
    n_missing_before = df.isna().sum().sum()
    df = df.dropna()
    n_missing_after = n_missing_before - 0  # all dropped
    if n_missing_before > 0:
        print(f"[Preprocessing] Dropped rows with missing values ({n_missing_before} NaN cells removed).")

    print(f"[Preprocessing] Cleaned data: {len(df)} rows, {df.shape[1]} columns.")
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from the Close price.

    New Columns Added
    -----------------
    - ``Daily_Return``  : Simple daily percentage return
          R_t = (Close_t - Close_{t-1}) / Close_{t-1}
    - ``Log_Return``    : Logarithmic daily return
          r_t = ln(Close_t / Close_{t-1})

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with return columns appended.
    """
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Calculate rolling historical volatility (annualised).

    Formula
    -------
    vol_t = std(R_{t-window+1}, ..., R_t) * sqrt(252)

    The standard deviation is computed over the most recent ``window``
    daily returns and then annualised by multiplying by sqrt(252)
    (approximately 252 trading days per year).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains a 'Daily_Return' column.
    window : int
        Rolling window size (default 21 ≈ one trading month).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Volatility_21d' column appended.
    """
    df = df.copy()
    col_name = f"Volatility_{window}d"
    df[col_name] = df["Daily_Return"].rolling(window=window).std() * np.sqrt(252)
    return df


def construct_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the prediction targets.

    We predict the NEXT trading day's return and volatility.  To avoid
    data leakage the targets are shifted forward:
        Target_Return_t     = Daily_Return_{t+1}    (tomorrow's return)
        Target_Volatility_t = Volatility_21d_{t+1}  (tomorrow's realised vol)

    The final row will have NaN targets and is dropped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Daily_Return' and 'Volatility_21d'.

    Returns
    -------
    pd.DataFrame
        DataFrame with target columns appended (NaN-tail rows dropped).
    """
    df = df.copy()
    df["Target_Return"] = df["Daily_Return"].shift(-1)
    if "Volatility_21d" in df.columns:
        df["Target_Volatility"] = df["Volatility_21d"].shift(-1)
    return df


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    """
    Chronological train / validation / test split.

    Unlike random splitting (which is standard for i.i.d. data), time-series
    data MUST be split in temporal order to prevent look-ahead bias.

    Default proportions:  Train 70% | Validation 15% | Test 15%

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset sorted chronologically.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"[Preprocessing] Split sizes  ->  Train: {len(train_df)}  |  "
          f"Val: {len(val_df)}  |  Test: {len(test_df)}")
    print(f"[Preprocessing] Train period : {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"[Preprocessing] Val period   : {val_df.index[0].date()} to {val_df.index[-1].date()}")
    print(f"[Preprocessing] Test period  : {test_df.index[0].date()} to {test_df.index[-1].date()}")
    return train_df, val_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Standardise features using z-score normalisation.

    CRITICAL: The scaler is fit ONLY on the training set.  Validation and
    test sets are transformed using the training-set statistics (mean, std).
    This prevents any information from the future leaking into the model
    through the scaling parameters.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Chronologically split DataFrames.
    feature_cols : list of str
        Column names to scale.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler)
        Scaled copies of (train, val, test) and the fitted scaler object.
    """
    scaler = StandardScaler()

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    # Fit on training data ONLY
    scaler.fit(train_df[feature_cols])

    # Transform all splits
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    print(f"[Preprocessing] Scaled {len(feature_cols)} features (fit on train only).")
    return train_scaled, val_scaled, test_scaled, scaler


def preprocess_pipeline(df: pd.DataFrame, volatility_window: int = 21) -> dict:
    """
    Run the full preprocessing pipeline end-to-end.

    Steps executed in order:
    1. Clean data
    2. Compute returns
    3. Compute rolling volatility
    4. Construct next-day targets
    5. Drop rows with NaN (from rolling windows / target shift)

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data.
    volatility_window : int
        Window for rolling volatility calculation.

    Returns
    -------
    dict with keys:
        'processed_df' : pd.DataFrame - full processed dataset (before split)
    """
    df = clean_data(df)
    df = compute_returns(df)
    df = compute_volatility(df, window=volatility_window)
    df = construct_targets(df)

    # Drop rows where rolling window or target shift produced NaNs
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"[Preprocessing] Dropped {n_before - n_after} rows with NaN "
          f"(from rolling windows / target shift). Remaining: {n_after} rows.")

    return {"processed_df": df}


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_data

    raw = load_data()
    result = preprocess_pipeline(raw)
    df = result["processed_df"]
    print("\nProcessed data shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nTarget statistics:")
    print(df[["Target_Return", "Target_Volatility"]].describe())
