"""
data_loader.py - Data Ingestion Module
=======================================
This module is responsible for downloading historical daily stock-market data
and providing a reproducible data pipeline.

Data Sources
------------
- **Primary**: Yahoo Finance via the `yfinance` library.
  We fetch Open, High, Low, Close, Adj Close, and Volume for a given ticker
  (e.g. "^GSPC" for S&P 500, "^NSEI" for NIFTY 50).

- **Fallback**: A synthetic data generator based on Geometric Brownian Motion
  (GBM).  This ensures the project can run even without internet access.
  GBM models stock prices as:
      S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
  where Z ~ N(0,1).  The generated data mimics realistic daily OHLCV structure.

Usage
-----
    from src.data_loader import load_data
    df = load_data(ticker="^GSPC", start="2010-01-01", end="2024-12-31")
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "^GSPC"          # S&P 500 index
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2024-12-31"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "^GSPC", "^NSEI", "AAPL").
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with columns:
        [Open, High, Low, Close, Adj Close, Volume].
    """
    import yfinance as yf
    print(f"[DataLoader] Downloading {ticker} data from Yahoo Finance ({start} to {end})...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check ticker symbol and date range.")

    # yfinance may return MultiIndex columns when downloading a single ticker;
    # flatten them to simple column names.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure the index is a DatetimeIndex named 'Date'
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in expected_cols:
        if col not in data.columns:
            raise ValueError(f"Expected column '{col}' not found in downloaded data.")

    data = data[expected_cols]
    print(f"[DataLoader] Downloaded {len(data)} rows for {ticker}.")
    return data


def generate_synthetic_data(
    num_days: int = 3750,
    start_date: str = "2010-01-01",
    initial_price: float = 1100.0,
    annual_return: float = 0.08,
    annual_volatility: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV data using Geometric Brownian Motion.

    This fallback ensures full reproducibility even without internet access.
    The generated data mimics realistic market structure (open/high/low/close
    relationships, volume patterns) but does NOT represent any real asset.

    Parameters
    ----------
    num_days : int
        Number of trading days to generate (~252 per year).
    start_date : str
        Start date for the synthetic time series.
    initial_price : float
        Starting price level.
    annual_return : float
        Expected annualised drift (mu).
    annual_volatility : float
        Expected annualised volatility (sigma).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Date with columns:
        [Open, High, Low, Close, Adj Close, Volume].

    Notes
    -----
    Geometric Brownian Motion (GBM):
        S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1) and dt = 1/252 (one trading day).
    """
    np.random.seed(seed)
    dt = 1.0 / 252.0  # one trading day expressed as a fraction of a year
    mu = annual_return
    sigma = annual_volatility

    # Generate close prices via GBM
    daily_shocks = np.random.normal(0, 1, num_days)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * daily_shocks
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    close_prices = np.exp(log_prices)

    # Construct realistic OHLC relationships
    # Open ≈ previous close with small overnight gap
    open_prices = np.empty(num_days)
    open_prices[0] = initial_price
    open_prices[1:] = close_prices[:-1] * (1 + np.random.normal(0, 0.001, num_days - 1))

    # High = max(open, close) + small positive deviation
    high_dev = np.abs(np.random.normal(0, 0.005, num_days)) * close_prices
    high_prices = np.maximum(open_prices, close_prices) + high_dev

    # Low = min(open, close) - small positive deviation
    low_dev = np.abs(np.random.normal(0, 0.005, num_days)) * close_prices
    low_prices = np.minimum(open_prices, close_prices) - low_dev

    # Volume: base level with some randomness and a slight trend
    base_volume = 3_000_000_000
    volume = (base_volume * np.abs(1 + np.random.normal(0, 0.3, num_days))).astype(int)

    # Build date index (business days only)
    dates = pd.bdate_range(start=start_date, periods=num_days, freq="B")

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Adj Close": close_prices,  # No splits/dividends in synthetic data
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    print(f"[DataLoader] Generated {num_days} days of synthetic GBM data.")
    return df


def load_data(
    ticker: str = DEFAULT_TICKER,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    use_cache: bool = True,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Main entry point: load historical stock data (real or synthetic).

    Workflow:
    1. If a cached CSV exists in data/ and use_cache=True, load from disk.
    2. Otherwise, try downloading from Yahoo Finance.
    3. If download fails (no internet, bad ticker), fall back to synthetic data.
    4. Cache the result to data/raw_data.csv.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    use_cache : bool
        If True, return cached data when available.
    force_synthetic : bool
        If True, skip the API call and generate synthetic data directly.

    Returns
    -------
    pd.DataFrame
        Historical OHLCV DataFrame indexed by Date.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, "raw_data.csv")

    # 1. Try cache
    if use_cache and os.path.exists(cache_path) and not force_synthetic:
        print(f"[DataLoader] Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        print(f"[DataLoader] Loaded {len(df)} rows from cache.")
        return df

    # 2. Try Yahoo Finance
    if not force_synthetic:
        try:
            df = fetch_yfinance(ticker, start, end)
            df.to_csv(cache_path)
            print(f"[DataLoader] Data cached to {cache_path}")
            return df
        except Exception as e:
            print(f"[DataLoader] Yahoo Finance download failed: {e}")
            print("[DataLoader] Falling back to synthetic data generation...")

    # 3. Synthetic fallback
    df = generate_synthetic_data()
    df.to_csv(cache_path)
    print(f"[DataLoader] Synthetic data cached to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print("\nData shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())
