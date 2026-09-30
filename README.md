# Stock Market Volatility Forecasting & Algorithmic Trading using Machine Learning
### *Master Comprehensive Study Guide & Technical Reference*

> **Educational ML Coursework Project** – This document serves as an exhaustive, first-principles academic study guide and engineering reference for financial time-series forecasting, neural network modeling (PyTorch LSTM), classical machine learning, and quantitative backtesting.

---

## 📑 Table of Contents
1. [Theoretical Foundations & Core Financial Principles](#1-theoretical-foundations--core-financial-principles)
   - 1.1 [The Efficient Market Hypothesis (EMH) vs. Machine Learning](#11-the-efficient-market-hypothesis-emh-vs-machine-learning)
   - 1.2 [Stationarity & The Random Walk Problem](#12-stationarity--the-random-walk-problem)
   - 1.3 [Why Raw Prices Fail: Simple Returns vs. Log Returns](#13-why-raw-prices-fail-simple-returns-vs-log-returns)
   - 1.4 [Realized Volatility, Clustering, & Annualization ($\sqrt{252}$)](#14-realized-volatility-clustering--annualization-sqrt252)
   - 1.5 [Geometric Brownian Motion (GBM) Data Generation](#15-geometric-brownian-motion-gbm-data-generation)
2. [Data Ingestion & Preprocessing Pipeline](#2-data-ingestion--preprocessing-pipeline)
   - 2.1 [Understanding OHLCV Data](#21-understanding-ohlcv-data)
   - 2.2 [Cleaning, Duplicate Handling, and Forward-Filling](#22-cleaning-duplicate-handling-and-forward-filling)
   - 2.3 [Target Construction & Temporal Alignment ($t+1$)](#23-target-construction--temporal-alignment-t1)
3. [Exhaustive Feature Engineering Reference](#3-exhaustive-feature-engineering-reference)
   - 3.1 [Momentum & Trend Indicators (SMA, EMA, Price Ratios, ROC)](#31-momentum--trend-indicators-sma-ema-price-ratios-roc)
   - 3.2 [Oscillators & Mean Reversion (RSI, MACD)](#32-oscillators--mean-reversion-rsi-macd)
   - 3.3 [Volatility & Range Indicators (Bollinger Bands, ATR, High-Low Range)](#33-volatility--range-indicators-bollinger-bands-atr-high-low-range)
   - 3.4 [Volume Dynamics (OBV, Volume SMA Ratio, Volume Change)](#34-volume-dynamics-obv-volume-sma-ratio-volume-change)
   - 3.5 [Statistical Properties (Rolling Volatilities, Skewness, Kurtosis, Autocorrelation)](#35-statistical-properties-rolling-volatilities-skewness-kurtosis-autocorrelation)
   - 3.6 [Summary Table of All 22 Engineered Features](#36-summary-table-of-all-22-engineered-features)
4. [Time-Series Validation & Avoiding Data Leakage](#4-time-series-validation--avoiding-data-leakage)
   - 4.1 [Why Standard K-Fold Cross Validation Destroys Financial ML](#41-why-standard-k-fold-cross-validation-destroys-financial-ml)
   - 4.2 [Chronological Splitting Architecture (70% Train, 15% Val, 15% Test)](#42-chronological-splitting-architecture-70-train-15-val-15-test)
   - 4.3 [Feature Normalization without Lookahead Bias](#43-feature-normalization-without-lookahead-bias)
   - 4.4 [Automated Feature Selection (Correlation Pruning & Importance Ranking)](#44-automated-feature-selection-correlation-pruning--importance-ranking)
5. [Classical Machine Learning Models](#5-classical-machine-learning-models)
   - 5.1 [Ridge Regression ($L_2$ Regularization)](#51-ridge-regression-l_2-regularization)
   - 5.2 [Random Forest Regressor (Bagging & Feature Subsampling)](#52-random-forest-regressor-bagging--feature-subsampling)
   - 5.3 [XGBoost Regressor (Gradient Boosted Decision Trees)](#53-xgboost-regressor-gradient-boosted-decision-trees)
6. [Deep Learning with PyTorch: Sequence Modeling (RNN, LSTM, GRU)](#6-deep-learning-with-pytorch-sequence-modeling-rnn-lstm-gru)
   - 6.1 [Vanilla RNNs, LSTMs, and GRUs (Architectural Differences)](#61-vanilla-rnns-lstms-and-grus-architectural-differences)
   - 6.2 [Detailed Anatomy of an LSTM Cell (Mathematical Gate Equations)](#62-detailed-anatomy-of-an-lstm-cell-mathematical-gate-equations)
   - 6.3 [Sliding Window Tensor Construction & Sequence Length Experiment](#63-sliding-window-tensor-construction--sequence-length-experiment)
   - 6.4 [Full PyTorch Network Architectures](#64-full-pytorch-network-architectures)
   - 6.5 [Training Loops, Gradient Clipping, Schedulers, & Early Stopping](#65-training-loops-gradient-clipping-schedulers--early-stopping)
7. [Algorithmic Trading & Quantitative Backtesting](#7-algorithmic-trading--quantitative-backtesting)
   - 7.1 [Trading Signal Generation Logic](#71-trading-signal-generation-logic)
   - 7.2 [Realistic Market Frictions: Transaction Costs & Slippage](#72-realistic-market-frictions-transaction-costs--slippage)
   - 7.3 [Portfolio Accounting & Equity Curve Evolution](#73-portfolio-accounting--equity-curve-evolution)
   - 7.4 [The Baseline: Buy-and-Hold Benchmark](#74-the-baseline-buy-and-hold-benchmark)
8. [Comprehensive Evaluation Metrics](#8-comprehensive-evaluation-metrics)
   - 8.1 [Machine Learning Regression Metrics (MSE, RMSE, MAE, MAPE, $R^2$)](#81-machine-learning-regression-metrics-mse-rmse-mae-mape-r2)
   - 8.2 [Directional & Classification Metrics (Hit Rate, Information Coefficient)](#82-directional--classification-metrics-hit-rate-information-coefficient)
   - 8.3 [Financial Performance & Risk-Adjusted Metrics (CAGR, Sharpe, Sortino, Max Drawdown, Calmar)](#83-financial-performance--risk-adjusted-metrics-cagr-sharpe-sortino-max-drawdown-calmar)
9. [Hyperparameter Tuning & Model Selection Framework](#9-hyperparameter-tuning--model-selection-framework)
   - 9.1 [Hyperparameter Search Strategy](#91-hyperparameter-search-strategy)
   - 9.2 [Time-Series Cross-Validation](#92-time-series-cross-validation)
   - 9.3 [Classical ML Hyperparameter Tuning](#93-classical-ml-hyperparameter-tuning)
   - 9.4 [Neural Network Hyperparameter Tuning](#94-neural-network-hyperparameter-tuning)
   - 9.5 [Early Stopping](#95-early-stopping)
   - 9.6 [Model Selection Using Validation Data](#96-model-selection-using-validation-data)
   - 9.7 [Final Evaluation on Held-Out Test Set](#97-final-evaluation-on-held-out-test-set)
10. [Project Codebase Architecture & File-by-File Breakdown](#10-project-codebase-architecture--file-by-file-breakdown)
11. [Quick Start & CLI Execution Guide](#11-quick-start--cli-execution-guide)
12. [Limitations, Pitfalls, & Future Improvements](#12-limitations-pitfalls--future-improvements)

---

# 🚀 End-to-End Project Flow
*A high-level overview of how data travels through this machine learning pipeline:*

1. **Data Ingestion (`src/data_loader.py`):** Downloads raw OHLCV (Open, High, Low, Close, Volume) stock data from Yahoo Finance, or generates synthetic Geometric Brownian Motion (GBM) data for offline testing.
2. **Preprocessing & Target Construction (`src/preprocessing.py`):** Cleans the data, fills missing values, and creates the target variable (shifting the *Next-Day Return* backward by 1 row to align with today's features).
3. **Feature Engineering (`src/features.py`):** Calculates 30+ complex technical indicators including Moving Averages, Momentum (RSI, MACD), Volatility (Bollinger Bands), and Volume ratios.
4. **Chronological Splitting (`src/preprocessing.py`):** Splits the timeline strictly forward-in-time (70% Train, 15% Validation, 15% Test) to guarantee zero Lookahead Bias.
5. **Automated Feature Selection (`src/feature_selection.py`):** Uses *only* the Training data to mathematically drop redundant features (>90% correlation) and ranks the remainder using a Random Forest, passing only the **Top 10** strongest signals to the neural networks.
6. **Feature Scaling (`src/preprocessing.py`):** Shrinks all features to a mean of 0 and standard deviation of 1. The scaler is fit strictly on the Training set to prevent data leakage.
7. **Model Training (`src/models.py` & `src/lstm_model.py`):** Trains Classical models (Ridge, Random Forest, XGBoost) and Deep Learning sequence models (Vanilla RNN, GRU, PyTorch LSTM) on the optimized features. Early Stopping prevents overfitting.
8. **Algorithmic Backtesting (`src/backtesting.py`):** Converts the AI's numerical predictions into actual trading signals (LONG/SHORT/FLAT) based on a confidence threshold, deducts realistic broker transaction fees, and outputs business metrics like the **Sharpe Ratio** and **Max Drawdown**.

---

# 1. Theoretical Foundations & Core Financial Principles

### 1.1 The Efficient Market Hypothesis (EMH) vs. Machine Learning
The **Efficient Market Hypothesis (EMH)**, formulated by Eugene Fama (1970), asserts that financial asset prices fully reflect all available information. Under EMH:
* **Weak-Form EMH:** Past prices and volume cannot predict future prices (technical analysis is useless).
* **Semi-Strong Form EMH:** All publicly available information (earnings, news) is instantly reflected in prices.
* **Strong Form EMH:** Even private/insider information is instantly reflected.

**Why does Machine Learning still have a role?**
Real-world markets exhibit **micro-inefficiencies**, **behavioral biases** (herding, panic selling), **liquidity constraints**, and **non-linear momentum/mean-reversion regimes**. While forecasting long-term stock prices is virtually impossible due to low signal-to-noise ratio, machine learning can capture subtle conditional distributions, nonlinear interactions between volume and momentum, and volatility clustering over short horizons ($t+1$).

---

### 1.2 Stationarity & The Random Walk Problem
A time-series $X_t$ is **strictly stationary** if its joint probability distribution does not change over time. It is **weakly (covariance) stationary** if:
1. Mean is constant over time: $\mathbb{E}[X_t] = \mu$ for all $t$.
2. Variance is finite and constant: $\text{Var}(X_t) = \sigma^2 < \infty$ for all $t$.
3. Autocovariance depends only on the lag $k$, not on time $t$: $\text{Cov}(X_t, X_{t+k}) = \gamma(k)$.

#### Why Raw Stock Prices are Non-Stationary
Raw stock prices $P_t$ follow an approximate **Geometric Random Walk with Drift**:
$$P_t = P_{t-1} e^{(\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} \epsilon_t}$$
Prices trend upwards over decades, meaning $\mathbb{E}[P_t]$ and $\text{Var}(P_t)$ grow over time ($I(1)$ integrated order 1). If an ML algorithm trains on prices from 2010 (\$1,200 for S&P 500) and attempts to predict 2024 (\$5,500), the model is operating outside its training domain and will fail completely.

---

### 1.3 Why Raw Prices Fail: Simple Returns vs. Log Returns
To make financial data stationary ($I(0)$ integrated order 0), we transform prices into **returns**.

#### 1. Simple (Linear) Returns
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$
* **Intuition:** The actual percentage gain or loss from day $t-1$ to day $t$.
* **Limitation:** Asymmetric. If an asset falls by $-50\%$ from \$100 to \$50, it requires a $+100\%$ gain to return to \$100. Furthermore, multi-period compounding is multiplicative: $1 + R_{0 \to T} = \prod_{t=1}^T (1 + R_t)$.

#### 2. Log (Continuously Compounded) Returns
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$

#### Mathematical Proof of Time-Additivity:
Consider the return over $T$ periods from $t=0$ to $t=T$:
$$r_{0 \to T} = \ln\left(\frac{P_T}{P_0}\right) = \ln(P_T) - \ln(P_0)$$
Expanding this telescoping sum:
$$r_{0 \to T} = \sum_{t=1}^T (\ln P_t - \ln P_{t-1}) = \sum_{t=1}^T r_t$$
* **Key Advantages of Log Returns:**
  1. **Time-Additive:** The $k$-day return is simply the arithmetic sum of the individual 1-day log returns.
  2. **Symmetry:** $+r$ and $-r$ represent equal and opposite proportional changes.
  3. **Normal Distribution Approximation:** In classical quantitative finance (Black-Scholes framework), log returns are assumed to be normally distributed, making $\ln P_t$ follow a normal distribution.

> [!NOTE]  
> **Interview Focus: The Problem Solved (Why Log Returns?)**  
> Interviewers often ask why Quants use Log Returns instead of standard percentage gains. If a stock drops 50% (\$100 to \$50), it takes a 100% gain to recover. Simple returns are completely asymmetrical. Log returns are perfectly symmetric ($+r$ and $-r$ cancel each other out) and time-additive, making them mathematically safe to feed into neural networks.

---

### 1.4 Realized Volatility, Clustering, & Annualization ($\sqrt{252}$)
**Volatility** ($\sigma$) measures the dispersion of returns—it is the financial proxy for uncertainty and risk.

#### 1. Realized Volatility (Standard Deviation of Log Returns)
For a rolling window of $W$ trading days (e.g., $W = 21$ business days $\approx 1$ calendar month):
$$\bar{r} = \frac{1}{W} \sum_{i=0}^{W-1} r_{t-i}$$
$$\sigma_{\text{daily}, t} = \sqrt{\frac{1}{W-1} \sum_{i=0}^{W-1} (r_{t-i} - \bar{r})^2}$$

#### 2. Why Multiply by $\sqrt{252}$ to Annualize?
Assuming daily returns are independently and identically distributed (i.i.d.) with daily variance $\sigma^2_{\text{daily}}$:
$$\text{Var}(r_{\text{annual}}) = \text{Var}\left(\sum_{t=1}^{252} r_t\right) = \sum_{t=1}^{252} \text{Var}(r_t) = 252 \times \sigma^2_{\text{daily}}$$
Taking the standard deviation (square root of variance):
$$\sigma_{\text{annual}} = \sqrt{252 \times \sigma^2_{\text{daily}}} = \sigma_{\text{daily}} \times \sqrt{252} \approx \sigma_{\text{daily}} \times 15.8745$$
*Standard market convention uses $252$ trading days per calendar year (excluding weekends and exchange holidays).*

#### 3. Volatility Clustering (Mandelbrot Phenomenon)
Benoit Mandelbrot (1963) observed: *"Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes."*
While return directions are nearly uncorrelated across days, **squared returns and volatility are highly auto-correlated**. This makes volatility forecasting significantly more predictable with ML models than pure return direction.

---

### 1.5 Geometric Brownian Motion (GBM) Data Generation
When offline or using synthetic testing mode (`--synthetic`), the system generates synthetic price paths via **Geometric Brownian Motion (GBM)** according to the stochastic differential equation:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$
Using Ito's Lemma, the exact discrete solution is:
$$S(t+\Delta t) = S(t) \cdot \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z_t \right)$$
* Where:
  * $S_0 = 100.0$ (initial price)
  * $\mu = 0.08$ (8% annualized drift/expected return)
  * $\sigma = 0.18$ (18% annualized volatility)
  * $\Delta t = \frac{1}{252}$ (daily time step)
  * $Z_t \sim \mathcal{N}(0, 1)$ (standard normal random variable)

---

# 2. Data Ingestion & Preprocessing Pipeline

```
Raw Yahoo Finance CSV ──▶ Chronological Sort ──▶ Deduplication ──▶ Forward-Fill NaNs
                                                                          │
Target Construction (t+1) ◀── Volatility Calculation ◀── Log Return Calc ◀─┘
```

### 2.1 Understanding OHLCV Data
Every trading session records 5 core variables:
* **Open ($O_t$):** Price of the first transaction when the exchange opens.
* **High ($H_t$):** Maximum price reached during continuous trading hours.
* **Low ($L_t$):** Minimum price reached during continuous trading hours.
* **Close ($C_t$):** Final settled price at exchange closing bell.
* **Volume ($V_t$):** Total count of shares/contracts transacted during that day.

### 2.2 Cleaning, Duplicate Handling, and Forward-Filling
Implemented in [`src/preprocessing.py`](file:///d:/ml_project/src/preprocessing.py):
1. **Datetime Indexing:** Convert date strings to UTC/timezone-naive `pd.Timestamp` and sort ascendingly.
2. **Deduplication:** Check for duplicate calendar dates and retain the latest verified observation.
3. **Missing Value Imputation:**
   * Holidays and shortened sessions can introduce `NaN` values.
   * We apply `df.ffill(limit=5)` (forward-fill up to 5 days). This propagates the last known valid price forward without leaking future information.
   * Any remaining leading `NaN` values are dropped.

### 2.3 Target Construction & Temporal Alignment ($t+1$)
In supervised machine learning, we need feature matrix $\mathbf{X}_t$ (known at time $t$) to predict label $y_t$ (which occurs at $t+1$).

```python
# Tomorrow's 1-day log return (Target for Return Forecasting)
df['target_return_1d'] = df['log_return'].shift(-1)

# Tomorrow's 21-day annualized volatility (Target for Volatility Forecasting)
df['target_vol_1d'] = df['volatility_21d'].shift(-1)
```

> [!IMPORTANT]  
> **Interview Focus: The Problem Solved (Predicting the Future vs The Present)**  
> A classic beginner mistake is predicting today's closing price using today's opening price. That is useless for algorithmic trading because by the time the market closes, you've already missed the opportunity to trade. In this pipeline, we explicitly shift the target variable by `-1` row. We feed the AI the features for Day $t$ so it can predict the return for Day $t+1$. 

---

* **Critical Rule:** Shifting by `-1` moves the future value backwards by 1 row so that the row indexed at time $t$ contains all features known at close of day $t$, alongside the outcome of day $t+1$. The very last row has no known target and is dropped during model training.

---

# 3. Exhaustive Feature Engineering Reference

Feature engineering in [`src/features.py`](file:///d:/ml_project/src/features.py) converts raw prices into 22 stationary, predictive signals across 5 categories.

### 3.1 Momentum & Trend Indicators
#### 1. Simple Moving Average (SMA)
Calculates the unweighted arithmetic mean of the closing price over $n$ periods:
$$\text{SMA}_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} C_{t-i}$$
* Windows used: $n \in \{5, 10, 20, 50\}$.
* **Price-to-SMA Ratio:** $\frac{C_t}{\text{SMA}_n(t)}$. Normalizes price relative to its moving average. Values $> 1.0$ indicate the price is above its trend (bullish momentum).

#### 2. Exponential Moving Average (EMA)
Applies exponentially decreasing weights to older prices. Gives higher weight to recent prices:
$$\alpha = \frac{2}{n + 1}$$
$$\text{EMA}_n(t) = \alpha \cdot C_t + (1 - \alpha) \cdot \text{EMA}_n(t-1)$$
* Windows used: $\text{EMA}_{12}$ (fast) and $\text{EMA}_{26}$ (slow).

#### 3. Rate of Change (ROC / Momentum)
Measures the percentage change in price over $n$ periods:
$$\text{ROC}_n(t) = \frac{C_t - C_{t-n}}{C_{t-n}}$$
* Windows used: $n \in \{5, 10, 20\}$ days.

---

### 3.2 Oscillators & Mean Reversion
#### 1. Relative Strength Index (RSI - 14 Days)
Quantifies the velocity and magnitude of price movements on an oscillator scale of $[0, 100]$.
1. Calculate daily changes: $\Delta C_t = C_t - C_{t-1}$.
2. Separate gains and losses:
   $$U_t = \max(\Delta C_t, 0), \quad D_t = \max(-\Delta C_t, 0)$$
3. Compute smoothed average gains and losses using Wilder's exponential smoothing ($\alpha = \frac{1}{14}$):
   $$\text{AvgGain}_t = \frac{13 \cdot \text{AvgGain}_{t-1} + U_t}{14}$$
   $$\text{AvgLoss}_t = \frac{13 \cdot \text{AvgLoss}_{t-1} + D_t}{14}$$
4. Calculate Relative Strength ($RS$) and $RSI$:
   $$RS_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}$$
   $$RSI_t = 100 - \frac{100}{1 + RS_t} = 100 \cdot \frac{\text{AvgGain}_t}{\text{AvgGain}_t + \text{AvgLoss}_t}$$
* **Interpretation:** $RSI > 70 \implies$ Overbought (potential reversal downward); $RSI < 30 \implies$ Oversold (potential bounce upward).

#### 2. Moving Average Convergence Divergence (MACD)
Captures changes in trend strength, direction, and duration.
$$\text{MACD Line}_t = \text{EMA}_{12}(C)_t - \text{EMA}_{26}(C)_t$$
$$\text{Signal Line}_t = \text{EMA}_9(\text{MACD Line})_t$$
$$\text{MACD Histogram}_t = \text{MACD Line}_t - \text{Signal Line}_t$$
* **Interpretation:** When the MACD line crosses above the signal line (Histogram turns positive), it signals upward momentum.

---

### 3.3 Volatility & Range Indicators
#### 1. Bollinger Bands & %B
Measures price dispersion relative to a 20-day moving average:
$$\text{Middle Band}_t = \text{SMA}_{20}(C)_t$$
$$\text{Upper Band}_t = \text{Middle Band}_t + 2 \cdot \sigma_{20}(C)_t$$
$$\text{Lower Band}_t = \text{Middle Band}_t - 2 \cdot \sigma_{20}(C)_t$$
$$\text{BB \%B}_t = \frac{C_t - \text{Lower Band}_t}{\text{Upper Band}_t - \text{Lower Band}_t}$$
$$\text{BB Bandwidth}_t = \frac{\text{Upper Band}_t - \text{Lower Band}_t}{\text{Middle Band}_t}$$
* **Interpretation:** $\%B > 1.0$ means price has pierced the upper band (extended); $\%B < 0.0$ means price has pierced the lower band.

#### 2. High-Low Normalized Range
$$\text{HL\_Range}_t = \frac{H_t - L_t}{C_t}$$
Measures intraday price dispersion normalized by the closing price.

---

### 3.4 Volume Dynamics
#### 1. Volume Change & Volume SMA Ratio
$$\text{Volume\_Change}_t = \frac{V_t - V_{t-1}}{V_{t-1}}$$
$$\text{Volume\_SMA\_Ratio}_t = \frac{V_t}{\text{SMA}_{20}(V)_t}$$
* High volume accompanying a price breakout confirms the strength of the trend.

#### 2. On-Balance Volume (OBV)
Accumulates trading volume based on price direction:
$$OBV_t = OBV_{t-1} + \begin{cases} +V_t & \text{if } C_t > C_{t-1} \\ 0 & \text{if } C_t = C_{t-1} \\ -V_t & \text{if } C_t < C_{t-1} \end{cases}$$

---

### 3.5 Statistical Properties
* **Rolling Volatilities:** Computed over $5, 10, 21, 63$ days to capture short-term, medium-term, and quarterly volatility dynamics.
* **Rolling Skewness & Kurtosis (63-day window):** Measures the asymmetry and tail heaviness of the return distribution.
* **Autocorrelations (Lag 1, 2, 5):** Rolling correlation between $r_t$ and $r_{t-k}$ to detect mean-reversion or persistence regimes.

---

### 3.6 Summary Table of All 22 Engineered Features

| Feature Name | Category | Mathematical Description |
|---|---|---|
| `Return_Lag_1`, `2`, `3`, `5` | Lagged Returns | Historical log returns $r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}$ |
| `Vol_Lag_1`, `2`, `3`, `5` | Lagged Volatility | Historical 21-day realized volatility $\sigma_{t-1}, \dots, \sigma_{t-5}$ |
| `Close_to_SMA_5`, `10`, `20`, `50` | Trend Ratios | $\frac{C_t}{\text{SMA}_k(C)_t}$ |
| `HL_Range` | Intraday Range | $\frac{H_t - L_t}{C_t}$ |
| `Volume_Change` | Volume Flow | $\frac{V_t - V_{t-1}}{V_{t-1}}$ |
| `Volume_SMA_Ratio` | Volume Relative | $\frac{V_t}{\text{SMA}_{20}(V)_t}$ |
| `ROC_5`, `10`, `20` | Momentum | $\frac{C_t - C_{t-k}}{C_{t-k}}$ |
| `RSI_14` | Oscillator | 14-day Wilder Relative Strength Index |
| `MACD`, `MACD_Signal`, `MACD_Hist` | Trend Oscillator | Difference of 12-day & 26-day EMA vs 9-day signal |
| `BB_PctB`, `BB_Bandwidth` | Volatility Envelope | Bollinger $\%B$ position and normalized band width |

---

# 4. Time-Series Validation & Avoiding Data Leakage

```
                                TIME FLOW ──────────────────────────────▶
┌──────────────────────────────────────┬──────────────────┬──────────────────┐
│          TRAIN SET (70%)             │ VALIDATION (15%) │  TEST SET (15%)  │
│         2010-01 to 2020-06           │ 2020-06 to 2022-09 │ 2022-09 to 2024-12 │
└──────────────────────────────────────┴──────────────────┴──────────────────┘
 [Fit Scaler] ──────────────────────────▶ [Transform Only] ──▶ [Transform Only]
 [Train Model Weights] ─────────────────▶ [Early Stopping] ───▶ [Final Evaluation]
```

### 4.1 Why Standard K-Fold Cross Validation Destroys Financial ML
In standard machine learning (e.g., image classification), samples are assumed to be independent and identically distributed ($i.i.d.$). Random shuffling allows sample #500 from 2023 to be placed in the training set while sample #200 from 2018 is in the test set.

In financial time-series, this creates **Lookahead Bias (Data Leakage)**:
1. Future market regimes, crashes, and trend changes are implicitly exposed to the training algorithm.
2. Rolling technical indicators (e.g., 200-day SMA) computed across shuffled data would embed future price information directly into training features.
3. The resulting model exhibits near-perfect training and test accuracy, but loses 100% of its capital when deployed on live forward data.

> [!IMPORTANT]  
> **Interview Focus: The Problem Solved (Lookahead Bias)**  
> Standard machine learning relies on random shuffling. In finance, if you shuffle time-series data, you accidentally train the model on 2023 data to predict 2021 prices. The model achieves 99% accuracy in testing but blows up your portfolio in live trading. We solve this by strictly enforcing chronological splitting (`TimeSeriesSplit`).

### 4.2 Chronological Splitting Architecture
We strictly enforce forward-in-time splitting:
* **Train Set (70%):** Model parameters (weights, decision boundaries) are learned solely from past data.
* **Validation Set (15%):** Used exclusively for hyperparameter tuning and PyTorch early stopping triggers.
* **Test Set (15%):** Pristine, unseen future period simulating a live trading environment. Evaluated exactly once.

### 4.3 Feature Normalization without Lookahead Bias
Feature scaling (z-score standardization) must estimate mean $\mu$ and standard deviation $\sigma$ **strictly from the training partition**:
$$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} X_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (X_i - \mu_{\text{train}})^2}$$
$$X_{\text{train, scaled}} = \frac{X_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
$$X_{\text{val, scaled}} = \frac{X_{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad X_{\text{test, scaled}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
*Never call `scaler.fit_transform(X_test)` or `scaler.fit(df)`. Only call `scaler.fit(X_train)`!*

---

### 4.4 Automated Feature Selection (Correlation Pruning & Importance Ranking)
Feeding deep learning models too many noisy or heavily correlated features causes overfitting and degrades out-of-sample performance. After generating 30+ features, we mathematically filter them using only the training data (to prevent leakage):

1. **Correlation Pruning:** We compute the Pearson correlation matrix of all features. If two features (e.g., `SMA_10` and `SMA_20`) have a correlation $> 0.90$, we automatically drop the one that is more globally redundant. This drastically reduces multicollinearity.
2. **Random Forest Importance Ranking:** We train a lightweight Random Forest on the remaining pruned features, extract the Gini `feature_importances_`, and keep only the **Top 10** most predictive signals. 

This compression reduces the input tensor size for the PyTorch models, speeding up training and forcing the models to learn robust financial patterns rather than memorizing noise.

> [!IMPORTANT]  
> **Interview Focus: The Problem Solved (The Curse of Dimensionality)**  
> Feeding Neural Networks 30+ highly correlated technical indicators causes severe overfitting. The model memorizes noise instead of finding true market patterns. By mathematically pruning redundant features and keeping only the Top 10 using Tree-Based Importance, we ensure the deep learning models receive only the highest-quality signal, improving out-of-sample trading performance.

---

# 5. Classical Machine Learning Models

Implemented in [`src/models.py`](file:///d:/ml_project/src/models.py).

### 5.1 Ridge Regression ($L_2$ Regularization)
Standard Ordinary Least Squares (OLS) minimizes $\sum (y_i - \mathbf{w}^T \mathbf{x}_i)^2$. When features are highly collinear (like multiple SMAs), OLS produces wildly unstable, high-variance weights.

**Ridge Regression** adds an $L_2$ penalty to shrink coefficients:
$$\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2 + \alpha \sum_{j=1}^P w_j^2$$
* **Closed-Form Solution:**
  $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
* $\alpha \ge 0$ controls the regularizer. When $\alpha=0$, it reduces to OLS. As $\alpha \to \infty$, weights shrink towards zero.

> [!NOTE]  
> **Interview Focus: The Problem Solved (Multicollinearity)**  
> Standard Linear Regression fails when you have highly correlated features (like SMA_5 and SMA_10) because the coefficients explode to infinity. We use Ridge Regression because the $L_2$ penalty acts as a shock absorber, shrinking the weights evenly and making the model robust to financial noise.

---

### 5.2 Random Forest Regressor
An ensemble learning technique utilizing **Bagging (Bootstrap Aggregation)** and **Random Subspace Method**:
1. **Bootstrap Sampling:** Draw $B$ random subsets of the training data with replacement.
2. **Tree Building:** Grow $B$ deep decision trees (e.g., $B = 200$, `max_depth=10`). At each candidate split, select the best feature from a random subset of $\sqrt{P}$ features.
3. **Aggregation:** Final prediction is the arithmetic average across all $B$ trees:
   $$\hat{y}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
* **Variance Reduction:** Averaging decorrelated trees dramatically reduces prediction variance without increasing bias.

> [!NOTE]  
> **Interview Focus: Bagging vs. Boosting (The Classic ML Question)**  
> **Random Forest (Bagging):** Builds 100 independent deep trees and averages them. It solves the problem of high *Variance* (overfitting).  
> **XGBoost (Boosting):** Builds 1 shallow tree, analyzes its mistakes, and builds the next tree specifically to fix those mistakes. It solves the problem of high *Bias* (underfitting). We benchmarked both to find the perfect middle ground for financial time-series.

---

### 5.3 XGBoost Regressor (Extreme Gradient Boosting)
Unlike Random Forest (which builds independent trees in parallel), **Gradient Boosting** builds trees **sequentially**, with each new tree learning to predict the **pseudo-residuals (errors)** of the preceding ensemble.

At step $m$, the objective is:
$$\mathcal{L}^{(m)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(m-1)} + f_m(\mathbf{x}_i)\right) + \Omega(f_m)$$
Where $\Omega(f_m) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$ penalizes tree complexity ($T$ leaves and leaf weights $w_j$).

Using a second-order Taylor expansion:
$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ l(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \Omega(f_m)$$
Where $g_i = \frac{\partial l(y_i, \hat{y})}{\partial \hat{y}}$ (1st derivative/gradient) and $h_i = \frac{\partial^2 l(y_i, \hat{y})}{\partial \hat{y}^2}$ (2nd derivative/Hessian).

---

# 6. Deep Learning with PyTorch: Sequence Modeling (RNN, LSTM, GRU)

Implemented in [`src/lstm_model.py`](file:///d:/ml_project/src/lstm_model.py) and [`src/training.py`](file:///d:/ml_project/src/training.py).

### 6.1 Vanilla RNNs, LSTMs, and GRUs (Architectural Differences)
A standard **Vanilla Recurrent Neural Network (RNN)** updates its hidden state according to:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
During **Backpropagation Through Time (BPTT)**, computing the gradient of the loss $\mathcal{L}$ with respect to the initial hidden state $h_0$ requires computing the matrix product chain:
$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=1}^T \text{diag}(1 - \tanh^2(\cdot)) W_{hh}^T$$
* **Vanishing Gradient:** If the largest eigenvalue of $W_{hh} < 1$, the gradients vanish exponentially as $T \to \infty$. The network cannot remember patterns from 30+ days ago.
* **Exploding Gradient:** If the largest eigenvalue of $W_{hh} > 1$, gradients explode to infinity (`NaN`).

To solve this, advanced architectures use **gating mechanisms**:
1. **LSTMs (Long Short-Term Memory)** use 3 gates (Forget, Input, Output) and maintain a separate internal cell memory $C_t$ to act as a gradient highway. 
2. **GRUs (Gated Recurrent Units)** are a streamlined variation using only 2 gates (Update, Reset) and no separate cell memory. They typically train faster than LSTMs while achieving similar performance on financial time-series.

> [!IMPORTANT]  
> **Interview Focus: The Problem Solved (The Vanishing Gradient)**  
> Standard RNNs try to remember the past, but during backpropagation, gradients are multiplied repeatedly across time steps. Over a 30-to-60 day sequence, these gradients exponentially shrink to zero, and the model "forgets" earlier days. We solve this using LSTMs, which introduce an additive "Cell State" (a gradient superhighway) to preserve long-term financial dependencies.

---

### 6.2 Detailed Anatomy of an LSTM Cell
The **Long Short-Term Memory (LSTM)** network (Hochreiter & Schmidhuber, 1997) solves this by introducing a linear memory highway called the **Cell State ($C_t$)**, regulated by three non-linear gates.

```
                  ┌──────────────────────────────────────────────┐
  C_{t-1} ───────▶│ (X) ───────────────────────────── (+) ──────▶│───────▶ C_t
                  │  ▲                                 ▲         │
                  │  │ f_t                             │ i_t * C~│
                  │  │                                 │         │
                  │ ┌┴───────┐   ┌─────────┐      ┌────┴────┐    │
                  │ │ σ Gate │   │ σ Gate  │      │tanh Gate│    │
                  │ │ Forget │   │  Input  │      │Candidate│    │
                  │ └───▲────┘   └───▲─────┘      └────▲────┘    │
                  │     │            │                 │         │
                  │  ───┴────────────┴─────────────────┴───────  │
                  │              [h_{t-1}, x_t]                  │
                  │  ──────────────────────────────────┬───────  │
                  │                                    │         │
                  │                              ┌─────▼─────┐   │
                  │                              │  σ Gate   │   │
                  │                              │  Output   │   │
                  │                              └─────┬─────┘   │
                  │                                    ▼ o_t     │
                  │                                   (X) ◀──────┼── tanh(C_t)
                  │                                    │         │
  h_{t-1} ───────▶│                                    ▼         │───────▶ h_t
  x_t     ───────▶│                                   h_t        │
                  └──────────────────────────────────────────────┘
```

#### The 6 Exact Mathematical Equations:
1. **Forget Gate ($f_t$):** Controls what percentage of past memory $C_{t-1}$ to discard ($0 = \text{erase}, 1 = \text{keep}$):
   $$f_t = \sigma\left( \mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f \right)$$
2. **Input Gate ($i_t$):** Decides which dimensions of the new input to store:
   $$i_t = \sigma\left( \mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i \right)$$
3. **Candidate Cell State ($\tilde{C}_t$):** Generates new candidate values:
   $$\tilde{C}_t = \tanh\left( \mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C \right)$$
4. **Cell State Update ($C_t$):** Linear combination creating an uninterrupted gradient highway:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5. **Output Gate ($o_t$):** Determines which parts of the cell state to output to the hidden state:
   $$o_t = \sigma\left( \mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o \right)$$
6. **Hidden State Output ($h_t$):**
   $$h_t = o_t \odot \tanh(C_t)$$

*Here $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function, $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, and $\odot$ is the Hadamard (element-wise) product.*

---

### 6.3 Sliding Window Tensor Construction & Sequence Length Experiment
Unlike classical ML (which takes a 2D matrix $\mathbf{X} \in \mathbb{R}^{N \times P}$), PyTorch recurrent models require a **3D Tensor**:
$$\mathbf{X}_{\text{tensor}} \in \mathbb{R}^{\text{Batch Size} \times \text{Sequence Length} \times \text{Number of Features}}$$

```
Sample i:
[ Day t-29: [feat_1, feat_2, ..., feat_22] ]
[ Day t-28: [feat_1, feat_2, ..., feat_22] ]
[ ...                                      ]  ──▶  Target: Return at Day t+1
[ Day t-1 : [feat_1, feat_2, ..., feat_22] ]
[ Day t   : [feat_1, feat_2, ..., feat_22] ]
```
* Default config: `seq_len = 30` days, `num_features = 22`, `batch_size = 32`.

**Sequence Length Memory Experiment:** The project includes a dedicated research script (`run_sequence_experiment.py`) to empirically test "How much historical context do NNs need?". It trains Vanilla RNN, LSTM, and GRU models across varying sequence lengths (5, 10, 20, 30, and 60 days) and plots the test MAE. This isolates the sequence length variable to observe the Vanishing Gradient problem in real-time on financial data.

---

### 6.4 Full PyTorch Network Architectures
Defined in `RNNModel`, `LSTMModel`, and `GRUModel` in [`src/lstm_model.py`](file:///d:/ml_project/src/lstm_model.py). All three models share the same structure except for the recurrent layer type:

```
Input Tensor: (32, 30, 22)
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Recurrent Layer 1: hidden_size=64, batch_first=True    │
│ (nn.RNN, nn.LSTM, or nn.GRU)                           │
└────────────────────────────────────────────────────────┘
    │  Output: (32, 30, 64)
    ▼
┌────────────────────────────────────────────────────────┐
│ Spatial Dropout: p = 0.20                              │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ LSTM Layer 2: hidden_size=64, batch_first=True         │
└────────────────────────────────────────────────────────┘
    │  Output: (32, 30, 64)
    ▼
┌────────────────────────────────────────────────────────┐
│ Extract Last Time Step h_T: Shape (32, 64)             │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Fully Connected Head: Linear(64 -> 32) + ReLU          │
│ Dropout(0.20) + Linear(32 -> 1)                        │
└────────────────────────────────────────────────────────┘
    │
    ▼
Output Scalar: Predicted Return y_hat (32, 1)
```

---

### 6.5 Training Loops, Gradient Clipping, Schedulers, & Early Stopping
1. **AdamW Optimizer:** Uses Adaptive Moment Estimation with decoupled weight decay ($\lambda = 10^{-4}$):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   $$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
2. **Gradient Clipping (`torch.nn.utils.clip_grad_norm_`):**
   $$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{\text{max\_norm}}{\|\mathbf{g}\|_2}\right) \quad (\text{max\_norm} = 1.0)$$
   Guarantees gradients cannot explode during BPTT.
3. **Learning Rate Scheduler (`ReduceLROnPlateau`):**
   * Monitors Validation Loss.
   * If validation loss fails to decrease for 5 consecutive epochs, learning rate is multiplied by factor $\gamma = 0.5$.
4. **Early Stopping with Best Checkpoint Preservation:**
   * Runs for up to 100 epochs with `patience = 15`.
   * When validation loss achieves a new all-time low, `torch.save(model.state_dict(), 'results/lstm_best.pt')`.
   * If 15 epochs pass without improvement, training terminates and the best checkpoint weights are restored.

> [!IMPORTANT]  
> **Interview Focus: The Problem Solved (Neural Network Overfitting)**  
> If you train an LSTM for 1,000 epochs, it will eventually memorize the training data and fail in real life. We implemented **Early Stopping**: the script monitors the Validation Loss, and if the AI fails to improve for 15 straight epochs, it instantly aborts training and restores the best weights. 

---

# 7. Algorithmic Trading & Quantitative Backtesting

Implemented in [`src/backtesting.py`](file:///d:/ml_project/src/backtesting.py).

### 7.1 Trading Signal Generation Logic
The backtesting engine transforms model predictions $\hat{y}_{t+1}$ into discrete market positions:

$$S_t = \begin{cases} +1 \text{ (LONG)} & \text{if } \hat{y}_{t+1} > +\tau_{\text{buy}} \\ 0 \text{ (FLAT)} & \text{if } \hat{y}_{t+1} < -\tau_{\text{sell}} \\ S_{t-1} \text{ (HOLD)} & \text{otherwise} \end{cases}$$

* **Threshold Selection ($\tau = 0.002 = 0.20\%$):** 
  * If $\tau$ is too small (e.g., $0.0001$), the model trades constantly on noise, losing all capital to transaction fees.
  * If $\tau$ is too high (e.g., $0.02 = 2.0\%$), the model almost never triggers a trade.
  * $\tau = 0.20\%$ represents a realistic hurdle matching exchange fees and bid-ask slippage.

---

### 7.2 Realistic Market Frictions: Transaction Costs & Slippage
Every time the position changes from day $t-1$ to day $t$ ($\Delta S_t = |S_t - S_{t-1}| > 0$), a transaction cost is deducted:
$$\text{Cost}_t = c \cdot |\Delta S_t| \cdot V_t$$
* Where $c = 0.0010$ ($10$ basis points / $0.10\%$ per trade).
* This penalty prevents high-turnover false strategies from appearing artificially profitable.

> [!IMPORTANT]  
> **Interview Focus: The Problem Solved (The Reality of Broker Fees)**  
> An AI that predicts a 0.05% gain every day looks like a billionaire-maker on paper. But if broker fees are 0.1%, the AI will bankrupt you. We solved this by implementing a strict confidence threshold ($\tau$) and deducting transaction costs for every signal flip. The model only buys when its predicted return is greater than the cost of the trade.

---

### 7.3 Portfolio Accounting & Equity Curve Evolution
Starting with initial capital $V_0 = \$100,000$:
$$\text{Daily Return}_t = S_{t-1} \cdot R_t - c \cdot |S_t - S_{t-1}|$$
$$V_t = V_{t-1} \cdot (1 + \text{Daily Return}_t)$$
$$\text{Cumulative Return}_t = \frac{V_t - V_0}{V_0}$$

---

### 7.4 The Baseline: Buy-and-Hold Benchmark
* $S_t = 1$ for all $t$.
* Buys on day 1 of the test set and holds until the final day.
* Any active algorithmic trading strategy must be judged against this baseline on both **absolute return** and **risk-adjusted return (Sharpe/Drawdown)**.

---

# 8. Comprehensive Evaluation Metrics

Implemented in [`src/evaluation.py`](file:///d:/ml_project/src/evaluation.py).

### 8.1 Machine Learning Regression Metrics
1. **Mean Squared Error (MSE):**
   $$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
3. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
4. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

> [!NOTE]  
> **Interview Focus: Why $R^2$ is irrelevant in Finance**  
> In standard ML (e.g., predicting house prices), a good $R^2$ is $> 0.80$. In finance, predicting exact numerical returns is nearly impossible because markets are highly noisy. An $R^2$ near zero is completely standard. We care much more about *Directional Accuracy* (guessing Up vs. Down) and actual backtested *Sharpe Ratio*.

---

### 8.2 Directional & Classification Metrics
1. **Directional Accuracy (Hit Rate):**
   $$\text{DA} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{sign}(y_i) == \text{sign}(\hat{y}_i) \right)$$
   * Even an accuracy of $53\%$ to $56\%$ with strict risk management can generate strong positive alpha in financial markets.
2. **Information Coefficient (IC):**
   $$\text{IC} = \text{Spearman Rank Correlation}(\hat{\mathbf{y}}, \mathbf{y})$$
   Measures whether higher model predictions reliably correlate with higher actual returns.

---

### 8.3 Financial Performance & Risk-Adjusted Metrics

| Metric | Formula | Financial Meaning |
|---|---|---|
| **Cumulative Return** | $\frac{V_T - V_0}{V_0}$ | Total net percentage gain over the test period. |
| **CAGR** | $\left(\frac{V_T}{V_0}\right)^{252 / T} - 1$ | **Compound Annual Growth Rate:** Annualized geometric growth. |
| **Annualized Volatility** | $\sigma_{\text{daily}} \times \sqrt{252}$ | Annualized standard deviation of strategy returns. |
| **Sharpe Ratio** | $\frac{\text{CAGR} - R_f}{\sigma_{\text{annual}}}$ | Excess return earned per unit of total volatility ($R_f = 4.0\%$). |
| **Sortino Ratio** | $\frac{\text{CAGR} - R_f}{\sigma_{\text{downside}} \times \sqrt{252}}$ | Excess return per unit of **harmful downside volatility** ($\sigma_{\text{downside}} = \text{std}(\min(R_t - 0, 0))$). |
| **Maximum Drawdown (MDD)** | $\max_{t} \left( \frac{\text{Peak}_t - V_t}{\text{Peak}_t} \right)$ | Largest peak-to-trough percentage loss experienced by the portfolio. |
| **Calmar Ratio** | $\frac{\text{CAGR}}{\|\text{Max Drawdown}\|}$ | Ratio of annual return to worst-case historical drawdown. |
| **Win Rate** | $\frac{\sum \mathbb{I}(R_{\text{trade}} > 0)}{\text{Total Trades}}$ | Percentage of profitable closed trading days. |
| **Profit Factor** | $\frac{\sum \text{Gains}}{\sum |\text{Losses}|}$ | Ratio of gross profits to gross losses. $> 1.0 \implies$ Profitable. |

---

# 9. Hyperparameter Tuning & Model Selection Framework

The project includes a robust hyperparameter optimization pipeline in [`src/tuning.py`](file:///d:/ml_project/src/tuning.py), built specifically for financial time-series to rigorously prevent data leakage.

### 9.1 Hyperparameter Search Strategy
Rather than relying on default parameters or randomized searches, we use **Deterministic Grid Search** to evaluate every predefined hyperparameter combination. This ensures absolute reproducibility of our optimization results.

### 9.2 Time-Series Cross-Validation
Standard K-Fold Cross-Validation randomly shuffles data, which implies training on the future to predict the past. We use `TimeSeriesSplit` for classical ML models, expanding the training window chronologically:
*   Fold 1: Train on Q1 $\to$ Validate on Q2
*   Fold 2: Train on Q1, Q2 $\to$ Validate on Q3
*   Fold 3: Train on Q1, Q2, Q3 $\to$ Validate on Q4

### 9.3 Classical ML Hyperparameter Tuning
For Scikit-learn and XGBoost architectures, we construct a parameter grid and iterate through the `TimeSeriesSplit`:
*   **Ridge:** Tunes the `alpha` (L2 regularization penalty).
*   **Random Forest:** Tunes `n_estimators`, `max_depth`, and `min_samples_leaf`.
*   **XGBoost:** Tunes tree complexity (`max_depth`), learning rate (`learning_rate`), and row/column subsampling to prevent overfitting.

### 9.4 Neural Network Hyperparameter Tuning
Tuning recurrent networks requires rebuilding the computational graph. Our PyTorch tuning loop systematically varies:
*   **Architecture:** Hidden units, number of recurrent layers, and dropout probabilities.
*   **Optimization:** Learning rate and batch sizes.
*   *Crucially:* We re-instantiate the model weights, optimizer, and the PyTorch DataLoaders (to accommodate potential `sequence_length` changes) from scratch on every iteration.

### 9.5 Early Stopping
To prevent the neural networks from memorizing the training noise, we employ an Early Stopping monitor. If the Validation Loss fails to improve for a set number of epochs (`patience=5`), training is halted prematurely, and the model weights from the best epoch are restored.

### 9.6 Model Selection Using Validation Data
The "Best" hyperparameter combination for each model is selected based **strictly on the Validation Set RMSE**. The model is then retrained on the full training set using these optimal parameters. 

### 9.7 Final Evaluation on Held-Out Test Set
The golden rule of machine learning is to never touch the Test Set during training or tuning. Our final model comparison and financial backtesting are executed on the pristine Test Set, providing an unbiased simulation of real-world, out-of-sample trading performance.

---

# 10. Project Codebase Architecture & File-by-File Breakdown

```
ml_project/
│
├── data/
│   └── raw_data.csv               # Cached OHLCV historical data
│
├── plots/                         # High-resolution generated analytical charts
│   ├── ...                        # (Various generated PNG plots)
│   ├── 12_tuning_comparison.png   # Before vs after hyperparameter tuning
│   └── 13_sequence_experiment.png # Sequence Length vs MAE for RNN/LSTM/GRU
│
├── results/                       # Serialized models and experimental metrics
│   ├── lstm_best.pt               # PyTorch saved weights checkpoint
│   ├── gru_best.pt                # GRU checkpoint
│   ├── vanilla_rnn_best.pt        # RNN checkpoint
│   ├── model_comparison.csv       # Regression & directional metrics (all models)
│   ├── financial_metrics.csv      # Financial metrics (Sharpe, MDD, CAGR)
│   ├── backtest_summary.csv       # Trade counts & position statistics
│   ├── tuning_summary.csv         # Best hyperparameter combinations
│   └── sequence_experiment.csv    # Results from sequence length tests
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Yahoo Finance API downloader + GBM fallback
│   ├── preprocessing.py           # Returns, rolling volatility, time-series splitting
│   ├── features.py                # 22 technical indicator calculation engine
│   ├── feature_selection.py       # Automated correlation pruning & RF importance ranking
│   ├── models.py                  # Scikit-learn & XGBoost classical baselines
│   ├── lstm_model.py              # PyTorch Dataset, DataLoader, LSTM Architecture
│   ├── training.py                # PyTorch training loop, early stopping, sweeps
│   ├── backtesting.py             # Vectorized backtesting & transaction cost engine
│   ├── tuning.py                  # Hyperparameter optimization module
│   └── evaluation.py              # ML and financial metric calculation functions
│
├── main.py                        # Master pipeline orchestrator & CLI entrypoint
├── run_sequence_experiment.py     # Standalone script for sequence length research
├── requirements.txt               # Locked project dependencies
└── README.md                      # This master study guide
```

---

# 11. Quick Start & CLI Execution Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Execution Modes

```bash
# 1. Run Complete End-to-End Pipeline on S&P 500 (Default)
python main.py

# 2. Run with Synthetic Data (100% Offline Mode - No Internet Needed)
python main.py --synthetic

# 3. Run with Hyperparameter Tuning (Extensive grid search)
python main.py --tune

# 4. Run with Fast Tuning (Reduced grids for quick testing)
python main.py --fast-tune

# 5. Run the Sequence Length Memory Experiment (RNN/LSTM/GRU comparison)
python run_sequence_experiment.py --synthetic

# 6. Run Fast Execution (Skips Hyperparameter Grid Search)
python main.py --skip-experiments

# 7. Run on NIFTY 50 (India) or Apple Inc.
python main.py --ticker ^NSEI
python main.py --ticker AAPL

# 8. Specify Custom Date Range
python main.py --ticker ^GSPC --start 2012-01-01 --end 2024-06-30
```

---

# 12. Limitations, Pitfalls, & Future Improvements

### Practical Limitations & Caveats
1. **Low Signal-to-Noise Ratio:** Stock market daily returns are dominated by noise. ML models will rarely achieve $R^2 > 0.05$. Focus must remain on **directional hit rate** and **risk-adjusted returns (Sharpe ratio)**.
2. **Execution Timing & Latency:** In this backtest, positions are assumed to execute at day's close. In live trading, orders are filled at next morning's open or with market-on-close (MOC) slippage.
3. **Regime Shifts:** A model trained during a 10-year quantitative easing bull market (2010–2020) can struggle during high-inflation interest rate hiking cycles (2022).

### Future Roadmap Extensions
* **Transformer Architectures:** Implement a Time-Series Patch Transformer (PatchTST) with Multi-Head Self-Attention.
* **Alternative Data Integration:** Incorporate news sentiment embeddings (via FinBERT) and macroeconomic indicators (VIX, 10-year Treasury yields).
* **Regime-Switching Models:** Combine Hidden Markov Models (HMM) to dynamically adjust position sizes across high-volatility vs low-volatility regimes.
* **Kelly Criterion Sizing:** Dynamically scale capital allocation based on predicted edge: $f^* = \frac{p(b+1) - 1}{b}$.

---

*Authored as an Educational ML Coursework Project demonstrating end-to-end quantitative financial engineering, deep learning sequence modeling, and algorithmic risk management.*
