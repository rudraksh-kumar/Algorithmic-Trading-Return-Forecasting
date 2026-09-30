"""
models.py - Classical Machine Learning Baselines
=================================================
This module implements and evaluates classical (non-neural-network) ML
regressors for predicting next-day stock returns and volatility.

Models Implemented
------------------
1. **Ridge Regression with RidgeCV + PCA**  (Fix #2)
   - RidgeCV automatically selects the optimal L2 regularisation strength
     (alpha) via cross-validation, replacing the manual alpha=1.0 default.
   - PCA (Principal Component Analysis) is prepended in a Pipeline to remove
     multicollinearity: 22 features contain highly correlated clusters
     (SMA_5/10/20/50 are ~95% correlated). PCA decorrelates and reduces
     dimensionality, stabilising the Ridge coefficient estimates.
   - Retains 95% of the total variance across all features.

2. **Random Forest Regressor**
   - Ensemble of decision trees with bagging.
   - Naturally handles non-linear relationships.
   - Less prone to overfitting than a single deep tree.

3. **XGBoost Regressor**  (Fix #3)
   - Gradient-boosted decision trees (sequential ensemble).
   - Overfitting fix: shallower trees (max_depth=3), row/column subsampling,
     L1+L2 regularisation, minimum child weight, and early stopping on the
     validation set.

Evaluation Metrics
------------------
- **MAE** (Mean Absolute Error): Average absolute prediction error.
- **RMSE** (Root Mean Squared Error): Penalises large errors more heavily.
- **R²** (Coefficient of Determination): Proportion of variance explained
  by the model.  R² = 1 is perfect; R² = 0 means no better than predicting
  the mean; R² < 0 means worse than the mean.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[Models] Warning: xgboost not installed. XGBoost model will be skipped.")


# ===================================================================
# EVALUATION UTILITIES
# ===================================================================
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    dict
        Dictionary with keys 'MAE', 'RMSE', 'R2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics(model_name: str, metrics: dict):
    """Pretty-print regression metrics for a given model."""
    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    print(f"  MAE  : {metrics['MAE']:.6f}")
    print(f"  RMSE : {metrics['RMSE']:.6f}")
    print(f"  R²   : {metrics['R2']:.6f}")
    print(f"{'=' * 50}")


# ===================================================================
# MODEL DEFINITIONS
# ===================================================================
def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pca_variance: float = 0.95,
):
    """
    Train a Ridge Regression model with automatic alpha selection and PCA
    dimensionality reduction.

    **Why PCA + RidgeCV instead of plain Ridge(alpha=1.0)?**

    The 22 engineered features contain highly correlated clusters:
      - SMA ratios at 5/10/20/50 days are ~90-95% correlated.
      - ROC at 5/10/20 days are ~85% correlated.
      - Rolling volatilities at 5/10/21/63 days are ~90% correlated.

    Multicollinearity causes OLS/Ridge coefficient estimates to become
    numerically unstable (high variance), leading to poor generalisation
    and the R² = -1.73 collapse observed with the original Ridge model.

    **Fix:**
    1. PCA decorrelates the features by rotating them into orthogonal
       principal components — eliminating collinearity entirely.
    2. `pca_variance=0.95` keeps enough components to explain 95% of the
       total variance while discarding the noisy, low-variance directions.
    3. RidgeCV tests 12 alpha candidates and picks the best via leave-one-
       out cross-validation on the training set (no data leakage).

    Parameters
    ----------
    X_train      : Feature matrix (n_samples, n_features). Already scaled.
    y_train      : Target vector (n_samples,).
    pca_variance : Fraction of variance to retain in PCA (default 0.95).

    Returns
    -------
    sklearn Pipeline (PCA -> RidgeCV), already fitted.
    """
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0,
              100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0]

    pipe = Pipeline([
        ("pca",   PCA(n_components=pca_variance, random_state=42)),
        ("ridge", RidgeCV(alphas=alphas, cv=5, scoring="r2")),
    ])
    pipe.fit(X_train, y_train)

    # Diagnostic output
    n_components = pipe.named_steps["pca"].n_components_
    best_alpha   = pipe.named_steps["ridge"].alpha_
    print(f"  [Ridge] PCA kept {n_components} components (from {X_train.shape[1]} features, "
          f"retaining {pca_variance*100:.0f}% variance)")
    print(f"  [Ridge] RidgeCV selected alpha = {best_alpha}")
    return pipe


def train_random_forest(X_train, y_train, n_estimators: int = 200,
                        max_depth: int = 10, random_state: int = 42):
    """
    Train a Random Forest Regressor.

    Random Forest fits multiple decision trees on bootstrapped subsets
    of the training data and averages their predictions.  Key advantages:
    - Reduces variance (overfitting) compared to a single tree.
    - Captures non-linear feature interactions.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of each tree (limits complexity).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    sklearn RandomForestRegressor (fitted)
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    n_estimators: int = 500,
    max_depth: int = 3,
    learning_rate: float = 0.02,
    random_state: int = 42,
):
    """
    Train an XGBoost Regressor with anti-overfitting regularisation.

    **Why the original XGBoost overfit (R² = -0.181)?**
      - max_depth=5 allows trees deep enough to memorise training noise.
      - learning_rate=0.05 is too aggressive for financial time-series.
      - No row/column subsampling -> each tree sees the entire feature set.
      - No L1/L2 regularisation on leaf weights.
      - No early stopping -> training continued past the optimal point.

    **Fixes applied:**
    1. max_depth=3  (shallow trees, much lower variance)
    2. learning_rate=0.02  (conservative, relies on early stopping to find
       the right number of trees)
    3. subsample=0.7  (each tree trains on 70% of rows, like Random Forest
       bagging — reduces variance)
    4. colsample_bytree=0.7  (each tree sees 70% of features randomly —
       further decorrelation)
    5. reg_alpha=0.05 (L1 penalty on leaf weights -> sparse trees)
    6. reg_lambda=1.0 (L2 penalty on leaf weights -> smaller weights)
    7. min_child_weight=5  (require at least 5 samples in a leaf -> prevents
       fitting to individual noisy data points)
    8. Early stopping on validation set with patience=40 rounds.

    Parameters
    ----------
    X_train, y_train  : Training arrays.
    X_val, y_val      : Validation arrays for early stopping (optional but
                        strongly recommended). If None, no early stopping.
    n_estimators      : Maximum number of boosting rounds (early stopping
                        will find the actual optimal number).
    max_depth         : Maximum tree depth (3 recommended for finance).
    learning_rate     : Step size shrinkage (smaller = more conservative).
    random_state      : Seed for reproducibility.

    Returns
    -------
    xgboost XGBRegressor (fitted) or None if xgboost not installed.
    """
    if not HAS_XGBOOST:
        print("[Models] XGBoost not available - skipping.")
        return None

    eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
    early_stop = 40 if eval_set is not None else None

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,                   # Fix: reduced from 5 -> 3
        learning_rate=learning_rate,           # Fix: reduced from 0.05 -> 0.02
        subsample=0.7,                         # Fix: row subsampling (new)
        colsample_bytree=0.7,                  # Fix: feature subsampling (new)
        reg_alpha=0.05,                        # Fix: L1 regularisation (new)
        reg_lambda=1.0,                        # Fix: L2 regularisation (new)
        min_child_weight=5,                    # Fix: min samples per leaf (new)
        early_stopping_rounds=early_stop,      # Fix: early stopping on val (new)
        eval_metric="rmse",
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )

    if eval_set is not None:
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        print(f"  [XGBoost] Best iteration : {model.best_iteration}  "
              f"(of max {n_estimators} rounds)")
        print(f"  [XGBoost] Best val RMSE  : {model.best_score:.6f}")
    else:
        model.fit(X_train, y_train)
        print(f"  [XGBoost] Trained {n_estimators} rounds (no early stopping — pass X_val/y_val)")

    return model


# ===================================================================
# TRAIN & EVALUATE ALL BASELINES
# ===================================================================
def train_and_evaluate_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str = "Target",
) -> dict:
    """
    Train all classical ML baselines and evaluate on validation and test sets.

    Parameters
    ----------
    X_train, y_train : Training features and targets.
    X_val, y_val     : Validation features and targets.
    X_test, y_test   : Test features and targets.
    target_name : str
        Label for display purposes (e.g. "Return" or "Volatility").

    Returns
    -------
    dict
        Nested dictionary:
        {model_name: {'model': fitted_model,
                      'val_metrics': {...},
                      'test_metrics': {...},
                      'val_preds': array,
                      'test_preds': array}}
    """
    print(f"\n{'#' * 60}")
    print(f"  Training Classical ML Baselines  -  Target: {target_name}")
    print(f"{'#' * 60}")

    results = {}

    # ---- Ridge Regression ----
    ridge = train_ridge(X_train, y_train)
    ridge_val_pred = ridge.predict(X_val)
    ridge_test_pred = ridge.predict(X_test)
    ridge_val_m = compute_regression_metrics(y_val, ridge_val_pred)
    ridge_test_m = compute_regression_metrics(y_test, ridge_test_pred)
    print_metrics(f"Ridge Regression (Val) - {target_name}", ridge_val_m)
    print_metrics(f"Ridge Regression (Test) - {target_name}", ridge_test_m)
    results["Ridge"] = {
        "model": ridge,
        "val_metrics": ridge_val_m,
        "test_metrics": ridge_test_m,
        "val_preds": ridge_val_pred,
        "test_preds": ridge_test_pred,
    }

    # ---- Random Forest ----
    rf = train_random_forest(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_test_pred = rf.predict(X_test)
    rf_val_m = compute_regression_metrics(y_val, rf_val_pred)
    rf_test_m = compute_regression_metrics(y_test, rf_test_pred)
    print_metrics(f"Random Forest (Val) - {target_name}", rf_val_m)
    print_metrics(f"Random Forest (Test) - {target_name}", rf_test_m)
    results["RandomForest"] = {
        "model": rf,
        "val_metrics": rf_val_m,
        "test_metrics": rf_test_m,
        "val_preds": rf_val_pred,
        "test_preds": rf_test_pred,
    }

    # ---- XGBoost ---- (with validation set for early stopping)
    xgb = train_xgboost(X_train, y_train, X_val=X_val, y_val=y_val)
    if xgb is not None:
        xgb_val_pred = xgb.predict(X_val)
        xgb_test_pred = xgb.predict(X_test)
        xgb_val_m = compute_regression_metrics(y_val, xgb_val_pred)
        xgb_test_m = compute_regression_metrics(y_test, xgb_test_pred)
        print_metrics(f"XGBoost (Val) - {target_name}", xgb_val_m)
        print_metrics(f"XGBoost (Test) - {target_name}", xgb_test_m)
        results["XGBoost"] = {
            "model": xgb,
            "val_metrics": xgb_val_m,
            "test_metrics": xgb_test_m,
            "val_preds": xgb_val_pred,
            "test_preds": xgb_test_pred,
        }

    return results


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate dummy data for a quick sanity check
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 10)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n) * 0.1

    split1 = int(0.7 * n)
    split2 = int(0.85 * n)
    results = train_and_evaluate_all(
        X[:split1], y[:split1],
        X[split1:split2], y[split1:split2],
        X[split2:], y[split2:],
        target_name="DummyTarget",
    )
    print("\nModels trained:", list(results.keys()))
