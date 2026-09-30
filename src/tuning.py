"""
tuning.py - Hyperparameter Tuning Pipeline
==========================================
Implements time-series-safe hyperparameter grid search for all 4 models.

Validation Protocol (Strict - No Test-Set Leakage)
---------------------------------------------------

  TRAIN (70%)  --+-->  TimeSeriesSplit(n_splits=3) folds
                 |         -> Ridge & RF CV RMSE -> pick best params
                 |
                 +-->  XGBoost grid: each combo uses early stopping on X_VAL
                            -> picks best (max_depth, lr, subsample) by val RMSE
                            -> n_estimators determined by early stopping, not the grid

  VAL   (15%)  ------>  LSTM: each combo trains with early stopping on val DataLoader
                            -> seq_len change triggers DataLoader reconstruction from scratch

  TEST  (15%)  ------>  Evaluated ONCE, ONLY after best params are chosen via val.
                        Never used to select hyperparameters.

Final Comparison
----------------
  Before tuning  -> validation performance  (default hyperparams)
  After tuning   -> validation performance  (tuned hyperparams, same val set)
  Final model    -> test performance        (selected by val, reported on test)
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from lstm_model import LSTMModel, make_dataloader
from training import train_model

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


# ===================================================================
# INTERNAL METRIC UTILITIES
# ===================================================================
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(
        np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    )))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(
        np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    ))

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(
        np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    ))


# ===================================================================
# DEFAULT HYPERPARAMETER GRIDS
# ===================================================================
DEFAULT_RIDGE_GRID = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 10000.0]
}

DEFAULT_RF_GRID = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [5, 10, None],
    "min_samples_leaf":[1, 3, 5],
}

# Reduced XGBoost grid: 3x2x2 = 12 combos.
# n_estimators is NOT in the grid -- chosen automatically by early stopping.
DEFAULT_XGB_GRID = {
    "max_depth":     [2, 3, 5],
    "learning_rate": [0.01, 0.05],
    "subsample":     [0.6, 0.8],
}

# LSTM grid: 3x2x2x2 = 24 combos.
# dropout and batch_size are fixed at best-practice defaults to keep grid manageable.
# seq_len IS in the grid -- each unique value triggers DataLoader reconstruction.
DEFAULT_LSTM_GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers":  [1, 2],
    "lr":          [1e-3, 5e-4],
    "seq_len":     [20, 30],
    "dropout":     [0.2],
    "batch_size":  [32],
}

# Fast-mode grids for quick testing (--fast-tune flag)
FAST_RIDGE_GRID = {"alpha": [0.1, 10.0, 1000.0]}
FAST_RF_GRID    = {"n_estimators": [100, 200], "max_depth": [5, 10],
                   "min_samples_leaf": [1, 3]}
FAST_XGB_GRID   = {"max_depth": [3], "learning_rate": [0.02, 0.05], "subsample": [0.7]}
FAST_LSTM_GRID  = {"hidden_size": [32, 64], "num_layers": [1, 2],
                   "lr": [1e-3], "seq_len": [30], "dropout": [0.2], "batch_size": [32]}


# ===================================================================
# 1. RIDGE TUNING  (TimeSeriesSplit on X_train)
# ===================================================================
def tune_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_splits: int = 3,
    param_grid: dict = None,
) -> dict:
    """
    Tune Ridge alpha using TimeSeriesSplit cross-validation on X_train.

    Selection protocol
    ------------------
    - TimeSeriesSplit(n_splits) divides X_train into n rolling folds.
    - For each alpha: PCA(0.95) + Ridge fitted on fold-train, scored on fold-val.
    - Best alpha = lowest mean CV RMSE across all folds.
    - X_val is NOT used for selection; it is only used to report the
      'after tuning' validation score of the selected model.
    - X_test is NEVER touched.

    Parameters
    ----------
    X_train, y_train : Training arrays (already StandardScaled).
    X_val, y_val     : Validation arrays -- only for final reporting.
    n_splits         : Number of TimeSeriesSplit folds (default 3).
    param_grid       : dict with key 'alpha' (list). Uses DEFAULT_RIDGE_GRID if None.

    Returns
    -------
    dict: best_params, best_cv_rmse, results_df
    """
    if param_grid is None:
        param_grid = DEFAULT_RIDGE_GRID

    alpha_list = param_grid["alpha"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\n  [TuneRidge] {len(alpha_list)} alphas x {n_splits} folds "
          f"= {len(alpha_list) * n_splits} total fits")

    records = []
    for alpha in alpha_list:
        fold_rmses = []
        for tr_idx, vl_idx in tscv.split(X_train):
            X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
            y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
            pipe = Pipeline([
                ("pca",   PCA(n_components=0.95, random_state=42)),
                ("ridge", Ridge(alpha=alpha)),
            ])
            pipe.fit(X_tr, y_tr)
            fold_rmses.append(_rmse(y_vl, pipe.predict(X_vl)))

        cv_mean = float(np.mean(fold_rmses))
        cv_std  = float(np.std(fold_rmses))
        records.append({
            "alpha":        alpha,
            "cv_rmse_mean": round(cv_mean, 7),
            "cv_rmse_std":  round(cv_std, 7),
        })
        print(f"    alpha={alpha:>10}  cv_rmse={cv_mean:.6f} +/- {cv_std:.6f}")

    results_df   = pd.DataFrame(records).sort_values("cv_rmse_mean").reset_index(drop=True)
    best_alpha   = float(results_df.iloc[0]["alpha"])
    best_cv_rmse = float(results_df.iloc[0]["cv_rmse_mean"])
    print(f"  [TuneRidge] Best alpha={best_alpha}  cv_rmse={best_cv_rmse:.6f}")

    return {
        "best_params":  {"alpha": best_alpha, "pca_variance": 0.95},
        "best_cv_rmse": best_cv_rmse,
        "results_df":   results_df,
    }


# ===================================================================
# 2. RANDOM FOREST TUNING  (TimeSeriesSplit on X_train)
# ===================================================================
def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_splits: int = 3,
    param_grid: dict = None,
) -> dict:
    """
    Tune Random Forest (n_estimators, max_depth, min_samples_leaf)
    via TimeSeriesSplit cross-validation on X_train.

    Same protocol as tune_ridge: best combo by mean CV RMSE.
    X_val not used for selection. X_test never touched.

    Returns
    -------
    dict: best_params, best_cv_rmse, results_df
    """
    if param_grid is None:
        param_grid = DEFAULT_RF_GRID

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    tscv   = TimeSeriesSplit(n_splits=n_splits)

    print(f"\n  [TuneRF] {len(combos)} combos x {n_splits} folds "
          f"= {len(combos) * n_splits} total fits")

    records = []
    for i, vals in enumerate(combos, 1):
        cfg        = dict(zip(keys, vals))
        fold_rmses = []

        for tr_idx, vl_idx in tscv.split(X_train):
            X_tr, X_vl = X_train[tr_idx], X_train[vl_idx]
            y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]
            rf = RandomForestRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                random_state=42, n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            fold_rmses.append(_rmse(y_vl, rf.predict(X_vl)))

        cv_mean = float(np.mean(fold_rmses))
        cv_std  = float(np.std(fold_rmses))
        row = {**cfg, "cv_rmse_mean": round(cv_mean, 7), "cv_rmse_std": round(cv_std, 7)}
        records.append(row)

        if i % 5 == 0 or i == 1 or i == len(combos):
            print(f"    [{i:>2}/{len(combos)}] {cfg}  cv_rmse={cv_mean:.6f}")

    results_df = pd.DataFrame(records).sort_values("cv_rmse_mean").reset_index(drop=True)
    best = results_df.iloc[0]

    raw_depth  = best["max_depth"]
    best_depth = (None if (raw_depth is None or
                           (isinstance(raw_depth, float) and np.isnan(raw_depth)))
                  else int(raw_depth))
    best_params = {
        "n_estimators":    int(best["n_estimators"]),
        "max_depth":       best_depth,
        "min_samples_leaf":int(best["min_samples_leaf"]),
    }
    print(f"  [TuneRF] Best: {best_params}  cv_rmse={best['cv_rmse_mean']:.6f}")

    return {
        "best_params":  best_params,
        "best_cv_rmse": float(best["cv_rmse_mean"]),
        "results_df":   results_df,
    }


# ===================================================================
# 3. XGBOOST TUNING  (Early stopping on X_val, 12-combo reduced grid)
# ===================================================================
_XGB_FIXED = {
    "colsample_bytree": 0.7,
    "reg_alpha":        0.05,
    "reg_lambda":       1.0,
    "min_child_weight": 5,
}

def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict = None,
    n_estimators_max: int = 500,
    early_stopping_rounds: int = 40,
) -> dict:
    """
    Tune XGBoost (max_depth, learning_rate, subsample) using X_val as
    the early-stopping evaluation set.

    Why X_val for XGBoost (not TimeSeriesSplit)?
    --------------------------------------------
    XGBoost's early stopping requires a single fixed eval_set checked after
    every boosting round. This is incompatible with K-fold CV, which needs a
    fresh model per fold and cannot combine per-round stopping across folds.
    The correct approach is:

      For each (max_depth, lr, subsample) combo:
        1. Train with n_estimators_max maximum rounds.
        2. Early stopping halts when X_val RMSE stops improving for
           `early_stopping_rounds` consecutive rounds -- automatically
           selecting the optimal n_estimators.
        3. Record val RMSE at best_iteration.

    n_estimators is NOT a grid parameter -- it is selected per-combo
    by early stopping. Grid: 3x2x2 = 12 combos (all fast).
    X_test is NEVER touched.

    Returns
    -------
    dict: best_params (includes n_estimators from early stopping), results_df
    """
    if not HAS_XGBOOST:
        print("  [TuneXGB] XGBoost not installed -- skipping.")
        return {"best_params": None, "best_val_rmse": np.nan,
                "results_df": pd.DataFrame()}

    if param_grid is None:
        param_grid = DEFAULT_XGB_GRID

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"\n  [TuneXGB] {len(combos)} combos with early stopping "
          f"(max {n_estimators_max} rounds, patience={early_stopping_rounds})")
    print(f"  [TuneXGB] Fixed params: {_XGB_FIXED}")

    records = []
    for i, vals in enumerate(combos, 1):
        cfg = dict(zip(keys, vals))
        model = XGBRegressor(
            n_estimators=n_estimators_max,
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=_XGB_FIXED["colsample_bytree"],
            reg_alpha=_XGB_FIXED["reg_alpha"],
            reg_lambda=_XGB_FIXED["reg_lambda"],
            min_child_weight=_XGB_FIXED["min_child_weight"],
            early_stopping_rounds=early_stopping_rounds,
            eval_metric="rmse",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = int(model.best_iteration)
        val_rmse  = float(model.best_score)
        val_preds = model.predict(X_val)

        row = {
            **cfg,
            "n_estimators_best": best_iter,
            "val_rmse": round(val_rmse, 7),
            "val_mae":  round(_mae(y_val, val_preds), 7),
            "val_r2":   round(_r2(y_val, val_preds), 5),
        }
        records.append(row)
        print(f"    [{i:>2}/{len(combos)}] depth={cfg['max_depth']} "
              f"lr={cfg['learning_rate']} sub={cfg['subsample']}  "
              f"best_iter={best_iter:>4}  val_rmse={val_rmse:.6f}")

    results_df = pd.DataFrame(records).sort_values("val_rmse").reset_index(drop=True)
    best = results_df.iloc[0]
    best_params = {
        "n_estimators":    int(best["n_estimators_best"]),
        "max_depth":       int(best["max_depth"]),
        "learning_rate":   float(best["learning_rate"]),
        "subsample":       float(best["subsample"]),
        "colsample_bytree":_XGB_FIXED["colsample_bytree"],
        "reg_alpha":       _XGB_FIXED["reg_alpha"],
        "reg_lambda":      _XGB_FIXED["reg_lambda"],
        "min_child_weight":_XGB_FIXED["min_child_weight"],
    }
    print(f"  [TuneXGB] Best: {best_params}  val_rmse={best['val_rmse']:.6f}")

    return {
        "best_params":   best_params,
        "best_val_rmse": float(best["val_rmse"]),
        "results_df":    results_df,
    }


# ===================================================================
# 4. LSTM TUNING  (Per-seq_len DataLoader rebuild + early stopping)
# ===================================================================
def tune_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    device: torch.device,
    param_grid: dict = None,
    max_epochs: int = 60,
    patience: int = 8,
) -> dict:
    """
    Tune LSTM (hidden_size, num_layers, lr, seq_len, dropout, batch_size)
    using the fixed chronological validation set with early stopping.

    Critical implementation: seq_len requires DataLoader reconstruction
    -------------------------------------------------------------------
    seq_len is NOT a simple model parameter -- it changes the shape of the
    input tensor (batch, seq_len, features) and the sliding-window indexing
    of the dataset. Passing a different seq_len to an existing DataLoader
    does NOT work and produces inputs with the wrong window length.

    Fix: Track (current_seq_len, current_batch_size) between combos.
         If EITHER changes -> call make_dataloader() to rebuild both
         train_loader and val_loader from scratch before training.
         If neither changes -> reuse existing loaders (saves significant time).

    X_test is NEVER touched in this function.

    Parameters
    ----------
    X_train, y_train : Training arrays (already StandardScaled).
    X_val, y_val     : Validation arrays (used for early stopping + selection).
    input_size       : Number of features per time step.
    device           : torch.device (cpu or cuda).
    param_grid       : dict with keys: hidden_size, num_layers, lr,
                       seq_len, dropout, batch_size.
    max_epochs       : Max training epochs per combo.
    patience         : Early stopping patience (in epochs).

    Returns
    -------
    dict: best_params, best_val_loss, results_df
    """
    if param_grid is None:
        param_grid = DEFAULT_LSTM_GRID

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"\n  [TuneLSTM] {len(combos)} combos  "
          f"(max_epochs={max_epochs}, patience={patience})")
    print("  [TuneLSTM] DataLoaders are rebuilt for each unique (seq_len, batch_size).")

    records            = []
    current_seq_len    = None
    current_batch_size = None
    train_loader       = None
    val_loader         = None

    for i, vals in enumerate(combos, 1):
        cfg        = dict(zip(keys, vals))
        seq_len    = int(cfg["seq_len"])
        hidden     = int(cfg["hidden_size"])
        n_layers   = int(cfg["num_layers"])
        lr         = float(cfg["lr"])
        dropout    = float(cfg["dropout"])
        batch_size = int(cfg["batch_size"])

        # Rebuild DataLoaders only when seq_len OR batch_size changes
        if seq_len != current_seq_len or batch_size != current_batch_size:
            print(f"    [TuneLSTM] Rebuilding DataLoaders: "
                  f"seq_len={seq_len}, batch_size={batch_size}")
            train_loader = make_dataloader(
                X_train, y_train,
                seq_len=seq_len,
                batch_size=batch_size,
                shuffle=False,
            )
            val_loader = make_dataloader(
                X_val, y_val,
                seq_len=seq_len,
                batch_size=batch_size,
                shuffle=False,
            )
            current_seq_len    = seq_len
            current_batch_size = batch_size

        # Build a fresh model for each combo
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout,
        )

        # Train with early stopping on val_loader
        t0 = time.time()
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=max_epochs,
            lr=lr,
            patience=patience,
            weight_decay=1e-4,
            device=device,
            verbose=False,
        )
        elapsed = time.time() - t0

        best_val_loss  = float(result["best_val_loss"])
        epochs_trained = int(result["epochs_trained"])

        row = {
            "hidden_size":   hidden,
            "num_layers":    n_layers,
            "lr":            lr,
            "seq_len":       seq_len,
            "dropout":       dropout,
            "batch_size":    batch_size,
            "best_val_loss": round(best_val_loss, 8),
            "epochs_trained":epochs_trained,
            "time_s":        round(elapsed, 1),
        }
        records.append(row)
        print(f"    [{i:>2}/{len(combos)}] h={hidden} L={n_layers} lr={lr} "
              f"seq={seq_len} drop={dropout}  "
              f"val_loss={best_val_loss:.6f}  ({epochs_trained}ep, {elapsed:.0f}s)")

    results_df = pd.DataFrame(records).sort_values("best_val_loss").reset_index(drop=True)
    best       = results_df.iloc[0]
    best_params = {
        "hidden_size": int(best["hidden_size"]),
        "num_layers":  int(best["num_layers"]),
        "lr":          float(best["lr"]),
        "seq_len":     int(best["seq_len"]),
        "dropout":     float(best["dropout"]),
        "batch_size":  int(best["batch_size"]),
    }
    print(f"  [TuneLSTM] Best: {best_params}  "
          f"val_loss={best['best_val_loss']:.6f}")

    return {
        "best_params":   best_params,
        "best_val_loss": float(best["best_val_loss"]),
        "results_df":    results_df,
    }


# ===================================================================
# 5. ORCHESTRATOR
# ===================================================================
def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_size: int,
    device: torch.device,
    baseline_classical_results: dict,
    baseline_lstm_val_metrics: dict,
    fast_mode: bool = False,
) -> dict:
    """
    Full hyperparameter tuning orchestrator for all 4 models.

    Strict three-way comparison (no test-set leakage):
    ---------------------------------------------------
    Before tuning  -> validation RMSE  (default hyperparams, on X_val)
    After tuning   -> validation RMSE  (tuned hyperparams, on same X_val)
    Final model    -> test RMSE        (selected by val, evaluated on X_test)

    Hyperparameter selection is NEVER based on test performance.

    Steps
    -----
    1. Tune Ridge, RF, XGB, LSTM using only X_train + X_val.
    2. Retrain classical models (Ridge/RF/XGB) with best params on X_train.
       LSTM: best_lstm_params returned for main pipeline to retrain with
       correctly rebuilt DataLoaders.
    3. Evaluate tuned classical models on X_val  -> 'After Tuning Val' score.
    4. Evaluate tuned classical models on X_test -> 'Final Test' score.
    5. Save comparison CSV (LSTM After row filled later via
       update_lstm_tuning_results() after main retrains LSTM).

    Parameters
    ----------
    baseline_classical_results : Output of models.train_and_evaluate_all()
                                 with DEFAULT params. Provides 'Before' val scores.
    baseline_lstm_val_metrics  : compute_regression_metrics() on LSTM val preds
                                 with DEFAULT params. Provides 'Before' LSTM val score.
    fast_mode                  : Use reduced grids for quick testing.

    Returns
    -------
    dict:
        best_params      : {model_name: best_params_dict}
        tuned_models     : {model_name: fitted classical model}
        comparison_df    : pd.DataFrame (before/after table, LSTM After=None)
        best_lstm_params : dict for main pipeline to retrain LSTM
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if fast_mode:
        print("  [Tuning] Fast mode -- using reduced grids.")
        ridge_grid, rf_grid = FAST_RIDGE_GRID, FAST_RF_GRID
        xgb_grid,   lstm_grid = FAST_XGB_GRID, FAST_LSTM_GRID
    else:
        ridge_grid = rf_grid = xgb_grid = lstm_grid = None  # use defaults

    print("\n" + "=" * 70)
    print("  HYPERPARAMETER TUNING PIPELINE")
    print("  Ridge/RF : TimeSeriesSplit(3) on X_train -- test set untouched")
    print("  XGBoost  : 12-combo grid + early stopping on X_val")
    print("  LSTM     : Grid search; seq_len change rebuilds DataLoaders")
    print("  Test set : NEVER used for selection -- final reporting ONLY")
    print("=" * 70)

    all_best_params: dict = {}
    tuning_dfs:      dict = {}

    # ------------------------------------------------------------------
    print(f"\n{'--'*35}")
    print("  [1/4] Tuning Ridge Regression")
    print(f"{'--'*35}")
    t0 = time.time()
    ridge_tune = tune_ridge(X_train, y_train, X_val, y_val, param_grid=ridge_grid)
    all_best_params["Ridge"] = ridge_tune["best_params"]
    tuning_dfs["Ridge"]      = ridge_tune["results_df"]
    print(f"  Ridge tuning done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print(f"\n{'--'*35}")
    print("  [2/4] Tuning Random Forest")
    print(f"{'--'*35}")
    t0 = time.time()
    rf_tune = tune_random_forest(X_train, y_train, X_val, y_val, param_grid=rf_grid)
    all_best_params["RandomForest"] = rf_tune["best_params"]
    tuning_dfs["RandomForest"]      = rf_tune["results_df"]
    print(f"  RF tuning done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print(f"\n{'--'*35}")
    print("  [3/4] Tuning XGBoost")
    print(f"{'--'*35}")
    t0 = time.time()
    xgb_tune = tune_xgboost(X_train, y_train, X_val, y_val, param_grid=xgb_grid)
    all_best_params["XGBoost"] = xgb_tune.get("best_params")
    tuning_dfs["XGBoost"]      = xgb_tune.get("results_df", pd.DataFrame())
    print(f"  XGB tuning done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print(f"\n{'--'*35}")
    print("  [4/4] Tuning LSTM")
    print(f"{'--'*35}")
    t0 = time.time()
    lstm_tune = tune_lstm(
        X_train, y_train, X_val, y_val,
        input_size=input_size, device=device,
        param_grid=lstm_grid,
    )
    all_best_params["LSTM"] = lstm_tune["best_params"]
    tuning_dfs["LSTM"]      = lstm_tune["results_df"]
    print(f"  LSTM tuning done in {time.time()-t0:.1f}s")

    # Save per-model tuning logs
    print(f"\n{'--'*35}")
    for name, df in tuning_dfs.items():
        if not df.empty:
            path = os.path.join(RESULTS_DIR, f"tuning_{name.lower()}.csv")
            df.to_csv(path, index=False)
            print(f"  [Saved] {name} log -> {path}")

    # Save best-params summary
    summary_rows = [
        {"Model": m, **({} if bp is None else {str(k): str(v) for k, v in bp.items()})}
        for m, bp in all_best_params.items()
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "tuning_summary.csv"), index=False)
    print(f"  [Saved] Best params -> results/tuning_summary.csv")

    # ------------------------------------------------------------------
    # Retrain classical models with best params on full X_train
    # ------------------------------------------------------------------
    print(f"\n{'--'*35}")
    print("  Retraining with best hyperparameters...")
    print(f"{'--'*35}")
    tuned_models: dict = {}

    rp = all_best_params["Ridge"]
    ridge_tuned = Pipeline([
        ("pca",   PCA(n_components=rp["pca_variance"], random_state=42)),
        ("ridge", Ridge(alpha=rp["alpha"])),
    ])
    ridge_tuned.fit(X_train, y_train)
    tuned_models["Ridge"] = ridge_tuned
    print(f"  Ridge    alpha={rp['alpha']}")

    rfp = all_best_params["RandomForest"]
    rf_tuned = RandomForestRegressor(
        n_estimators=rfp["n_estimators"],
        max_depth=rfp["max_depth"],
        min_samples_leaf=rfp["min_samples_leaf"],
        random_state=42, n_jobs=-1,
    )
    rf_tuned.fit(X_train, y_train)
    tuned_models["RandomForest"] = rf_tuned
    print(f"  RF       {rfp}")

    tuned_models["XGBoost"] = None
    if HAS_XGBOOST and all_best_params.get("XGBoost"):
        xp = all_best_params["XGBoost"]
        xgb_tuned = XGBRegressor(
            n_estimators=xp["n_estimators"],
            max_depth=xp["max_depth"],
            learning_rate=xp["learning_rate"],
            subsample=xp["subsample"],
            colsample_bytree=xp["colsample_bytree"],
            reg_alpha=xp["reg_alpha"],
            reg_lambda=xp["reg_lambda"],
            min_child_weight=xp["min_child_weight"],
            random_state=42, verbosity=0, n_jobs=-1,
        )
        xgb_tuned.fit(X_train, y_train)
        tuned_models["XGBoost"] = xgb_tuned
        print(f"  XGBoost  depth={xp['max_depth']} lr={xp['learning_rate']} "
              f"n_est={xp['n_estimators']}")

    lp = all_best_params["LSTM"]
    print(f"  LSTM     params returned to main: {lp}")

    # ------------------------------------------------------------------
    # Build before / after / test comparison table
    # ------------------------------------------------------------------
    print(f"\n{'--'*35}")
    print("  Building comparison table...")
    print(f"{'--'*35}")
    comparison_rows = []

    for model_name in ["Ridge", "RandomForest", "XGBoost"]:
        bl = baseline_classical_results.get(model_name, {}).get("val_metrics", {})
        before_val_rmse = float(bl.get("RMSE", np.nan))
        before_val_r2   = float(bl.get("R2",   np.nan))

        tm = tuned_models.get(model_name)
        if tm is not None:
            val_preds  = tm.predict(X_val)
            test_preds = tm.predict(X_test)
            after_val_rmse  = _rmse(y_val, val_preds)
            after_val_r2    = _r2(y_val, val_preds)
            final_test_rmse = _rmse(y_test, test_preds)
            final_test_r2   = _r2(y_test, test_preds)
            improvement_pct = (
                (before_val_rmse - after_val_rmse) / before_val_rmse * 100
                if (not np.isnan(before_val_rmse) and before_val_rmse > 0) else np.nan
            )
        else:
            after_val_rmse = after_val_r2 = final_test_rmse = final_test_r2 = np.nan
            improvement_pct = np.nan

        def _safe_round(v, n):
            return round(float(v), n) if not np.isnan(float(v) if not isinstance(v, float) else v) else np.nan

        comparison_rows.append({
            "Model":                  model_name,
            "Best_Params":            str(all_best_params.get(model_name, {})),
            "Before_Val_RMSE":        _safe_round(before_val_rmse, 6),
            "After_Val_RMSE":         _safe_round(after_val_rmse, 6),
            "Val_RMSE_Improvement_%": _safe_round(improvement_pct, 2),
            "Before_Val_R2":          _safe_round(before_val_r2, 4),
            "After_Val_R2":           _safe_round(after_val_r2, 4),
            "Final_Test_RMSE":        _safe_round(final_test_rmse, 6),
            "Final_Test_R2":          _safe_round(final_test_r2, 4),
        })

    # LSTM row -- After/Test columns left None, filled by update_lstm_tuning_results()
    lstm_before_rmse = float(baseline_lstm_val_metrics.get("RMSE", np.nan))
    lstm_before_r2   = float(baseline_lstm_val_metrics.get("R2",   np.nan))
    comparison_rows.append({
        "Model":                  "LSTM",
        "Best_Params":            str(all_best_params.get("LSTM", {})),
        "Before_Val_RMSE":        round(lstm_before_rmse, 6) if not np.isnan(lstm_before_rmse) else np.nan,
        "After_Val_RMSE":         None,
        "Val_RMSE_Improvement_%": None,
        "Before_Val_R2":          round(lstm_before_r2, 4)   if not np.isnan(lstm_before_r2)   else np.nan,
        "After_Val_R2":           None,
        "Final_Test_RMSE":        None,
        "Final_Test_R2":          None,
    })

    comparison_df = pd.DataFrame(comparison_rows)
    comp_path     = os.path.join(RESULTS_DIR, "before_vs_after_tuning.csv")
    comparison_df.to_csv(comp_path, index=False)

    # Print classical results (LSTM After pending)
    print("\n" + "=" * 95)
    print("  TUNING RESULTS  --  BEFORE vs AFTER  (VALIDATION SET)")
    print("  Selection: val performance ONLY  |  Test shown for information")
    print("=" * 95)
    display = ["Model", "Before_Val_RMSE", "After_Val_RMSE",
               "Val_RMSE_Improvement_%", "Before_Val_R2", "After_Val_R2",
               "Final_Test_RMSE"]
    mask = comparison_df["Model"] != "LSTM"
    print(comparison_df.loc[mask, display].to_string(index=False))
    print("  [LSTM: after-tuning scores added after main retrains with best params]")
    print("=" * 95)

    return {
        "best_params":      all_best_params,
        "tuned_models":     tuned_models,
        "comparison_df":    comparison_df,
        "best_lstm_params": all_best_params.get("LSTM", {}),
    }


# ===================================================================
# 6. LSTM ROW UPDATER  (called from main after LSTM retraining)
# ===================================================================
def update_lstm_tuning_results(
    after_val_rmse: float,
    after_val_r2: float,
    final_test_rmse: float,
    final_test_r2: float,
) -> None:
    """
    Fill the LSTM 'After Tuning' and 'Final Test' columns in the comparison CSV.

    Called by main.py AFTER LSTM is retrained with best_lstm_params and
    evaluated on both val and test sets.

    This enforces the protocol:
      - Hyperparams selected by val (not test)
      - Test score reported only at the end, never fed back into selection
    """
    comp_path = os.path.join(RESULTS_DIR, "before_vs_after_tuning.csv")
    if not os.path.exists(comp_path):
        print("  [Tuning] before_vs_after_tuning.csv not found -- skipping LSTM update.")
        return

    df        = pd.read_csv(comp_path)
    lstm_mask = df["Model"] == "LSTM"
    if lstm_mask.sum() == 0:
        return

    before_rmse = df.loc[lstm_mask, "Before_Val_RMSE"].iloc[0]
    try:
        improvement = (float(before_rmse) - after_val_rmse) / float(before_rmse) * 100
    except Exception:
        improvement = np.nan

    df.loc[lstm_mask, "After_Val_RMSE"]         = round(after_val_rmse, 6)
    df.loc[lstm_mask, "Val_RMSE_Improvement_%"]  = round(improvement, 2)
    df.loc[lstm_mask, "After_Val_R2"]            = round(after_val_r2, 4)
    df.loc[lstm_mask, "Final_Test_RMSE"]         = round(final_test_rmse, 6)
    df.loc[lstm_mask, "Final_Test_R2"]           = round(final_test_r2, 4)
    df.to_csv(comp_path, index=False)

    print("\n" + "=" * 95)
    print("  FINAL TUNING COMPARISON -- ALL MODELS")
    print("  Before tuning  -> Validation RMSE (default hyperparams)")
    print("  After tuning   -> Validation RMSE (tuned, selected by val NOT test)")
    print("  Final Test     -> Test RMSE        (reported, never used for selection)")
    print("=" * 95)
    display = ["Model", "Before_Val_RMSE", "After_Val_RMSE",
               "Val_RMSE_Improvement_%", "Final_Test_RMSE", "Final_Test_R2"]
    print(df[display].to_string(index=False))
    print("=" * 95)
    print(f"  [Saved] Complete comparison -> {comp_path}")
