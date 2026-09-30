"""
feature_selection.py - Automated Feature Selection Module
==========================================================
Implements a two-step feature selection pipeline:

    Step 1 — Correlation Pruning:
        Drop features that are >90% correlated with another feature
        already retained.  This eliminates redundancy (e.g. SMA_5 and
        SMA_10 moving in near-lockstep) without losing information.

    Step 2 — Tree-Based Importance Ranking:
        Train a quick RandomForest on the pruned feature set, rank
        features by Gini importance, and keep only the top-K.

Design Constraints
------------------
*   Feature selection is performed using ONLY the training split to
    prevent any information leakage from validation or test data.
*   The module is stateless — it returns a filtered list of column
    names that the caller can use to slice the DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ===================================================================
# STEP 1: CORRELATION PRUNING
# ===================================================================
def remove_correlated_features(
    df: pd.DataFrame,
    feature_cols: list,
    threshold: float = 0.90,
) -> list:
    """
    Remove features whose pairwise Pearson correlation exceeds
    `threshold`.

    Algorithm
    ---------
    1.  Compute the absolute correlation matrix of all features.
    2.  Iterate over the upper triangle.  For every pair (A, B)
        with |corr| > threshold, drop the feature that has the
        higher mean absolute correlation with all other features
        (i.e. the "more redundant" one).

    Parameters
    ----------
    df           : DataFrame containing the feature columns.
    feature_cols : List of feature column names to evaluate.
    threshold    : Correlation cutoff (default 0.90).

    Returns
    -------
    remaining : List of feature column names that survived pruning.
    """
    corr_matrix = df[feature_cols].corr().abs()

    # Upper triangle mask (exclude diagonal)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    # For each feature, compute mean correlation with all others
    mean_corr = corr_matrix.mean(axis=1)

    to_drop = set()
    for col in upper.columns:
        # Find features correlated above threshold with this column
        highly_correlated = upper.index[upper[col] > threshold].tolist()
        for other in highly_correlated:
            # Drop whichever of the pair has higher mean correlation
            if mean_corr[col] > mean_corr[other]:
                to_drop.add(col)
            else:
                to_drop.add(other)

    remaining = [c for c in feature_cols if c not in to_drop]

    print(f"  [Feature Selection] Correlation pruning (threshold={threshold}):")
    print(f"    Dropped {len(to_drop)} features: {sorted(to_drop)}")
    print(f"    Remaining: {len(remaining)} features")

    return remaining


# ===================================================================
# STEP 2: TREE-BASED IMPORTANCE RANKING
# ===================================================================
def select_top_k_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list,
    k: int = 10,
    random_state: int = 42,
) -> list:
    """
    Train a lightweight Random Forest on the training data and
    return the top-K features ranked by Gini importance.

    Parameters
    ----------
    X_train      : Training feature matrix (n_samples, n_features).
    y_train      : Training target vector.
    feature_cols : List of feature column names (same order as columns
                   in X_train).
    k            : Number of top features to retain (default 10).
    random_state : Seed for reproducibility.

    Returns
    -------
    selected : List of the top-K feature column names, ordered by
               descending importance.
    """
    # Clamp k to the number of available features
    k = min(k, len(feature_cols))

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    selected = importances.head(k).index.tolist()

    print(f"  [Feature Selection] Top-{k} features by RF importance:")
    for i, (feat, imp) in enumerate(importances.head(k).items(), 1):
        print(f"    {i:>2d}. {feat:<25s}  importance = {imp:.4f}")

    dropped_here = [c for c in feature_cols if c not in selected]
    if dropped_here:
        print(f"    Dropped {len(dropped_here)} low-importance features: {dropped_here}")

    return selected


# ===================================================================
# COMBINED PIPELINE
# ===================================================================
def run_feature_selection(
    df: pd.DataFrame,
    feature_cols: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    corr_threshold: float = 0.90,
    top_k: int = 10,
) -> list:
    """
    Full two-step feature selection pipeline.

    1. Correlation pruning on the full DataFrame.
    2. RF importance ranking using ONLY training data.

    Parameters
    ----------
    df             : Full DataFrame (used for correlation computation).
    feature_cols   : Initial list of all feature column names.
    X_train        : Training feature matrix (columns match feature_cols).
    y_train        : Training target vector.
    corr_threshold : Pearson correlation cutoff for pruning.
    top_k          : Number of best features to retain after ranking.

    Returns
    -------
    final_features : List of selected feature column names.
    """
    print("\n" + "=" * 60)
    print("  AUTOMATED FEATURE SELECTION")
    print("=" * 60)
    print(f"  Starting with {len(feature_cols)} features")

    # Step 1: Correlation Pruning
    pruned_cols = remove_correlated_features(df, feature_cols, corr_threshold)

    # Step 2: RF Importance (on pruned features only)
    # Re-index X_train to match pruned columns
    pruned_indices = [feature_cols.index(c) for c in pruned_cols]
    X_train_pruned = X_train[:, pruned_indices]

    final_features = select_top_k_features(
        X_train_pruned, y_train, pruned_cols, k=top_k
    )

    print(f"\n  [OK] Final selection: {len(final_features)} features")
    print("=" * 60)

    return final_features
