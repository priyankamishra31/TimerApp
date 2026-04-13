"""
feature_selection.py
--------------------
Deterministic feature selection pipeline:
  1. Drop highly correlated features
  2. Rank remaining features by Mutual Information
  3. Resolve MI ties using secondary criteria:
       - Lower missingness
       - Lower average correlation within tie group
       - Alphabetical fallback
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# ─────────────────────────────────────────────
# Step 1: Drop highly correlated features
# ─────────────────────────────────────────────

def drop_correlated_features(X: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Drop features with pairwise absolute correlation above `threshold`.
    Keeps the first feature encountered in each correlated pair.

    Parameters
    ----------
    X         : Feature DataFrame
    threshold : Correlation cutoff (default 0.90)

    Returns
    -------
    X with correlated features removed.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"[Correlation Drop] Removed {len(to_drop)} feature(s): {to_drop}")
    return X.drop(columns=to_drop)


# ─────────────────────────────────────────────
# Step 2: Compute MI scores
# ─────────────────────────────────────────────

def compute_mi_scores(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute Mutual Information scores for each feature.

    Parameters
    ----------
    X            : Feature DataFrame
    y            : Target Series
    task         : 'classification' or 'regression'
    random_state : Seed for reproducibility

    Returns
    -------
    DataFrame with columns: feature, mi, missingness
    """
    if task == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=random_state)
    elif task == "regression":
        mi_scores = mutual_info_regression(X, y, random_state=random_state)
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    mi_df = pd.DataFrame({
        "feature":     X.columns.tolist(),
        "mi":          mi_scores,
        "missingness": X.isnull().mean().values,
    })

    return mi_df


# ─────────────────────────────────────────────
# Step 3: Resolve ties with secondary criteria
# ─────────────────────────────────────────────

def resolve_ties(
    mi_df: pd.DataFrame,
    X: pd.DataFrame,
    tie_tolerance: float = 1e-4,
) -> pd.DataFrame:
    """
    Within each MI tie group, break ties using:
      1. Lower missingness  (prefer cleaner features)
      2. Lower avg correlation with others in the tie group  (prefer unique features)
      3. Alphabetical name  (pure determinism fallback)

    Parameters
    ----------
    mi_df         : Output of compute_mi_scores()
    X             : Original feature DataFrame (used for correlation)
    tie_tolerance : MI difference below which features are considered tied

    Returns
    -------
    Fully ranked DataFrame sorted by MI desc, ties resolved deterministically.
    """
    mi_df = mi_df.copy()
    mi_df["mi_group"] = (mi_df["mi"] / tie_tolerance).round() * tie_tolerance

    resolved_groups = []

    for _, group in mi_df.groupby("mi_group", sort=False):
        group = group.copy()

        if len(group) == 1:
            resolved_groups.append(group)
            continue

        # Secondary criterion 1: missingness (lower is better)
        group["miss_rank"] = group["missingness"].rank(method="first")

        # Secondary criterion 2: avg absolute correlation within tie group
        group_features = group["feature"].tolist()
        corr_matrix = X[group_features].corr().abs()
        n = len(group_features)
        # Subtract 1/n to remove self-correlation contribution
        avg_corr = (corr_matrix.sum(axis=1) - 1) / max(n - 1, 1)
        group["avg_corr"] = avg_corr.values
        group["corr_rank"] = group["avg_corr"].rank(method="first")

        # Sort: miss_rank → corr_rank → alphabetical
        group = group.sort_values(
            ["miss_rank", "corr_rank", "feature"],
            ascending=[True, True, True],
        )

        resolved_groups.append(group)

    result = pd.concat(resolved_groups, ignore_index=True)
    result = result.sort_values("mi", ascending=False).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────

def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int,
    task: str = "classification",
    corr_threshold: float = 0.90,
    random_state: int = 42,
    tie_tolerance: float = 1e-4,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full deterministic feature selection pipeline.

    Steps
    -----
    1. Drop highly correlated features
    2. Rank by Mutual Information (seeded)
    3. Resolve ties via missingness → avg correlation → alphabetical

    Parameters
    ----------
    X              : Feature DataFrame
    y              : Target Series
    top_n          : Number of features to select
    task           : 'classification' or 'regression'
    corr_threshold : Pairwise correlation cutoff for step 1
    random_state   : Seed — set once, apply everywhere
    tie_tolerance  : MI delta below which features are considered tied
    verbose        : Print ranking summary if True

    Returns
    -------
    X_selected : DataFrame with top_n features
    ranking_df : Full ranked DataFrame for inspection
    """
    # Step 1
    X_filtered = drop_correlated_features(X, threshold=corr_threshold)

    # Step 2
    mi_df = compute_mi_scores(X_filtered, y, task=task, random_state=random_state)

    # Step 3
    ranking_df = resolve_ties(mi_df, X_filtered, tie_tolerance=tie_tolerance)

    # Select top N
    top_features = ranking_df.head(top_n)["feature"].tolist()
    X_selected = X_filtered[top_features]

    if verbose:
        display_cols = ["feature", "mi", "missingness"]
        if "avg_corr" in ranking_df.columns:
            display_cols.append("avg_corr")
        print("\n[Feature Ranking — Top features]")
        print(ranking_df[display_cols].head(top_n + 5).to_string(index=True))
        print(f"\n→ Selected {top_n} features: {top_features}")

    return X_selected, ranking_df


# ─────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X_raw, y_raw = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    X_raw = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(20)])
    y_raw = pd.Series(y_raw)

    X_selected, ranking = select_features(
        X=X_raw,
        y=y_raw,
        top_n=10,
        task="classification",
        corr_threshold=0.90,
        random_state=42,
        tie_tolerance=1e-4,
        verbose=True,
    )
