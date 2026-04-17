"""
feature_selector.py — Stepwise, forward, backward, and manual selection.

Mirrors SAS SELECTION=STEPWISE/FORWARD/BACKWARD exactly:
  - p_entry  → SAS SLENTRY
  - p_removal → SAS SLSTAY
  - Every iteration logged: step, variable, action, R², Adj R², AIC, BIC, p-value
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config_loader import PipelineConfig
from .exceptions import SelectionError
from .utils import format_p_value, get_logger

logger = get_logger("feature_selector")


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Run feature selection per config.feature_selection.method.
    Returns (selected_variables, iteration_log_df).

    iteration_log_df columns:
      Step | Variable | Action | R2 | Adj_R2 | AIC | BIC | P_Value
    """
    method = config.feature_selection.method

    if method == "manual":
        return _manual(X, y, config)
    elif method == "forward":
        return _forward(X, y, config)
    elif method == "backward":
        return _backward(X, y, config)
    else:
        return _stepwise(X, y, config)


# ── Selection algorithms ──────────────────────────────────────────────────────

def _stepwise(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Stepwise selection — forward pass then backward pass each iteration.
    Mirrors SAS SELECTION=STEPWISE (SLENTRY / SLSTAY).
    """
    p_entry   = config.feature_selection.p_entry
    p_removal = config.feature_selection.p_removal
    max_iter  = config.feature_selection.max_iterations

    included: List[str] = []
    log: list[dict] = []
    step = 0
    prev_included: List[str] | None = None

    logger.info(
        f"Stepwise selection — "
        f"p_entry={p_entry}, p_removal={p_removal}, "
        f"criterion={config.feature_selection.criterion}"
    )

    for _ in range(max_iter):
        # ── Forward step ──────────────────────────────────────────────
        excluded = [c for c in X.columns if c not in included]
        best_p, best_var = 1.0, None

        for var in excluded:
            p = _pvalue_of(X[included + [var]], y, var)
            if p < best_p:
                best_p, best_var = p, var

        if best_var and best_p < p_entry:
            step += 1
            included.append(best_var)
            stats = _model_stats(X[included], y)
            log.append(_log_row(step, best_var, "ENTER", best_p, stats))
            logger.info(
                f"Step {step}: ENTER '{best_var}' "
                f"p={format_p_value(best_p)} R²={stats['R2']:.4f}"
            )
        else:
            logger.info("No variable meets entry criterion — stepwise stopping.")
            break

        # ── Backward step ─────────────────────────────────────────────
        if len(included) > 1:
            worst_p, worst_var = 0.0, None
            for var in included:
                p = _pvalue_of(X[included], y, var)
                if p > worst_p:
                    worst_p, worst_var = p, var

            if worst_var and worst_p > p_removal:
                step += 1
                included.remove(worst_var)
                stats = _model_stats(X[included], y)
                log.append(_log_row(step, worst_var, "REMOVE", worst_p, stats))
                logger.info(
                    f"Step {step}: REMOVE '{worst_var}' "
                    f"p={format_p_value(worst_p)}"
                )

        # Convergence check
        if set(included) == (set(prev_included) if prev_included else set()):
            break
        prev_included = included[:]

    if not included:
        raise SelectionError(
            "Stepwise selection found no significant variables "
            f"at p_entry={p_entry}. "
            "Try relaxing p_entry (e.g. 0.10) or check your feature set."
        )

    logger.info(
        f"Stepwise complete in {step} step(s). "
        f"Selected {len(included)} variable(s): {included}"
    )
    return included, _log_df(log)


def _forward(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
) -> Tuple[List[str], pd.DataFrame]:
    """Forward selection — mirrors SAS SELECTION=FORWARD."""
    p_entry  = config.feature_selection.p_entry
    max_iter = config.feature_selection.max_iterations

    included: List[str] = []
    log: list[dict] = []
    step = 0

    logger.info(f"Forward selection — p_entry={p_entry}")

    for _ in range(max_iter):
        excluded = [c for c in X.columns if c not in included]
        best_p, best_var = 1.0, None

        for var in excluded:
            p = _pvalue_of(X[included + [var]], y, var)
            if p < best_p:
                best_p, best_var = p, var

        if best_var and best_p < p_entry:
            step += 1
            included.append(best_var)
            stats = _model_stats(X[included], y)
            log.append(_log_row(step, best_var, "ENTER", best_p, stats))
            logger.info(f"Step {step}: ENTER '{best_var}' p={format_p_value(best_p)}")
        else:
            break

    if not included:
        raise SelectionError(
            f"Forward selection found no significant variables at p_entry={p_entry}."
        )

    logger.info(f"Forward complete. Selected: {included}")
    return included, _log_df(log)


def _backward(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
) -> Tuple[List[str], pd.DataFrame]:
    """Backward elimination — mirrors SAS SELECTION=BACKWARD."""
    p_removal = config.feature_selection.p_removal
    max_iter  = config.feature_selection.max_iterations

    included = list(X.columns)
    log: list[dict] = []
    step = 0

    logger.info(f"Backward elimination — p_removal={p_removal}")

    for _ in range(max_iter):
        worst_p, worst_var = 0.0, None
        for var in included:
            p = _pvalue_of(X[included], y, var)
            if p > worst_p:
                worst_p, worst_var = p, var

        if worst_var and worst_p > p_removal:
            step += 1
            included.remove(worst_var)
            stats = (
                _model_stats(X[included], y)
                if included
                else {"R2": 0.0, "Adj_R2": 0.0, "AIC": 0.0, "BIC": 0.0}
            )
            log.append(_log_row(step, worst_var, "REMOVE", worst_p, stats))
            logger.info(f"Step {step}: REMOVE '{worst_var}' p={format_p_value(worst_p)}")
        else:
            break

    if not included:
        raise SelectionError(
            f"Backward elimination removed all variables at p_removal={p_removal}. "
            "Try increasing p_removal (e.g. 0.20)."
        )

    logger.info(f"Backward complete. Selected: {included}")
    return included, _log_df(log)


def _manual(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
) -> Tuple[List[str], pd.DataFrame]:
    """Manual selection — analyst specifies variable list explicitly."""
    vars_ = config.feature_selection.manual_variables or []
    missing = [v for v in vars_ if v not in X.columns]
    if missing:
        raise SelectionError(
            f"Manual variables not found in data: {missing}. "
            f"Available columns: {list(X.columns)}"
        )

    stats = _model_stats(X[vars_], y)
    log = [_log_row(1, ", ".join(vars_), "MANUAL", None, stats)]
    logger.info(f"Manual selection — using: {vars_}")
    return vars_, _log_df(log)


# ── Stat helpers ──────────────────────────────────────────────────────────────

def _pvalue_of(X_sub: pd.DataFrame, y: pd.Series, var: str) -> float:
    """OLS p-value for a specific variable — used in selection decisions."""
    try:
        Xc = sm.add_constant(X_sub, has_constant="add")
        res = sm.OLS(y, Xc).fit(disp=0)
        return float(res.pvalues.get(var, 1.0))
    except Exception:
        return 1.0


def _model_stats(X_sub: pd.DataFrame, y: pd.Series) -> dict:
    """Fit OLS and extract R², Adj R², AIC, BIC for iteration logging."""
    try:
        Xc = sm.add_constant(X_sub, has_constant="add")
        res = sm.OLS(y, Xc).fit(disp=0)
        return {
            "R2":     round(float(res.rsquared), 6),
            "Adj_R2": round(float(res.rsquared_adj), 6),
            "AIC":    round(float(res.aic), 4),
            "BIC":    round(float(res.bic), 4),
        }
    except Exception:
        return {"R2": 0.0, "Adj_R2": 0.0, "AIC": 0.0, "BIC": 0.0}


def _log_row(
    step: int,
    variable: str,
    action: str,
    p: float | None,
    stats: dict,
) -> dict:
    return {
        "Step":     step,
        "Variable": variable,
        "Action":   action,
        "R2":       stats["R2"],
        "Adj_R2":   stats["Adj_R2"],
        "AIC":      stats["AIC"],
        "BIC":      stats["BIC"],
        "P_Value":  format_p_value(p) if p is not None else "—",
    }


def _log_df(log: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(log)
    if df.empty:
        return df
    for col in ["R2", "Adj_R2"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    for col in ["AIC", "BIC"]:
        if col in df.columns:
            df[col] = df[col].round(2)
    return df
