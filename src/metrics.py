"""
Evaluation metrics.

Implements the official Soft-Threshold Relative Absolute Error (ST-RAE) used by the
OpenADMET CYP Challenge, plus standard regression metrics (RMSE, MAE, R²,
Pearson, Spearman).

ST-RAE definition (from the challenge blog post and public reference implementations):
    For compound i, given prediction p_i, truth y_i with 95 % credible interval
    [low_i, high_i]:
        error_i = max(0, low_i - p_i, p_i - high_i)
    I.e. the distance from p_i to the nearest CI bound, or 0 if inside the CI.
    The denominator is the same error computed for a constant-mean predictor
    (mean of training target).
    ST-RAE = mean(error_i) / mean(error_i_mean_predictor)

    Compounds with y_i < 4 are downweighted (below assay resolution).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def soft_threshold_abs_error(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    ci_low: Optional[np.ndarray] = None,
    ci_high: Optional[np.ndarray] = None,
    downweight_floor: float = 4.0,
    y_train: Optional[np.ndarray] = None,
    y_floor: float = -np.inf,
    y_cap: float = np.inf,
) -> Tuple[float, np.ndarray]:
    """
    Compute per-compound soft-threshold absolute error and the aggregate ST-AE
    (not normalised by the constant-mean denominator yet).

    Returns (normalised_ST_RAE, per_compound_error_weights) where per_compound_error
    is the weighted error per compound and normalised_ST_RAE is divided by the
    constant-mean baseline so that 1.0 == no better than predicting the mean.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if ci_low is None:
        ci_low = y_true
    if ci_high is None:
        ci_high = y_true
    ci_low = np.asarray(ci_low, dtype=np.float64)
    ci_high = np.asarray(ci_high, dtype=np.float64)

    # Clip predictions to plausible range (safety)
    y_pred_clipped = np.clip(y_pred, y_floor, y_cap)

    # Per-compound error: distance to nearest CI bound (0 if inside)
    err = np.maximum.reduce([
        np.zeros_like(y_pred_clipped),
        ci_low - y_pred_clipped,
        y_pred_clipped - ci_high,
    ])

    # Weights: downweight low-activity compounds (pIC50 < 4) by factor w<1.
    # The challenge states these are downweighted; we use w=0.25 for floor
    # compounds as a conservative approximation consistent with public refs.
    w = np.ones_like(y_true)
    floor_mask = y_true < downweight_floor
    w[floor_mask] = 0.25
    # Zero-weight compounds where truth is NaN (shouldn't happen but safe)
    w[~np.isfinite(y_true)] = 0.0

    weighted_err = err * w

    # Constant-mean baseline: mean(y_train), or mean(y_true) if no train
    # reference provided (in CV, provide the training-fold mean; in OOF eval
    # we approximate with the fold train mean).
    if y_train is None:
        y_bar = np.nanmean(y_true)
    else:
        y_bar = np.nanmean(y_train)
    err_mean = np.maximum.reduce([
        np.zeros_like(y_true),
        ci_low - y_bar,
        y_bar - ci_high,
    ])
    weighted_err_mean = err_mean * w

    denom = np.mean(weighted_err_mean)
    numer = np.mean(weighted_err)

    st_rae = numer / denom if denom > 0 else np.nan
    return st_rae, weighted_err


def regression_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    ci_low: Optional[np.ndarray] = None,
    ci_high: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    downweight_floor: float = 4.0,
) -> Dict[str, float]:
    """Return a dict of standard + ST-RAE metrics."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yp = y_pred[mask]
    yt = y_true[mask]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    r2   = float(r2_score(yt, yp))
    pearson = float(pearsonr(yt, yp)[0]) if len(yt) > 2 else float("nan")
    spearman = float(spearmanr(yt, yp)[0]) if len(yt) > 2 else float("nan")

    ci_low_m = ci_low[mask] if ci_low is not None else yt
    ci_high_m = ci_high[mask] if ci_high is not None else yt
    st_rae, _ = soft_threshold_abs_error(
        yp, yt, ci_low=ci_low_m, ci_high=ci_high_m,
        downweight_floor=downweight_floor,
        y_train=y_train,
    )

    return {
        "n":        int(mask.sum()),
        "RMSE":     rmse,
        "MAE":      mae,
        "R2":       r2,
        "Pearson":  pearson,
        "Spearman": spearman,
        "ST_RAE":   float(st_rae),
    }


def brier_score_binary(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Brier score for probabilistic binary classification (TDI track)."""
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def mcc(y_pred_binary: np.ndarray, y_true: np.ndarray) -> float:
    """Matthews Correlation Coefficient (binary)."""
    yp = np.asarray(y_pred_binary, dtype=np.int64)
    yt = np.asarray(y_true, dtype=np.int64)
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    num = tp * tn - fp * fn
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return float(num / den) if den > 0 else 0.0


if __name__ == "__main__":
    # sanity check
    rng = np.random.default_rng(0)
    n = 200
    yt = rng.normal(5, 1, n)
    ci_w = 0.2 + 0.3 * rng.random(n)
    ci_low = yt - ci_w
    ci_high = yt + ci_w
    yp = yt + rng.normal(0, 0.5, n)
    print(regression_metrics(yp, yt, ci_low, ci_high))
