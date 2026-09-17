"""
Ensemble methods: simple/weighted averaging of OOF predictions.

We search for the optimal linear combination weights on OOF predictions
(per CYP) using a non-negative least-squares / constrained optimisation on
ST-RAE.  This happens strictly on held-out (OOF) data, not on the test set.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import constants as C
from .metrics import soft_threshold_abs_error


def _st_rae_from_weights(
    weights: np.ndarray, preds_list: List[np.ndarray],
    y_true: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray,
    y_train_mean: float,
) -> float:
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()  # normalise to sum to 1
    blend = sum(w[i] * preds_list[i] for i in range(len(preds_list)))
    r, _ = soft_threshold_abs_error(blend, y_true, ci_low, ci_high, y_train=np.array([y_train_mean]))
    return float(r)


def optimal_weights(
    preds_list: List[np.ndarray],
    y_true: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    y_train_mean: float,
    n_models: int,
) -> np.ndarray:
    """
    Find non-negative weights summing to 1 that minimize ST-RAE.
    If optimisation fails, returns uniform weights.
    """
    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    try:
        res = minimize(
            _st_rae_from_weights, x0,
            args=(preds_list, y_true, ci_low, ci_high, y_train_mean),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )
        if res.success:
            w = np.array(res.x, dtype=np.float64)
            return w / w.sum()
    except Exception:
        pass
    return x0


def blend_predictions(
    oof_preds: Dict[str, np.ndarray],        # model_name -> oof (n_labelled,)
    y_true: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    y_train_mean: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Given a dict of model-name -> OOF predictions for a single CYP, find
    optimal non-negative weights and return (blend_pred, weight_dict).
    """
    names = list(oof_preds.keys())
    preds_list = [oof_preds[n] for n in names]
    w = optimal_weights(preds_list, y_true, ci_low, ci_high, y_train_mean, len(names))
    blend = sum(w[i] * preds_list[i] for i in range(len(names)))
    weights = {n: float(w[i]) for i, n in enumerate(names)}
    return blend, weights
