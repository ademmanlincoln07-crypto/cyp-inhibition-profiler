"""
CYP-specific baseline & tuned models.

Leak-free nested CV pipeline:
  Outer: scaffold k-fold (or random k-fold) for evaluation.
  Inner: 3-fold random CV + Optuna TPE for hyperparameter tuning.

Models evaluated:
  - Ridge regression on descriptors
  - Random Forest on fingerprints, descriptors, combined
  - XGBoost / LightGBM / HistGradientBoosting
Model selection uses ST-RAE (the challenge primary metric).

All models are CYP-specific (one per isoform, trained only on rows with
non-missing labels for that CYP).
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

from . import constants as C
from .features import clean_descriptors
from .metrics import regression_metrics, soft_threshold_abs_error
from .splits import compute_scaffolds, random_kfold, scaffold_kfold

warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------
# Leak-free feature assembly
# ------------------------------------------------------------------
class FeatureBuilder:
    """
    Fit on a training slice, then transform any slice.
    Handles: descriptor scaling constant-filter, high-correlation removal
    (using train stats only), concatenation with fingerprints.
    """

    def __init__(self, feature_set: str = "both", descr_corr_threshold: float = 0.95):
        self.feature_set = feature_set
        self.descr_corr_threshold = descr_corr_threshold
        # Fitted state:
        self.descr_imputer_: Optional[SimpleImputer] = None
        self.descr_scaler_: Optional[StandardScaler] = None
        self.descr_keep_: Optional[np.ndarray] = None
        self.n_fp_: int = 0
        self.n_descr_in_: int = 0

    def fit(self, fp_train: np.ndarray, descr_train: np.ndarray) -> "FeatureBuilder":
        self.n_fp_ = fp_train.shape[1]
        self.n_descr_in_ = descr_train.shape[1]

        if self.feature_set in ("descr", "both"):
            # Step 1: remove NaN/Inf + impute with train-set median
            d = descr_train.astype(np.float64).copy()
            d[~np.isfinite(d)] = np.nan
            self.descr_imputer_ = SimpleImputer(strategy="median")
            d_imp = self.descr_imputer_.fit_transform(d)
            # Step 2: remove constant columns on train
            std = d_imp.std(axis=0)
            keep = std > 1e-8
            # Step 3: iterative correlation removal (greedy by train corr)
            if self.descr_corr_threshold and 0 < self.descr_corr_threshold < 1:
                d_k = d_imp[:, keep]
                corr = np.corrcoef(d_k, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0)
                m = corr.shape[0]
                to_drop = set()
                for i in range(m):
                    if i in to_drop:
                        continue
                    for j in range(i+1, m):
                        if j in to_drop:
                            continue
                        if abs(corr[i, j]) > self.descr_corr_threshold:
                            to_drop.add(j)
                keep_idx = np.where(keep)[0]
                for j in to_drop:
                    keep[keep_idx[j]] = False
            self.descr_keep_ = keep
            # Step 4: standard scaler on kept descriptors
            self.descr_scaler_ = StandardScaler()
            self.descr_scaler_.fit(d_imp[:, keep])
        return self

    def transform(self, fp: np.ndarray, descr: np.ndarray) -> np.ndarray:
        parts = []
        if self.feature_set in ("fp", "both"):
            parts.append(fp.astype(np.float32))
        if self.feature_set in ("descr", "both"):
            d = descr.astype(np.float64).copy()
            d[~np.isfinite(d)] = np.nan
            d_imp = self.descr_imputer_.transform(d)
            d_k = d_imp[:, self.descr_keep_]
            d_sc = self.descr_scaler_.transform(d_k)
            parts.append(d_sc.astype(np.float32))
        return np.hstack(parts) if parts else np.empty((fp.shape[0], 0), dtype=np.float32)

    def n_features_out(self, fp: np.ndarray, descr: np.ndarray) -> int:
        return self.transform(fp[:1], descr[:1]).shape[1]


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------
def get_model(name: str, params: Optional[Dict] = None, seed: int = C.SEED):
    p = dict(params or {})
    if name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(**{k: v for k, v in p.items() if k != "n_estimators"}, random_state=seed)),
        ])
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 500),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            max_features=p.get("max_features", 0.33),
            n_jobs=-1, random_state=seed,
        )
    if name == "et":
        return ExtraTreesRegressor(
            n_estimators=p.get("n_estimators", 500),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            max_features=p.get("max_features", 0.33),
            n_jobs=-1, random_state=seed,
        )
    if name == "gbm":
        return GradientBoostingRegressor(
            n_estimators=p.get("n_estimators", 300),
            learning_rate=p.get("lr", 0.05),
            max_depth=p.get("max_depth", 3),
            subsample=p.get("subsample", 0.8),
            random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=p.get("n_estimators", 400),
            learning_rate=p.get("lr", 0.05),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            random_state=seed,
        )
    if name == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 1000),
            learning_rate=p.get("lr", 0.05),
            num_leaves=p.get("num_leaves", 63),
            min_child_samples=p.get("min_data_in_leaf", 20),
            colsample_bytree=p.get("feature_fraction", 0.8),
            subsample=p.get("bagging_fraction", 0.8),
            subsample_freq=5,
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 0.0),
            random_state=seed,
            verbose=-1, n_jobs=-1,
        )
    if name == "xgb":
        return xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 1000),
            learning_rate=p.get("lr", 0.05),
            max_depth=p.get("max_depth", 6),
            min_child_weight=p.get("min_child_weight", 3),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 0.0),
            random_state=seed,
            verbosity=0, n_jobs=-1,
            objective="reg:squarederror", tree_method="hist",
        )
    raise ValueError(f"Unknown model: {name}")


# ------------------------------------------------------------------
# Inner-CV hyperparameter search (Optuna)
# ------------------------------------------------------------------
def _param_space(trial, model_name: str) -> Dict:
    if model_name in ("rf", "et"):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            max_depth=trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_float("max_features", 0.1, 1.0),
        )
    if model_name == "ridge":
        return dict(alpha=trial.suggest_float("alpha", 1e-2, 1e3, log=True))
    if model_name == "gbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600),
            lr=trial.suggest_float("lr", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
        )
    if model_name == "hgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            lr=trial.suggest_float("lr", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_categorical("max_depth", [None, 4, 8, 12, 16]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 60),
        )
    if model_name == "lgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1500),
            lr=trial.suggest_float("lr", 1e-3, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 60),
            feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if model_name == "xgb":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1500),
            lr=trial.suggest_float("lr", 1e-3, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    return {}


def optuna_tune(
    X_train: np.ndarray, y_train: np.ndarray,
    ci_low: np.ndarray, ci_high: np.ndarray,
    model_name: str, n_trials: int = 20, cv_folds: int = 3, seed: int = C.SEED,
) -> Dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_ids = np.zeros(n, dtype=int)
    for i, idx in enumerate(np.array_split(perm, cv_folds)):
        fold_ids[idx] = i

    def objective(trial):
        params = _param_space(trial, model_name)
        raes = []
        for k in range(cv_folds):
            tr = fold_ids != k
            va = fold_ids == k
            if tr.sum() < 20 or va.sum() < 10:
                continue
            m = get_model(model_name, params=params, seed=seed)
            m.fit(X_train[tr], y_train[tr])
            p = m.predict(X_train[va])
            r, _ = soft_threshold_abs_error(
                p, y_train[va], ci_low=ci_low[va], ci_high=ci_high[va],
                y_train=y_train[tr],
            )
            raes.append(r)
        return float(np.mean(raes)) if raes else float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ------------------------------------------------------------------
# Cross-validation driver
# ------------------------------------------------------------------
@dataclass
class CYPResult:
    cyp: str
    model_name: str
    feature_set: str
    oof_preds: np.ndarray
    oof_metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    best_params: Dict[str, Any] = field(default_factory=dict)
    final_model: Any = None
    final_feature_builder: Optional[FeatureBuilder] = None
    training_time_sec: float = 0.0
    labelled_index: Optional[np.ndarray] = None
    n_labelled: int = 0


def cv_cyp(
    cyp: str, df: pd.DataFrame,
    fp: np.ndarray, descr: np.ndarray, descr_names: List[str],
    model_name: str = "lgbm", feature_set: str = "both",
    n_outer: int = 5, split_type: str = "scaffold",
    n_trials_inner: int = 20, seed: int = C.SEED,
) -> CYPResult:
    """Nested CV for a single CYP; refits final model on all labelled data."""
    target = C.TARGET_COLS[cyp]
    ci_low_col = C.CI_LOW_COLS[cyp]
    ci_high_col = C.CI_HIGH_COLS[cyp]

    labelled = df[target].notna().values
    df_lab = df.loc[labelled].reset_index(drop=True)
    fp_lab = fp[labelled]
    descr_lab = descr[labelled]
    y = df_lab[target].values.astype(np.float64)
    ci_low = df_lab[ci_low_col].values.astype(np.float64)
    ci_high = df_lab[ci_high_col].values.astype(np.float64)
    mols = df_lab["mol"].tolist()
    n = len(y)

    print(f"\n[cv_cyp={cyp} model={model_name} feat={feature_set}] n={n} "
          f"split={split_type} outer={n_outer} inner_trials={n_trials_inner}")

    if split_type == "scaffold":
        scs = compute_scaffolds(mols)
        splits = scaffold_kfold(scs, n_splits=n_outer, seed=seed)
    else:
        splits = random_kfold(n, n_splits=n_outer, seed=seed)

    oof = np.full(n, np.nan, dtype=np.float64)
    fold_metrics = []
    best_params_per_fold = []

    t0 = time.time()
    for k, (tr_mask, te_mask) in enumerate(splits):
        fb = FeatureBuilder(feature_set=feature_set).fit(fp_lab[tr_mask], descr_lab[tr_mask])
        X_tr = fb.transform(fp_lab[tr_mask], descr_lab[tr_mask])
        X_te = fb.transform(fp_lab[te_mask], descr_lab[te_mask])
        y_tr, y_te = y[tr_mask], y[te_mask]
        cl_tr, ch_tr = ci_low[tr_mask], ci_high[tr_mask]
        cl_te, ch_te = ci_low[te_mask], ci_high[te_mask]

        if n_trials_inner > 0:
            best_p = optuna_tune(
                X_tr, y_tr, cl_tr, ch_tr,
                model_name=model_name, n_trials=n_trials_inner,
                cv_folds=3, seed=seed + 100 * k,
            )
        else:
            best_p = {}
        best_params_per_fold.append(best_p)

        model = get_model(model_name, params=best_p, seed=seed + k)
        model.fit(X_tr, y_tr)
        p_te = model.predict(X_te)
        oof[te_mask] = p_te

        m = regression_metrics(p_te, y_te, cl_te, ch_te, y_train=y_tr)
        fold_metrics.append(m)
        print(f"  Fold {k}: n_tr={tr_mask.sum()}, n_te={te_mask.sum()}, "
              f"ST-RAE={m['ST_RAE']:.3f}  RMSE={m['RMSE']:.3f}  "
              f"R2={m['R2']:.3f}  ρ={m['Spearman']:.3f}")

    oof_m = regression_metrics(oof, y, ci_low, ci_high, y_train=y)
    print(f"  -> OOF : ST-RAE={oof_m['ST_RAE']:.3f}  RMSE={oof_m['RMSE']:.3f}  "
          f"R2={oof_m['R2']:.3f}  ρ={oof_m['Spearman']:.3f}")

    # Refit on all data with aggregated params
    final_params = _aggregate_params(best_params_per_fold)
    fb_final = FeatureBuilder(feature_set=feature_set).fit(fp_lab, descr_lab)
    X_all = fb_final.transform(fp_lab, descr_lab)
    final_model = get_model(model_name, params=final_params, seed=seed)
    final_model.fit(X_all, y)
    elapsed = time.time() - t0

    return CYPResult(
        cyp=cyp, model_name=model_name, feature_set=feature_set,
        oof_preds=oof, oof_metrics=oof_m, fold_metrics=fold_metrics,
        best_params=final_params, final_model=final_model,
        final_feature_builder=fb_final,
        training_time_sec=elapsed,
        labelled_index=np.where(labelled)[0],
        n_labelled=n,
    )


def _aggregate_params(params_list: List[Dict]) -> Dict:
    if not params_list:
        return {}
    keys = set().union(*params_list)
    out = {}
    for k in keys:
        vals = [p[k] for p in params_list if k in p and p[k] is not None]
        if not vals:
            continue
        v0 = vals[0]
        if isinstance(v0, bool) or v0 is None:
            out[k] = v0
        elif isinstance(v0, (int, np.integer)) and not isinstance(v0, bool):
            out[k] = int(np.median(vals))
        elif isinstance(v0, (float, np.floating)):
            out[k] = float(np.median(vals))
        else:
            out[k] = v0
    return out


def baseline_screen(
    df: pd.DataFrame,
    fp: np.ndarray, descr: np.ndarray, descr_names: List[str],
    models: Optional[List[str]] = None,
    feature_sets: Optional[List[str]] = None,
    n_outer: int = 5, split_type: str = "scaffold",
    n_trials_inner: int = 15, cyps: Optional[List[str]] = None,
) -> pd.DataFrame:
    models = models or ["lgbm", "xgb", "rf", "hgb", "ridge"]
    feature_sets = feature_sets or ["fp", "descr", "both"]
    cyps = cyps or C.CYP_ISOFORMS
    rows = []
    for cyp in cyps:
        for fs in feature_sets:
            for mn in models:
                if mn == "ridge" and fs == "fp":
                    continue  # ridge on 2048-bit FP is not useful
                try:
                    res = cv_cyp(
                        cyp, df, fp, descr, descr_names,
                        model_name=mn, feature_set=fs,
                        n_outer=n_outer, split_type=split_type,
                        n_trials_inner=n_trials_inner,
                    )
                    rows.append({
                        "CYP": cyp, "model": mn, "features": fs,
                        **{k: v for k, v in res.oof_metrics.items() if k != "n"},
                        "n": res.oof_metrics["n"],
                        "time_sec": res.training_time_sec,
                    })
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  !! {cyp}/{mn}/{fs} failed: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .data_loader import load_train_inhibition
    from .preprocessing import process_smiles
    from .features import build_features_for_df
    df = load_train_inhibition().head(500)
    res = process_smiles(df, name="smoke")
    feats = build_features_for_df(res.df, cache_tag="smoke", use_cache=False)
    r = cv_cyp(
        "CYP3A4", res.df, feats["fp"], feats["descr_raw"], feats["descr_names_raw"],
        model_name="lgbm", feature_set="fp", n_outer=3, n_trials_inner=5,
    )
    print("OOF metrics:", r.oof_metrics)
