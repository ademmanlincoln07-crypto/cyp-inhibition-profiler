"""
Multi-task neural network for CYP inhibition regression.

Architecture:
  Input (fingerprint + descriptors, ~2.5k features after cleaning)
    -> Shared MLP encoder (2 layers, SiLU, dropout, LayerNorm)
    -> Per-CYP head (1 hidden -> 1 output)
  Loss: masked MSE over 4 CYPs (missing targets masked out per compound).

This is a deliberately simple, robust MLP — no graphs, no attention. Trained
with AdamW + cosine LR + early stopping on scaffold-split validation.
"""
from __future__ import annotations

import math
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import constants as C
from .metrics import regression_metrics
from .splits import compute_scaffolds, scaffold_kfold


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class CYPDataSet(Dataset):
    """
    X: (n, d) features
    Y: (n, 4) pIC50 targets (NaN where label missing)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.mask = torch.isfinite(self.Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.mask[i]


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
class SharedEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, latent: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.SiLU(),
            nn.LayerNorm(latent),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CYPHead(nn.Module):
    def __init__(self, latent: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class MultiTaskCYP(nn.Module):
    def __init__(self, in_dim: int, enc_hidden: int = 512, enc_latent: int = 256,
                 head_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_dim, enc_hidden, enc_latent, dropout)
        self.heads = nn.ModuleList([CYPHead(enc_latent, head_hidden) for _ in range(4)])

    def forward(self, x):
        z = self.encoder(x)
        return torch.stack([h(z) for h in self.heads], dim=-1)  # (B, 4)


# ---------------------------------------------------------------
# Masked MSE loss
# ---------------------------------------------------------------
def masked_mse(pred, target, mask, weight=None):
    """Weighted masked MSE; weight per task (shape (4,))."""
    diff = (pred - target) ** 2
    diff = diff * mask.float()
    if weight is None:
        return diff.sum() / mask.float().sum().clamp(min=1.0)
    w = weight.to(pred.device).view(1, -1)
    diff = diff * w
    return diff.sum() / (mask.float() * w).sum().clamp(min=1.0)


# ---------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, weight=None):
    model.train()
    total = 0.0
    n = 0
    for X, Y, mask in loader:
        X = X.to(device); Y = Y.to(device); mask = mask.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = masked_mse(pred, Y, mask, weight=weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, X_or_loader, device="cpu") -> np.ndarray:
    model.eval()
    if isinstance(X_or_loader, np.ndarray):
        X = torch.tensor(X_or_loader, dtype=torch.float32).to(device)
        return model(X).cpu().numpy()
    preds = []
    for X, *_ in X_or_loader:
        X = X.to(device)
        preds.append(model(X).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------
# Fit a multi-task model with early stopping on a validation fold
# ---------------------------------------------------------------
@dataclass
class MultiTaskResult:
    model: Optional[MultiTaskCYP] = None
    history: Dict[str, List[float]] = field(default_factory=lambda: {"train": [], "val": []})
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    val_predictions: Optional[np.ndarray] = None
    feature_builder: Any = None  # FeatureBuilder fitted on train
    config: Dict[str, Any] = field(default_factory=dict)


def _make_Y_matrix(df_subset: pd.DataFrame) -> np.ndarray:
    Y = np.full((len(df_subset), 4), np.nan, dtype=np.float32)
    for j, cyp in enumerate(C.CYP_ISOFORMS):
        col = C.TARGET_COLS[cyp]
        if col in df_subset.columns:
            Y[:, j] = df_subset[col].values.astype(np.float32)
    return Y


def fit_multitask(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, Y_val: Optional[np.ndarray] = None,
    enc_hidden: int = 512, enc_latent: int = 256, head_hidden: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    batch_size: int = 128, max_epochs: int = 80, patience: int = 15,
    task_weighting: str = "inverse_std",
    device: str = "cpu", seed: int = C.SEED,
) -> MultiTaskResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = X_train.shape[1]
    model = MultiTaskCYP(in_dim, enc_hidden, enc_latent, head_hidden, dropout).to(device)

    # Task weights
    if task_weighting == "inverse_std":
        stds = []
        for j in range(4):
            m = np.isfinite(Y_train[:, j])
            s = Y_train[m, j].std() if m.sum() > 5 else 1.0
            stds.append(max(s, 0.1))
        weight = torch.tensor([1.0 / s for s in stds], dtype=torch.float32)
    elif task_weighting == "equal":
        weight = torch.ones(4, dtype=torch.float32)
    else:
        weight = None

    train_ds = CYPDataSet(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    result = MultiTaskResult(model=model, config=dict(
        enc_hidden=enc_hidden, enc_latent=enc_latent, head_hidden=head_hidden,
        dropout=dropout, lr=lr, weight_decay=weight_decay,
        batch_size=batch_size, max_epochs=max_epochs, patience=patience,
    ))

    best_state = None
    wait = 0
    t0 = time.time()
    for epoch in range(max_epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, weight=weight)
        sched.step()
        result.history["train"].append(tr_loss)

        if X_val is not None and Y_val is not None:
            val_ds = CYPDataSet(X_val, Y_val)
            val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
            model.eval()
            vloss_total = 0.0; n = 0
            with torch.no_grad():
                for Xb, Yb, mb in val_loader:
                    Xb = Xb.to(device); Yb = Yb.to(device); mb = mb.to(device)
                    pred = model(Xb)
                    vloss = masked_mse(pred, Yb, mb, weight=weight)
                    vloss_total += vloss.item() * Xb.size(0); n += Xb.size(0)
            val_loss = vloss_total / max(n, 1)
            result.history["val"].append(val_loss)
            if val_loss < result.best_val_loss - 1e-5:
                result.best_val_loss = val_loss
                result.best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if X_val is not None:
        result.val_predictions = predict(model, X_val, device=device)

    return result


def cv_multitask(
    df: pd.DataFrame, fp: np.ndarray, descr: np.ndarray, descr_names: List[str],
    feature_set: str = "both", n_outer: int = 5, split_type: str = "scaffold",
    seed: int = C.SEED, device: str = "cpu",
    **fit_kwargs,
) -> Dict[str, Any]:
    """
    Run scaffold-CV for the multi-task NN.
    Returns dict with:
      'oof': (n_total_labelled_any, 4) OOF predictions (NaN where label missing)
      'metrics_per_cyp': dict of cyp -> regression metrics
      'results_per_fold': list of MultiTaskResult
    """
    from .models import FeatureBuilder

    # We use ALL rows with at least one label
    any_label = np.zeros(len(df), dtype=bool)
    for cyp in C.CYP_ISOFORMS:
        any_label |= df[C.TARGET_COLS[cyp]].notna().values
    df_sub = df.loc[any_label].reset_index(drop=True)
    fp_sub = fp[any_label]; descr_sub = descr[any_label]
    Y_all = _make_Y_matrix(df_sub)
    mols = df_sub["mol"].tolist()
    n = len(df_sub)
    print(f"\n[cv_multitask] n={n} (any label), feature_set={feature_set}")

    if split_type == "scaffold":
        scs = compute_scaffolds(mols)
        splits = scaffold_kfold(scs, n_splits=n_outer, seed=seed)
    else:
        from .splits import random_kfold
        splits = random_kfold(n, n_splits=n_outer, seed=seed)

    oof = np.full((n, 4), np.nan, dtype=np.float64)
    fold_metrics = {cyp: [] for cyp in C.CYP_ISOFORMS}

    for k, (tr_mask, te_mask) in enumerate(splits):
        fb = FeatureBuilder(feature_set=feature_set).fit(fp_sub[tr_mask], descr_sub[tr_mask])
        X_tr = fb.transform(fp_sub[tr_mask], descr_sub[tr_mask])
        X_te = fb.transform(fp_sub[te_mask], descr_sub[te_mask])
        Y_tr = Y_all[tr_mask]; Y_te = Y_all[te_mask]

        # Inner small val split for early stopping (random 10% of train)
        rng = np.random.default_rng(seed + k)
        perm = rng.permutation(X_tr.shape[0])
        cut = int(0.9 * len(perm))
        tr_idx = perm[:cut]; va_idx = perm[cut:]

        res = fit_multitask(
            X_tr[tr_idx], Y_tr[tr_idx],
            X_val=X_tr[va_idx], Y_val=Y_tr[va_idx],
            device=device, seed=seed + k, **fit_kwargs,
        )
        p_te = predict(res.model, X_te, device=device)
        oof[te_mask] = p_te

        for j, cyp in enumerate(C.CYP_ISOFORMS):
            y_true = Y_te[:, j]
            y_pred = p_te[:, j]
            m = np.isfinite(y_true) & np.isfinite(y_pred)
            if m.sum() >= 10:
                # CI columns available in df_sub for ST-RAE
                cl_col = C.CI_LOW_COLS[cyp]; ch_col = C.CI_HIGH_COLS[cyp]
                yt = y_true[m]; yp = y_pred[m]
                cl = df_sub.iloc[np.where(te_mask)[0][m]][cl_col].values
                ch = df_sub.iloc[np.where(te_mask)[0][m]][ch_col].values
                met = regression_metrics(yp, yt, cl, ch, y_train=Y_tr[tr_idx, j][np.isfinite(Y_tr[tr_idx, j])])
                fold_metrics[cyp].append(met)
        print(f"  Fold {k}: n_tr={tr_mask.sum()}, n_te={te_mask.sum()}, best_epoch={res.best_epoch}")

    # Aggregate per-CYP metrics (concatenate OOF)
    agg_metrics = {}
    for j, cyp in enumerate(C.CYP_ISOFORMS):
        y_true = Y_all[:, j]; y_pred = oof[:, j]
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[m]; yp = y_pred[m]
        cl = df_sub.iloc[np.where(m)[0]][C.CI_LOW_COLS[cyp]].values
        ch = df_sub.iloc[np.where(m)[0]][C.CI_HIGH_COLS[cyp]].values
        y_train_all = Y_all[:, j]; y_train_all = y_train_all[np.isfinite(y_train_all)]
        agg_metrics[cyp] = regression_metrics(yp, yt, cl, ch, y_train=y_train_all)
        print(f"  -> {cyp} OOF: ST-RAE={agg_metrics[cyp]['ST_RAE']:.3f} "
              f"RMSE={agg_metrics[cyp]['RMSE']:.3f} R2={agg_metrics[cyp]['R2']:.3f} "
              f"Spearman={agg_metrics[cyp]['Spearman']:.3f}")

    return {
        "oof": oof, "Y": Y_all, "metrics_per_cyp": agg_metrics,
        "df_labelled": df_sub, "any_label_mask": any_label,
    }


def fit_final_multitask(
    df: pd.DataFrame, fp: np.ndarray, descr: np.ndarray, descr_names: List[str],
    feature_set: str = "both", device: str = "cpu", seed: int = C.SEED, **fit_kwargs
):
    """Fit final multi-task model on all available labelled data for test prediction."""
    from .models import FeatureBuilder
    any_label = np.zeros(len(df), dtype=bool)
    for cyp in C.CYP_ISOFORMS:
        any_label |= df[C.TARGET_COLS[cyp]].notna().values
    df_sub = df.loc[any_label].reset_index(drop=True)
    fp_sub = fp[any_label]; descr_sub = descr[any_label]
    Y = _make_Y_matrix(df_sub)

    fb = FeatureBuilder(feature_set=feature_set).fit(fp_sub, descr_sub)
    X = fb.transform(fp_sub, descr_sub)
    # small val split for early-stopping on final fit
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    cut = int(0.92 * len(perm))
    tr_idx = perm[:cut]; va_idx = perm[cut:]
    res = fit_multitask(
        X[tr_idx], Y[tr_idx], X_val=X[va_idx], Y_val=Y[va_idx],
        device=device, seed=seed, **fit_kwargs,
    )
    return res, fb, df_sub


if __name__ == "__main__":
    print("multitask_model loaded")
