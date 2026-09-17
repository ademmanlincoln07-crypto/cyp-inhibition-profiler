"""
Molecular feature generation: fingerprints + RDKit 2D descriptors.

Design choices:
- Morgan (ECFP-style) circular fingerprint, radius 2 or 3, 1024/2048 bits.
- RDKit 2D descriptor block (~200 descriptors), filtered of NaN/constant/highly-correlated.
- All generation is deterministic given a SMILES column.
- Cached to data/processed so we do not recompute across runs.
"""
from __future__ import annotations

import os
import pickle
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdFingerprintGenerator,
    Lipinski,
    Crippen,
    rdMolDescriptors,
)

from . import constants as C


# ------------------------------------------------------------------
# Descriptor list
# ------------------------------------------------------------------
# We use a curated set of ~200 RDKit descriptors; any that are constant or
# produce NaNs across a dataset are removed at feature-matrix construction time.

def _descriptor_fns():
    """Return list of (name, fn) pairs for RDKit 2D descriptors."""
    fns = []
    for name, fn in Descriptors._descList:
        fns.append((name, fn))
    return fns

DESCRIPTOR_FNS = _descriptor_fns()


# ------------------------------------------------------------------
# Fingerprint generation
# ------------------------------------------------------------------
def make_morgan_generator(radius: int = 2, nbits: int = 2048, use_chirality: bool = False):
    """Build an RDKit Morgan fingerprint generator (API stable across RDKit versions)."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nbits,
        includeChirality=use_chirality,
    )


def morgan_fingerprints(
    mols: List[Chem.Mol],
    radius: int = 2,
    nbits: int = 2048,
    use_chirality: bool = False,
    as_numpy: bool = True,
) -> np.ndarray:
    """
    Compute Morgan (ECFP) fingerprints for a list of RDKit molecules.
    Returns a dense np.ndarray of shape (n_mols, nbits) by default.
    """
    gen = make_morgan_generator(radius=radius, nbits=nbits, use_chirality=use_chirality)
    fps = []
    for mol in mols:
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.int8))
            continue
        fp = gen.GetFingerprint(mol)
        if as_numpy:
            arr = np.zeros((nbits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(fp)
    if as_numpy:
        return np.vstack(fps)
    return fps


def rdkit_descriptors(mols: List[Chem.Mol]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all RDKit 2D descriptors.
    Returns (matrix, list_of_names). NaNs are replaced with 0 (and flagged).
    """
    n = len(mols)
    m = len(DESCRIPTOR_FNS)
    X = np.zeros((n, m), dtype=np.float64)
    names = [name for name, _ in DESCRIPTOR_FNS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, (_, fn) in enumerate(DESCRIPTOR_FNS):
            for i, mol in enumerate(mols):
                if mol is None:
                    X[i, j] = np.nan
                    continue
                try:
                    val = fn(mol)
                    X[i, j] = val
                except Exception:
                    X[i, j] = np.nan
    return X, names


def clean_descriptors(
    X: np.ndarray,
    names: List[str],
    remove_constant: bool = True,
    corr_threshold: Optional[float] = 0.95,
    train_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Clean a descriptor matrix:
      - Replace +/- inf with NaN
      - Fill NaNs with column median (computed on train_mask only!)
      - Drop constant columns (based on train_mask)
      - Drop one of each pair with |Pearson r| > corr_threshold (on train_mask)
    Returns (X_clean, kept_names).
    """
    X = X.astype(np.float64).copy()
    X[~np.isfinite(X)] = np.nan

    if train_mask is None:
        train_mask = np.ones(X.shape[0], dtype=bool)

    # Impute NaNs with median from training
    medians = np.nanmedian(X[train_mask], axis=0)
    # Where all-train was NaN -> 0 (rare)
    medians = np.where(np.isnan(medians), 0.0, medians)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(medians, inds[1])

    keep = np.ones(X.shape[1], dtype=bool)
    if remove_constant:
        col_std = np.nanstd(X[train_mask], axis=0)
        keep &= (col_std > 1e-8)
    if corr_threshold is not None and 0 < corr_threshold < 1:
        # Iteratively remove highly correlated features
        cols_keep_idx = np.where(keep)[0]
        Xt = X[np.ix_(train_mask, cols_keep_idx)]
        corr = np.corrcoef(Xt, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        n = corr.shape[0]
        to_drop = set()
        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i+1, n):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > corr_threshold:
                    to_drop.add(j)
        for j in to_drop:
            keep[cols_keep_idx[j]] = False

    X_kept = X[:, keep]
    names_kept = [n for n, k in zip(names, keep) if k]
    return X_kept, names_kept


# ------------------------------------------------------------------
# Combined feature builder (with caching)
# ------------------------------------------------------------------
CACHE_DIR = C.DATA_PROC

def build_features_for_df(
    df: pd.DataFrame,
    mol_col: str = "mol",
    fp_radius: int = 2,
    fp_nbits: int = 2048,
    fp_use_chirality: bool = False,
    include_descriptors: bool = True,
    cache_tag: str = "features",
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build fingerprint + descriptor matrices for a DataFrame with an 'mol' column.
    Returns dict:
       'fp': (n, nbits) Morgan fingerprints
       'descr': (n, d_kept) cleaned descriptors  (only if include_descriptors)
       'descr_names': list of kept descriptor names
    """
    cache_fn = os.path.join(CACHE_DIR, f"{cache_tag}_r{fp_radius}_b{fp_nbits}{'_chiral' if fp_use_chirality else ''}.pkl")
    if use_cache and os.path.exists(cache_fn):
        with open(cache_fn, "rb") as f:
            cached = pickle.load(f)
        print(f"[features] Loaded cached features from {cache_fn}")
        return cached

    mols = df[mol_col].tolist()
    fp = morgan_fingerprints(mols, radius=fp_radius, nbits=fp_nbits, use_chirality=fp_use_chirality)

    out: Dict[str, np.ndarray] = {"fp": fp}

    if include_descriptors:
        d_raw, d_names = rdkit_descriptors(mols)
        # Since this is dataset-wide pre-feature-selection, we do NOT dedup by
        # correlation here — that happens inside CV folds to avoid leakage.
        d_raw = d_raw.astype(np.float64)
        d_raw[~np.isfinite(d_raw)] = np.nan
        # Global constant removal (safe) — removes 0/0 descriptors like
        # NumRadicalElectrons which may be constant across drug-like sets
        global_std = np.nanstd(d_raw, axis=0)
        keep = global_std > 1e-10
        d_raw = d_raw[:, keep]
        d_names_kept = [n for n, k in zip(d_names, keep) if k]
        out["descr_raw"] = d_raw
        out["descr_names_raw"] = d_names_kept

    if use_cache:
        with open(cache_fn, "wb") as f:
            pickle.dump(out, f)
        print(f"[features] Cached features to {cache_fn}")

    return out


if __name__ == "__main__":
    from data_loader import load_train_inhibition
    from preprocessing import process_smiles
    df = load_train_inhibition().head(200)
    r = process_smiles(df, name="smoke_test")
    feats = build_features_for_df(r.df, cache_tag="smoke", use_cache=False)
    for k, v in feats.items():
        if hasattr(v, "shape"):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v), len(v) if hasattr(v, '__len__') else v)
