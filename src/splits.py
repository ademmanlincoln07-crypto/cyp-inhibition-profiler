"""
Splitting strategies for cross-validation.

- Random k-fold
- Scaffold k-fold (Bemis-Murcko): ensures test folds contain scaffolds not
  seen during training — a harder and more realistic estimate of generalization.
- Group k-fold by scaffold (strict scaffold split).

We also expose a scaffold-to-train/val split helper for holdout evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import constants as C
from .preprocessing import get_scaffold


def compute_scaffolds(mols: List[Chem.Mol]) -> List[str]:
    """Return Bemis-Murcko scaffold SMILES for each mol (or '' for None)."""
    scaffolds = []
    for mol in mols:
        s = get_scaffold(mol)
        scaffolds.append(s if s is not None else "")
    return scaffolds


def scaffold_split(
    scaffolds: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = C.SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return boolean arrays (train, val, test) partitioning by scaffold.
    Scaffolds are sorted by size (largest first) and placed greedily.
    Molecules with no scaffold ("") are placed in the training set.
    """
    rng = np.random.default_rng(seed)
    n = len(scaffolds)
    # Count molecules per scaffold
    counts = Counter(scaffolds)
    # Sort scaffolds by frequency desc, then random tie-break
    scaffold_list = list(counts.keys())
    rng.shuffle(scaffold_list)
    scaffold_list.sort(key=lambda s: (-counts[s], s))

    train_idx, val_idx, test_idx = [], [], []
    n_train_target = int(n * frac_train)
    n_val_target = int(n * frac_valid)

    # Assign scaffolds greedily; train first, then val, then test
    for sc in scaffold_list:
        idxs = [i for i, s in enumerate(scaffolds) if s == sc]
        if len(train_idx) + len(idxs) <= n_train_target:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val_target:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    def to_mask(indices):
        m = np.zeros(n, dtype=bool)
        m[indices] = True
        return m

    return to_mask(train_idx), to_mask(val_idx), to_mask(test_idx)


def scaffold_kfold(
    scaffolds: List[str],
    n_splits: int = 5,
    seed: int = C.SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_mask, test_mask) for n_splits folds of scaffold CV.
    Scaffolds are partitioned approximately evenly across folds.
    """
    rng = np.random.default_rng(seed)
    n = len(scaffolds)
    counts = Counter(scaffolds)
    scaffold_list = list(counts.keys())
    rng.shuffle(scaffold_list)
    scaffold_list.sort(key=lambda s: (-counts[s], s))

    # Greedy balanced assignment of scaffolds to folds
    folds = [[] for _ in range(n_splits)]
    fold_counts = np.zeros(n_splits, dtype=int)
    for sc in scaffold_list:
        # assign to fold with smallest current count
        f = int(np.argmin(fold_counts))
        idxs = [i for i, s in enumerate(scaffolds) if s == sc]
        folds[f].extend(idxs)
        fold_counts[f] += len(idxs)

    splits = []
    for k in range(n_splits):
        test_idx = folds[k]
        train_idx = [i for j in range(n_splits) if j != k for i in folds[j]]
        tr = np.zeros(n, dtype=bool); tr[train_idx] = True
        te = np.zeros(n, dtype=bool); te[test_idx] = True
        splits.append((tr, te))
    return splits


def random_kfold(n: int, n_splits: int = 5, seed: int = C.SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    splits = []
    for k in range(n_splits):
        te_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        tr = np.zeros(n, dtype=bool); tr[tr_idx] = True
        te = np.zeros(n, dtype=bool); te[te_idx] = True
        splits.append((tr, te))
    return splits


if __name__ == "__main__":
    from data_loader import load_train_inhibition
    from preprocessing import process_smiles
    df = load_train_inhibition()
    res = process_smiles(df, name="split_test")
    scs = compute_scaffolds(res.df["mol"].tolist())
    splits = scaffold_kfold(scs, n_splits=5)
    for i, (tr, te) in enumerate(splits):
        print(f"Fold {i}: train={tr.sum()}, test={te.sum()}")
