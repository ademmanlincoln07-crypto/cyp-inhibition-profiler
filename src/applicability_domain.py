"""
Applicability-domain analysis:
  - Tanimoto similarity to nearest training neighbour (Morgan FP)
  - Scaffold coverage
  - Descriptor-space Mahalanobis/leverage flag
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from . import constants as C


_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp_of_mol(mol):
    if mol is None:
        return None
    return _GEN.GetFingerprint(mol)


def nearest_neighbor_similarity(
    train_mols: List[Chem.Mol],
    test_mols: List[Chem.Mol],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (similarity_to_nearest_train_nn, index_of_nn_in_train)."""
    train_fps = [_fp_of_mol(m) for m in train_mols]
    sims = np.zeros(len(test_mols), dtype=np.float64)
    idx_of_nn = np.zeros(len(test_mols), dtype=np.int64)
    for i, m in enumerate(test_mols):
        fp = _fp_of_mol(m)
        if fp is None:
            sims[i] = 0.0; idx_of_nn[i] = -1; continue
        # bulk TanimotoSimilarity is O(N) per test mol, acceptable
        best = 0.0; best_j = -1
        for j, tfp in enumerate(train_fps):
            if tfp is None:
                continue
            s = DataStructs.TanimotoSimilarity(fp, tfp)
            if s > best:
                best = s; best_j = j
        sims[i] = best; idx_of_nn[i] = best_j
    return sims, idx_of_nn


def flag_out_of_domain(
    train_mols: List[Chem.Mol], test_mols: List[Chem.Mol],
    threshold_low: float = 0.3, threshold_med: float = 0.5,
) -> np.ndarray:
    """
    Return integer AD flag per test molecule:
      0 = high confidence (max Tanimoto >= threshold_med)
      1 = medium (>= threshold_low)
      2 = low / out-of-domain (< threshold_low)
    """
    sims, _ = nearest_neighbor_similarity(train_mols, test_mols)
    flags = np.ones(len(test_mols), dtype=np.int8)
    flags[sims >= threshold_med] = 0
    flags[sims < threshold_low] = 2
    return flags, sims
