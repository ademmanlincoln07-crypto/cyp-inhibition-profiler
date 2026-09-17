"""
RDKit molecular preprocessing and validation.

Pipeline:
  SMILES
   -> parse
   -> validation flag
   -> (optional) neutralisation / fragment selection (largest fragment)
   -> canonical SMILES
   -> audit log of any failed or modified structures

We intentionally keep all original rows — invalid molecules are flagged, not dropped,
so downstream code can handle them explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass
class PreprocessResult:
    df: pd.DataFrame                       # original data + new columns
    invalid_idx: List[int] = field(default_factory=list)
    modified_idx: List[int] = field(default_factory=list)
    canonical_smiles_map: Dict[str, str] = field(default_factory=dict)  # orig -> canon


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest covalent fragment (drops counterions, solvents)."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return mol
    # choose fragment with heaviest atom count
    frags_sorted = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return frags_sorted[0]


def _uncharge(mol: Chem.Mol) -> Chem.Mol:
    """Neutralise molecule where reasonable using RDKit standardizer."""
    try:
        uncharger = rdMolStandardize.Uncharger()
        return uncharger.uncharge(mol)
    except Exception:
        return mol


def process_smiles(
    df: pd.DataFrame,
    name: str = "dataset",
    smiles_col: str = "SMILES",
    keep_largest_fragment: bool = True,
    uncharge: bool = False,
) -> PreprocessResult:
    """
    Parse and standardise a DataFrame containing SMILES.

    Adds columns:
      - 'mol':           RDKit mol object (None for invalid)
      - 'canon_smiles':  canonical SMILES (or NaN)
      - 'valid_mol':     bool
      - 'was_modified':  bool (True if we changed something vs raw SMILES)
    """
    result = PreprocessResult(df=df.copy())

    mols = []
    canon = []
    valid = []
    modified = []

    for i, smi in enumerate(df[smiles_col]):
        if pd.isna(smi) or str(smi).strip() == "":
            mols.append(None)
            canon.append(np.nan)
            valid.append(False)
            modified.append(False)
            result.invalid_idx.append(i)
            continue

        raw = str(smi)
        mol = Chem.MolFromSmiles(raw)
        was_mod = False

        if mol is None:
            mols.append(None)
            canon.append(np.nan)
            valid.append(False)
            modified.append(False)
            result.invalid_idx.append(i)
            continue

        try:
            # Largest fragment (drops salts/counterions) — common practice, always safe
            if keep_largest_fragment:
                mol_frag = _largest_fragment(mol)
                if mol_frag.GetNumHeavyAtoms() != mol.GetNumHeavyAtoms():
                    mol = mol_frag
                    was_mod = True
            # Optionally uncharge
            if uncharge:
                mol_unch = _uncharge(mol)
                can_before = Chem.MolToSmiles(mol)
                can_after = Chem.MolToSmiles(mol_unch)
                if can_before != can_after:
                    mol = mol_unch
                    was_mod = True
            # Ensure aromaticity / valence flags are recomputed
            Chem.SanitizeMol(mol)
        except Exception as e:
            # If standardisation fails, keep original parse
            mol = Chem.MolFromSmiles(raw)
            was_mod = False

        can_smi = Chem.MolToSmiles(mol)
        mols.append(mol)
        canon.append(can_smi)
        valid.append(True)
        modified.append(was_mod)
        result.canonical_smiles_map[raw] = can_smi
        if was_mod:
            result.modified_idx.append(i)

    out = result.df
    out["mol"] = mols
    out["canon_smiles"] = canon
    out["valid_mol"] = valid
    out["was_modified"] = modified

    n_total = len(out)
    n_valid = sum(valid)
    n_invalid = len(result.invalid_idx)
    n_mod = len(result.modified_idx)

    print(f"[{name}] Preprocessed {n_total} molecules: "
          f"{n_valid} valid, {n_invalid} invalid, {n_mod} standardised "
          f"(fragments/charges adjusted).")

    return result


def get_scaffold(mol: Chem.Mol) -> Optional[str]:
    """Return Bemis-Murcko scaffold SMILES for a molecule, or None."""
    if mol is None:
        return None
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


if __name__ == "__main__":
    from data_loader import load_train_inhibition, load_test_blinded
    for nm, d in [("train", load_train_inhibition()), ("test", load_test_blinded())]:
        r = process_smiles(d, name=nm)
        print(f"  -> first invalid indices: {r.invalid_idx[:5]}")
