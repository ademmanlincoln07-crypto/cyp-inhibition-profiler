"""
Build and validate the submission CSV for the OpenADMET CYP Challenge.

Direct-inhibition regression file must contain exactly:
  SMILES, Molecule_Name,
  CYP1A2_pIC50_direct_inhibition,
  CYP2C9_pIC50_direct_inhibition,
  CYP2D6_pIC50_direct_inhibition,
  CYP3A4_pIC50_direct_inhibition

...in exactly the 750 rows of the blinded test set, with predictions for all.

TDI classification file must contain exactly:
  SMILES, Molecule_Name, CYP2D6_is_TDI, CYP3A4_is_TDI
(boolean values).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import constants as C


def build_regression_submission(
    test_df: pd.DataFrame,
    preds: dict,
    out_path: str,
) -> pd.DataFrame:
    """
    test_df: the loaded blinded-test DataFrame (with Molecule_Name + SMILES).
    preds: dict {CYP1A2_pIC50_direct_inhibition: np.ndarray of length 750, ...}
           keys must use exact target column names.
    Returns the submission DataFrame and writes it to out_path.
    """
    df = test_df[["SMILES", "Molecule_Name"]].copy()
    expected_cols = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]
    for c in expected_cols:
        if c not in preds:
            raise ValueError(f"Missing predictions for {c}")
        arr = np.asarray(preds[c], dtype=np.float64)
        if arr.shape[0] != len(df):
            raise ValueError(f"Predictions for {c} have length {arr.shape[0]}; expected {len(df)}")
        if not np.all(np.isfinite(arr)):
            bad = int((~np.isfinite(arr)).sum())
            raise ValueError(f"Predictions for {c} contain {bad} non-finite values")
        df[c] = arr

    # Reorder columns exactly
    df = df[C.SUBMISSION_COLS_REGRESSION]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[submission] Saved regression submission -> {out_path}")
    _validate_regression(df)
    return df


def build_tdi_submission(
    test_df: pd.DataFrame,
    preds_tdi: dict,
    out_path: str,
) -> pd.DataFrame:
    df = test_df[["SMILES", "Molecule_Name"]].copy()
    for c in ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]:
        if c not in preds_tdi:
            raise ValueError(f"Missing predictions for {c}")
        arr = np.asarray(preds_tdi[c])
        if arr.shape[0] != len(df):
            raise ValueError(f"Predictions for {c} have length {arr.shape[0]}; expected {len(df)}")
        # Accept bool, 0/1, or probability; threshold probabilities at 0.5
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = (arr >= 0.5).astype(int)
        df[c] = arr.astype(int)
    df = df[C.SUBMISSION_COLS_TDI]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[submission] Saved TDI submission -> {out_path}")
    _validate_tdi(df)
    return df


def _validate_regression(df: pd.DataFrame):
    expected = C.SUBMISSION_COLS_REGRESSION
    got = list(df.columns)
    if got != expected:
        raise ValueError(f"Submission columns mismatch.\nExpected: {expected}\nGot:      {got}")
    if len(df) != 750:
        raise ValueError(f"Submission must have 750 rows, got {len(df)}")
    for c in expected[2:]:
        if not np.all(np.isfinite(df[c].values)):
            raise ValueError(f"Non-finite values in {c}")
    print("[submission] Regression file validated (750 rows, 6 cols, all finite).")


def _validate_tdi(df: pd.DataFrame):
    expected = C.SUBMISSION_COLS_TDI
    got = list(df.columns)
    if got != expected:
        raise ValueError(f"TDI Submission columns mismatch.\nExpected: {expected}\nGot: {got}")
    if len(df) != 750:
        raise ValueError(f"TDI submission must have 750 rows, got {len(df)}")
    for c in expected[2:]:
        vals = df[c].unique()
        if not set(vals).issubset({0, 1, True, False}):
            raise ValueError(f"Non-boolean values in {c}: {vals}")
    print("[submission] TDI file validated (750 rows, 4 cols, boolean).")


if __name__ == "__main__":
    # Sanity check: write a dummy submission
    rng = np.random.default_rng(0)
    test_df = pd.DataFrame({
        "SMILES": ["CCO"] * 750,
        "Molecule_Name": [f"OCNT-{i:07d}" for i in range(750)],
    })
    preds = {c: rng.normal(5, 1, 750) for c in [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]}
    build_regression_submission(
        test_df, preds,
        os.path.join(C.RESULTS_DIR, "FINAL_CYP_CHALLENGE_PREDICTIONS_dummy.csv"),
    )
