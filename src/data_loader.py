"""
Data loading utilities for the OpenADMET CYP Challenge.
"""
import os
import pandas as pd
from . import constants as C

def _path(fn):
    return os.path.join(C.DATA_RAW, fn)

def load_train_inhibition() -> pd.DataFrame:
    """Primary direct-inhibition training set (pIC50 + CIs for 4 CYPs)."""
    return pd.read_csv(_path(C.TRAIN_INHIBITION_FN))

def load_test_blinded() -> pd.DataFrame:
    """Blinded test set (Molecule_Name + SMILES only)."""
    return pd.read_csv(_path(C.TEST_BLINDED_FN))

def load_train_tdi() -> pd.DataFrame:
    """TDI classification + paired-condition pIC50s."""
    return pd.read_csv(_path(C.TRAIN_TDI_FN))

def load_train_single_concentration() -> pd.DataFrame:
    """Single-concentration primary screen (long format)."""
    return pd.read_csv(_path(C.TRAIN_SINGLE_FN))

def load_train_emax() -> pd.DataFrame:
    """Emax training set."""
    return pd.read_csv(_path(C.TRAIN_EMAX_FN))

def load_all():
    """Return dict with all five datasets."""
    return {
        "train_inhibition": load_train_inhibition(),
        "test_blinded":     load_test_blinded(),
        "train_tdi":        load_train_tdi(),
        "train_single":     load_train_single_concentration(),
        "train_emax":       load_train_emax(),
    }

if __name__ == "__main__":
    for name, df in load_all().items():
        print(f"{name:25s} {df.shape}")
