"""
Constants, paths, endpoint definitions — must match official challenge contract.
"""
import os

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_RAW    = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROC   = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

for d in [DATA_PROC, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------
# Official file names (in data/raw)
# ---------------------------------------------------------------
TRAIN_INHIBITION_FN = "cyp-challenge-TRAIN_inhibition.csv"
TEST_BLINDED_FN     = "cyp-challenge-TEST-BLINDED.csv"
TRAIN_TDI_FN        = "cyp-challenge-TRAIN_TDI.csv"
TRAIN_SINGLE_FN     = "cyp-challenge-single-concentration-TRAIN.csv"
TRAIN_EMAX_FN       = "cyp-challenge-TRAIN_Emax.csv"

# ---------------------------------------------------------------
# Primary regression track targets (direct-inhibition pIC50)
# ---------------------------------------------------------------
CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

TARGET_COLS = {
    "CYP1A2": "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9": "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6": "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4": "CYP3A4_pIC50_direct_inhibition",
}

CI_HIGH_COLS = {
    "CYP1A2": "CYP1A2_pIC50_direct_inhibition_conf_high",
    "CYP2C9": "CYP2C9_pIC50_direct_inhibition_conf_high",
    "CYP2D6": "CYP2D6_pIC50_direct_inhibition_conf_high",
    "CYP3A4": "CYP3A4_pIC50_direct_inhibition_conf_high",
}

CI_LOW_COLS = {
    "CYP1A2": "CYP1A2_pIC50_direct_inhibition_conf_low",
    "CYP2C9": "CYP2C9_pIC50_direct_inhibition_conf_low",
    "CYP2D6": "CYP2D6_pIC50_direct_inhibition_conf_low",
    "CYP3A4": "CYP3A4_pIC50_direct_inhibition_conf_low",
}

STD_COLS = {
    "CYP1A2": "CYP1A2_pIC50_direct_inhibition_std",
    "CYP2C9": "CYP2C9_pIC50_direct_inhibition_std",
    "CYP2D6": "CYP2D6_pIC50_direct_inhibition_std",
    "CYP3A4": "CYP3A4_pIC50_direct_inhibition_std",
}

# ---------------------------------------------------------------
# TDI classification track targets
# ---------------------------------------------------------------
TDI_TARGET_COLS = {
    "CYP2D6": "CYP2D6_is_TDI",
    "CYP3A4": "CYP3A4_is_TDI",
}

# ---------------------------------------------------------------
# Submission column names (must be exact / case-sensitive)
# ---------------------------------------------------------------
SUBMISSION_COLS_REGRESSION = [
    "SMILES",
    "Molecule_Name",
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]

SUBMISSION_COLS_TDI = [
    "SMILES",
    "Molecule_Name",
    "CYP2D6_is_TDI",
    "CYP3A4_is_TDI",
]

ID_COL = "Molecule_Name"
SMILES_COL = "SMILES"

# ---------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------
SEED = 42
