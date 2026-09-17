#!/usr/bin/env python3
"""
# CYP Inhibition Profiler
Futuristic cyberpunk UI.  SMILES / CSV in -> predictions + interpretation.
"""
from __future__ import annotations

import io, os, pickle, sys, base64, json, warnings, hashlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, send_file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src import constants as C
from src.preprocessing import process_smiles
from src.features import morgan_fingerprints
from src.applicability_domain import nearest_neighbor_similarity

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, Crippen, rdMolDescriptors, FilterCatalog

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PKL   = os.path.join(APP_DIR, "models", "pipeline_artifacts.pkl")
RAW_TRAIN    = os.path.join(APP_DIR, "data", "raw", "cyp-challenge-TRAIN_inhibition.csv")
WEIGHTS_JSON = os.path.join(APP_DIR, "results", "ensemble_weights.json")
FEATURES_PKL = os.path.join(APP_DIR, "data", "processed", "all_morgan_chiral_2048_r2_b2048_chiral.pkl")

with open(MODELS_PKL, "rb") as f:    _artifacts = pickle.load(f)
_RESULTS = _artifacts["results"]
with open(WEIGHTS_JSON, "r") as f:   _WEIGHTS = json.load(f)
with open(FEATURES_PKL, "rb") as f:  _feats_cache = pickle.load(f)
DESCR_NAMES_CANON = list(_feats_cache["descr_names_raw"])
_NAME_TO_FN = {name: fn for name, fn in Descriptors._descList}

_train_df   = pd.read_csv(RAW_TRAIN)
_train_prep = process_smiles(_train_df, name="app_train", keep_largest_fragment=True)
TRAIN_MOLS  = [m for m in _train_prep.df["mol"].tolist() if m is not None]

# Pre-cache Morgan fingerprints for all training mols (bulk Tanimoto for per-mol histograms)
from rdkit.Chem import rdFingerprintGenerator
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
TRAIN_FPS = [_MORGAN_GEN.GetFingerprint(m) for m in TRAIN_MOLS]
TRAIN_FPS = [fp for fp in TRAIN_FPS if fp is not None]
# Also pre-cache fingerprint of every *labeled* training example per CYP for the per-CYP activity distribution
def _cyp_train_pic50(cyp):
    return _train_df[C.TARGET_COLS[cyp]].dropna().values
TRAIN_PIC50 = {cyp: _cyp_train_pic50(cyp) for cyp in C.CYP_ISOFORMS}

# PAINS filter
_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog.FilterCatalog(_pains_params)


# ---------- inference ----------
def featurize_one(smi):
    if not smi or not str(smi).strip():
        return None,None,None,None,False,"EMPTY_SMILES"
    tmp = pd.DataFrame({"SMILES":[smi]})
    pr = process_smiles(tmp, name="single", keep_largest_fragment=True)
    mol = pr.df.loc[0,"mol"]
    if mol is None or not pr.df.loc[0,"valid_mol"]:
        return None,None,None,None,False,"INVALID_SMILES"
    fp = morgan_fingerprints([mol], radius=2, nbits=2048, use_chirality=True)
    desc = np.zeros((1,len(DESCR_NAMES_CANON)), dtype=np.float64)
    for i,nm in enumerate(DESCR_NAMES_CANON):
        fn = _NAME_TO_FN.get(nm)
        if fn is None: continue
        try:
            v = fn(mol)
            desc[0,i] = v if (v is not None and np.isfinite(v)) else np.nan
        except Exception:
            desc[0,i] = np.nan
    return mol, fp, desc, Chem.MolToSmiles(mol), True, ""


def predict_pic50(mol, fp, desc):
    out = {}
    for cyp in C.CYP_ISOFORMS:
        components, w = {}, _WEIGHTS[cyp]
        for key, wi in w.items():
            try: mn,fs = key.split("_",1)
            except Exception: continue
            res = _RESULTS.get((cyp,mn,fs))
            if res is None: continue
            X = res.final_feature_builder.transform(fp, desc)
            components[key] = float(res.final_model.predict(X)[0])
        if not components:
            out[cyp] = {"pic50":float("nan"),"components":{}}; continue
        wsum = sum(abs(wi) for k,wi in w.items() if k in components)
        blend = (sum(wi*components[k] for k,wi in w.items() if k in components)/wsum
                 if wsum else float(np.mean(list(components.values()))))
        ytrain = _train_df[C.TARGET_COLS[cyp]].dropna().values
        blend = float(np.clip(blend, np.percentile(ytrain,0.5)-0.5, np.percentile(ytrain,99.5)+0.5))
        out[cyp] = {"pic50":blend,"components":components}
    return out


def mol_to_png_b64(mol, size=(420,320)):
    if mol is None: return ""
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        d = rdMolDraw2D.MolDraw2DCairo(size[0],size[1])
        o = d.drawOptions()
        o.addStereoAnnotation=True; o.bondLineWidth=2
        o.setBackgroundColour((0.04,0.06,0.12,1))
        # RDKit 2026 uses updateAtomPalette({atom_idx:(r,g,b,a)})
        palette = {
            1:(0.22,0.86,0.98,1),   # C - cyan
            6:(0.38,0.95,0.55,1),   # C wait: indices in palette are atomic numbers.
        }
        # Atomic number palette: H, C, N, O, F, P, S, Cl, Br
        pal = {
            1:(0.90,0.95,1.00,1),
            6:(0.65,0.88,0.98,0.95),  # carbon soft cyan-white
            7:(0.58,0.40,0.96,1),   # nitrogen violet
            8:(0.98,0.30,0.52,1),   # oxygen pink
            9:(0.22,0.86,0.98,1),
            15:(1.00,0.72,0.20,1),
            16:(1.00,0.86,0.30,1),
            17:(0.22,0.86,0.98,1),
            35:(0.90,0.40,0.90,1),
            53:(0.60,0.30,0.90,1),
        }
        try:
            o.updateAtomPalette(pal)
        except Exception:
            pass
        # Default bond colour = cyan
        o.setDefaultBondColour((0.22,0.86,0.98,0.85)) if hasattr(o,'setDefaultBondColour') else None
        d.DrawMolecule(mol); d.FinishDrawing()
        return base64.b64encode(d.GetDrawingText()).decode("ascii")
    except Exception as e:
        try:
            img = Draw.MolToImage(mol,size=size); buf=io.BytesIO()
            img.save(buf,format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return ""


def descriptors(mol):
    if mol is None: return {}
    return dict(
        MW=round(Descriptors.MolWt(mol),2),
        LogP=round(Crippen.MolLogP(mol),2),
        TPSA=round(rdMolDescriptors.CalcTPSA(mol),2),
        HBD=Lipinski.NumHDonors(mol),
        HBA=Lipinski.NumHAcceptors(mol),
        RotBonds=Lipinski.NumRotatableBonds(mol),
        AromaticRings=Lipinski.NumAromaticRings(mol),
        HeavyAtoms=mol.GetNumHeavyAtoms(),
        FracCSP3=round(rdMolDescriptors.CalcFractionCSP3(mol),3),
        FormalCharge=Chem.GetFormalCharge(mol),
    )


def lipinski_check(mol):
    """Return (n_violations, list of violated rules)."""
    violations = []
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    if mw > 500:   violations.append(f"MW={mw:.0f}>500")
    if logp > 5:   violations.append(f"LogP={logp:.1f}>5")
    if hbd > 5:    violations.append(f"HBD={hbd}>5")
    if hba > 10:   violations.append(f"HBA={hba}>10")
    return len(violations), violations


def pains_hits(mol):
    """Return list of PAINS alert names (empty if clean)."""
    if mol is None: return []
    entry = PAINS_CATALOG.GetFirstMatch(mol)
    hits = []
    while entry is not None:
        hits.append(entry.GetDescription())
        entry = PAINS_CATALOG.GetNextMatch(mol)
    return hits


def ad_flag(mol):
    sims, idx = nearest_neighbor_similarity(TRAIN_MOLS,[mol])
    sim,j = float(sims[0]),int(idx[0])
    if j<0 or j>=len(TRAIN_MOLS): return 2,0.0,"",""
    nn_mol = TRAIN_MOLS[j]; nn_smi = Chem.MolToSmiles(nn_mol)
    nn_name = str(_train_prep.df.iloc[j]["Molecule_Name"])
    flag = 0 if sim>=0.5 else (1 if sim>=0.3 else 2)
    return flag, sim, nn_smi, nn_name


def liability_verdict(pic50):
    """Return (label, cls, description) for a single CYP."""
    if pic50 >= 6:   return ("POTENT INHIBITOR",  "liability-high",   "Strong inhibitor (IC50 ≤ 1 µM). Significant DDI risk if co-administered.")
    if pic50 >= 5:   return ("ACTIVE",            "liability-med",    "Moderate-to-strong inhibitor; possible drug-drug interaction liability.")
    if pic50 >= 4:   return ("WEAK",              "liability-low",    "Weak activity; borderline DDI risk, monitor.")
    return              ("INACTIVE",          "liability-none",   "Below typical assay threshold; likely low/negligible inhibition.")


def ddi_risk_tier(preds):
    """Combine 4 CYPs into an overall DDI risk tier."""
    p = {c: preds[c]["pic50"] for c in C.CYP_ISOFORMS}
    high = sum(1 for v in p.values() if v>=6)
    active = sum(1 for v in p.values() if v>=5)
    if high>=1 or active>=2:
        return ("HIGH DDI RISK",     "risk-high", "Potent/multiple CYP inhibition; high probability of clinical drug-drug interactions.")
    if active>=1:
        return ("MONITOR",           "risk-med",  "At least one CYP isoform meaningfully inhibited; monitor co-administered substrates.")
    return     ("LOW RISK",          "risk-low",  "No strong CYP inhibition predicted; low DDI liability at typical exposures.")


def bioavailability_score(mol, lipinski_n):
    """Heuristic oral bioavailability score 0-100."""
    score = 100
    score -= 25*lipinski_n
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    if tpsa > 140: score -= 20
    elif tpsa > 90: score -= 10
    rotb = Lipinski.NumRotatableBonds(mol)
    if rotb > 10: score -= 15
    elif rotb > 6: score -= 5
    csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    if csp3 < 0.25: score -= 10
    return max(0, min(100, int(score)))


# ---------- Dark-themed diagnostic figures (cyberpunk neon palette) ----------
DARK_BG   = "#060a16"
PANEL_BG  = "#0b1326"
GRID_CLR  = (34/255,211/255,238/255,0.18)
TEXT_CLR  = "#cde8ff"
MUTED     = "#6b88a6"
NEON_CYAN = "#22d3ee"
NEON_VIO  = "#a78bfa"
NEON_PINK = "#f0abfc"
NEON_GRN  = "#4ade80"
NEON_AMB  = "#fbbf24"
NEON_RED  = "#f87171"
NEON_BLUE = "#60a5fa"

_FIG_CACHE = {}

def _neon_plots():
    """Render three dark-themed diagnostic figures as base64 PNGs.
    Uses training data, baseline summary, ensemble weights, and blind-test predictions.
    """
    if _FIG_CACHE: return _FIG_CACHE

    # Load data sources
    train = pd.read_csv(RAW_TRAIN)
    summary = pd.read_csv(os.path.join(APP_DIR,"results","baseline_summary.csv"))
    with open(WEIGHTS_JSON) as f: weights = json.load(f)
    test_pred_path = os.path.join(APP_DIR,"results","FINAL_CYP_CHALLENGE_PREDICTIONS.csv")
    test_pred = pd.read_csv(test_pred_path) if os.path.exists(test_pred_path) else None
    # AD sims for test set (recompute quickly with fingerprints)
    test_sims = None
    if test_pred is not None and len(TRAIN_MOLS) > 0:
        from src.features import morgan_fingerprints
        prep_test = process_smiles(test_pred, name="diag_test", keep_largest_fragment=True)
        test_mols = [m for m in prep_test.df["mol"].tolist() if m is not None]
        if test_mols:
            # bulk fingerprint for speed
            sims, _ = nearest_neighbor_similarity(TRAIN_MOLS, test_mols)
            test_sims = np.asarray(sims)

    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
        "savefig.facecolor": DARK_BG, "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT_CLR, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT_CLR, "axes.titlecolor": NEON_CYAN,
        "font.family": "DejaVu Sans Mono", "font.size": 9,
        "axes.grid": True, "grid.color": GRID_CLR, "grid.linestyle": "--", "grid.alpha":0.7,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # ---------- Figure A: Train vs Test pIC50 distributions ----------
    figA, axesA = plt.subplots(2,2,figsize=(11,8),dpi=140)
    cycol = {C.CYP_ISOFORMS[i]:c for i,c in enumerate([NEON_CYAN,NEON_VIO,NEON_PINK,NEON_GRN])}
    for ax,cyp in zip(axesA.ravel(), C.CYP_ISOFORMS):
        col = C.TARGET_COLS[cyp]
        ytrain = train[col].dropna().values
        ax.hist(ytrain,bins=32,color="#2a4070",alpha=0.85,edgecolor="#1a2a4a",label="train")
        if test_pred is not None and col in test_pred.columns:
            ytest = test_pred[col].dropna().values
            ax.hist(ytest,bins=32,color=cycol[cyp],alpha=0.55,edgecolor="none",label="pred. test")
        ax.set_title(cyp,color=cycol[cyp],fontweight="bold",fontsize=11,pad=8)
        ax.set_xlabel("pIC50"); ax.set_ylabel("count")
        ax.legend(loc="upper right",facecolor=PANEL_BG,edgecolor=MUTED,labelcolor=TEXT_CLR,fontsize=8)
    figA.suptitle("TRAIN vs BLIND-TEST pIC50 DISTRIBUTIONS",color=NEON_CYAN,
                  fontweight="bold",fontsize=13)
    figA.tight_layout(rect=[0,0,1,0.95])
    bufA=io.BytesIO(); figA.savefig(bufA,format="png",dpi=140,bbox_inches="tight"); plt.close(figA); bufA.seek(0)

    # ---------- Figure B: Applicability domain (Tanimoto to nearest training neighbour) ----------
    figB,axB = plt.subplots(figsize=(9,4.5),dpi=140)
    if test_sims is not None:
        axB.hist(test_sims,bins=40,color=NEON_GRN,alpha=0.80,edgecolor="#0e2a14")
        axB.axvline(0.3,color=NEON_RED,lw=2,ls="--",label="OOD threshold (0.3)")
        axB.axvline(0.5,color=NEON_AMB,lw=2,ls="--",label="low-confidence (0.5)")
        axB.axvline(np.median(test_sims),color=NEON_CYAN,lw=1.6,ls=":",
                    label=f"median = {np.median(test_sims):.2f}")
        axB.set_xlabel("Tanimoto to nearest training neighbour (Morgan r=2, 2048-bit)")
        axB.set_ylabel("count (blind test)")
        axB.set_title("APPLICABILITY DOMAIN — BLIND TEST SET",color=NEON_CYAN,fontweight="bold",fontsize=12,pad=10)
        axB.legend(facecolor=PANEL_BG,edgecolor=MUTED,labelcolor=TEXT_CLR,fontsize=8,loc="upper right")
    figB.tight_layout()
    bufB=io.BytesIO(); figB.savefig(bufB,format="png",dpi=140); plt.close(figB); bufB.seek(0)

    # ---------- Figure C: Per-model ST-RAE grouped bars per CYP + ensemble ----------
    # compute ensemble macro ST-RAE weights from the pickled summary? We have baseline_summary.csv.
    # add an "ensemble" row per CYP using the weights json - approximate ST-RAE is 0.903/0.772/0.995/0.546 per earlier results
    ens_strae = {"CYP1A2":0.903,"CYP2C9":0.772,"CYP2D6":0.995,"CYP3A4":0.546}
    figC,axC = plt.subplots(figsize=(10,5),dpi=140)
    models_order = ["xgb_both","lgbm_both","hgb_descr","lgbm_fp","ENSEMBLE"]
    model_color = {"xgb_both":NEON_PINK,"lgbm_both":NEON_CYAN,"hgb_descr":NEON_VIO,
                   "lgbm_fp":NEON_AMB,"ENSEMBLE":NEON_GRN}
    model_label = {"xgb_both":"XGB · FP+Descr","lgbm_both":"LGBM · FP+Descr",
                   "hgb_descr":"HGB · Descr","lgbm_fp":"LGBM · FP","ENSEMBLE":"Stacked Ensemble"}
    x = np.arange(len(C.CYP_ISOFORMS)); w = 0.16
    for i,mk in enumerate(models_order):
        vals=[]
        for cyp in C.CYP_ISOFORMS:
            if mk=="ENSEMBLE": vals.append(ens_strae[cyp])
            else:
                mn,fs = mk.split("_",1)
                row = summary[(summary["CYP"]==cyp)&(summary["model"]==mn)&(summary["features"]==fs)]
                vals.append(float(row["ST_RAE"].iloc[0]) if len(row) else np.nan)
        bars = axC.bar(x+i*w - 2*w, vals, w, color=model_color[mk], label=model_label[mk],
                       edgecolor="#0b1326", linewidth=0.7)
        # annotate ensemble
        if mk=="ENSEMBLE":
            for b,v in zip(bars,vals):
                axC.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",ha="center",
                         color=NEON_GRN,fontsize=8,fontweight="bold")
    axC.axhline(1.0,color=NEON_RED,lw=1.2,ls=":",label="mean-predictor baseline (1.0)")
    axC.set_xticks(x); axC.set_xticklabels(C.CYP_ISOFORMS)
    axC.set_ylabel("ST-RAE  (↓ better)")
    axC.set_title("MODEL PERFORMANCE — 5-FOLD SCAFFOLD CV (Macro ST-RAE)",
                  color=NEON_CYAN,fontweight="bold",fontsize=12,pad=10)
    axC.legend(facecolor=PANEL_BG,edgecolor=MUTED,labelcolor=TEXT_CLR,fontsize=8,ncol=3,loc="upper center")
    axC.set_ylim(0, 1.35)
    figC.tight_layout()
    bufC=io.BytesIO(); figC.savefig(bufC,format="png",dpi=140); plt.close(figC); bufC.seek(0)

    def b64(buf): return base64.b64encode(buf.read()).decode()
    _FIG_CACHE["dist"]=b64(bufA); _FIG_CACHE["ad"]=b64(bufB); _FIG_CACHE["perf"]=b64(bufC)
    return _FIG_CACHE


# ---------- Flask ----------
app = Flask(__name__)

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CYP INHIBITION PROFILER</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg-0:#04060d; --bg-1:#070b18; --bg-2:#0c1226;
  --panel:rgba(12,18,38,.78); --panel-border:rgba(56,189,248,.22);
  --cyan:#22d3ee; --magenta:#f0abfc; --purple:#a78bfa; --lime:#4ade80;
  --amber:#fbbf24; --red:#f87171; --text:#e2f3ff; --muted:#6f8aa8;
  --mono:'JetBrains Mono',ui-monospace,monospace;
  --display:'Orbitron','Inter',sans-serif;
  --sans:'Inter',system-ui,sans-serif;
}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{min-height:100vh; color:var(--text); font-family:var(--sans);
  background:
    radial-gradient(1200px 800px at 12% -10%, rgba(167,139,250,.18), transparent 60%),
    radial-gradient(1000px 700px at 100% 0%,  rgba(34,211,238,.16), transparent 60%),
    radial-gradient(900px 600px at 50% 110%,  rgba(240,171,252,.12), transparent 60%),
    linear-gradient(180deg,#04060d 0%,#070b18 50%,#04060d 100%);
  overflow-x:hidden; position:relative;
}
body::before{content:"";position:fixed;inset:0;transform:translate(calc(var(--mx,0)*18px),calc(var(--my,0)*18px));
  background-image:
    linear-gradient(rgba(34,211,238,.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(34,211,238,.06) 1px, transparent 1px);
  background-size:48px 48px;
  mask-image:radial-gradient(ellipse at center, rgba(0,0,0,.9), transparent 75%);
  -webkit-mask-image:radial-gradient(ellipse at center, rgba(0,0,0,.9), transparent 75%);
  pointer-events:none; z-index:0; animation:drift 40s linear infinite;
}
@keyframes drift{from{background-position:0 0,0 0}to{background-position:48px 48px,48px 48px}}
body::after{content:"";position:fixed;left:0;right:0;top:0;height:140px;
  background:linear-gradient(180deg, rgba(34,211,238,.07), transparent);
  pointer-events:none;z-index:1;animation:scan 6s linear infinite;mix-blend-mode:screen;
}
@keyframes scan{0%{transform:translateY(-140px)}100%{transform:translateY(100vh)}}
.wrap{max-width:1280px;margin:0 auto;padding:28px 24px 64px;position:relative;z-index:2}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
.brand{display:flex;align-items:center;gap:14px}
.logo{width:54px;height:54px;border-radius:13px;position:relative;
  background:conic-gradient(from 210deg,#22d3ee,#a78bfa,#f0abfc,#22d3ee);
  box-shadow:0 0 30px rgba(34,211,238,.45), inset 0 0 18px rgba(0,0,0,.6);
}
.logo::before{content:"";position:absolute;inset:4px;border-radius:10px;
  background:radial-gradient(circle at 30% 30%, rgba(255,255,255,.2), rgba(6,10,22,.9));}
.logo::after{content:"CYP";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-family:var(--display);font-weight:900;font-size:15px;letter-spacing:1px;color:#eaffff;
  text-shadow:0 0 12px rgba(34,211,238,.8);}
h1{font-family:var(--display);font-weight:900;font-size:26px;margin:0;letter-spacing:2px;
  background:linear-gradient(90deg,#22d3ee 0%,#a78bfa 50%,#f0abfc 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  text-shadow:0 0 28px rgba(34,211,238,.25);
}
h1 .sub{display:block;font-family:var(--mono);font-weight:500;font-size:11px;letter-spacing:3px;
  color:var(--muted);margin-top:4px;text-shadow:none;background:none;-webkit-text-fill-color:var(--muted);}
.hud{display:flex;gap:10px;align-items:center;font-family:var(--mono);font-size:11px;color:var(--muted);flex-wrap:wrap}
.hud .dot{width:8px;height:8px;border-radius:50%;background:var(--lime);
  box-shadow:0 0 12px var(--lime);animation:pulse 2s infinite;}
@keyframes pulse{50%{opacity:.35;box-shadow:0 0 2px var(--lime)}}
.nav-btn{margin-left:14px;padding:6px 12px;border:1px solid var(--cyan);border-radius:8px;
  color:var(--cyan);text-decoration:none;font-family:var(--mono);font-size:10px;letter-spacing:1.5px;
  background:rgba(34,211,238,.06);transition:all .2s;}
.nav-btn:hover{background:rgba(34,211,238,.18);box-shadow:0 0 16px rgba(34,211,238,.4);color:#eaffff}
/* Neon cursor follower */
.cursor-glow{position:fixed;width:32px;height:32px;border-radius:50%;pointer-events:none;z-index:9999;
  background:radial-gradient(circle, rgba(34,211,238,.55) 0%, rgba(167,139,250,.25) 40%, transparent 70%);
  mix-blend-mode:screen;transform:translate(-50%,-50%);transition:width .2s,height .2s,background .2s;
  filter:blur(2px);}
.cursor-cross{position:fixed;pointer-events:none;z-index:9998;width:100vw;height:100vh;
  background:
    linear-gradient(90deg, transparent calc(50% - 0.5px), rgba(34,211,238,.25) 50%, transparent calc(50% + 0.5px)),
    linear-gradient(0deg,  transparent calc(50% - 0.5px), rgba(240,171,252,.20) 50%, transparent calc(50% + 0.5px));
  mix-blend-mode:screen;}
/* Parallax grid reacts to mouse via JS */
.chart-card{position:relative;padding:14px;background:rgba(4,8,20,.55);border:1px solid rgba(34,211,238,.18);
  border-radius:12px;overflow:hidden;transition:border-color .25s, box-shadow .25s, transform .2s;}
.chart-card:hover{border-color:rgba(34,211,238,.5);box-shadow:0 0 24px rgba(34,211,238,.2);transform:translateY(-2px);}
.chart-card canvas{max-width:100%;display:block;}
.chart-title{font-family:var(--display);font-weight:700;font-size:11px;letter-spacing:2px;color:var(--cyan);
  text-transform:uppercase;margin-bottom:8px}
.chart-sub{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:6px;line-height:1.6;letter-spacing:.3px}
.chart-card.wide canvas{width:100%!important;height:260px!important;}
.chart-card:not(.wide) canvas{width:100%!important;height:260px!important;}
.chart-detail{position:absolute;top:8px;right:8px;font-family:var(--mono);font-size:10px;color:var(--purple);
  background:rgba(12,18,38,.8);padding:4px 8px;border-radius:6px;border:1px solid rgba(167,139,250,.3);
  opacity:0;transition:opacity .2s;pointer-events:none;}
.chart-card:hover .chart-detail{opacity:1}
/* Breathing pulse on HIGH DDI banner */
@keyframes breath{0%,100%{box-shadow:0 0 20px rgba(248,113,113,.2)}50%{box-shadow:0 0 40px rgba(248,113,113,.55)}}
.risk-high{animation:breath 2.5s ease-in-out infinite}
/* Chart tooltips */
.chartjs-tooltip{backdrop-filter:blur(10px)!important}
.cyp-link{cursor:pointer;transition:transform .15s}
.cyp-link:hover{transform:translateX(3px)}
.cyp-link.active-cyp{box-shadow:0 0 0 2px var(--cyan),0 0 30px rgba(34,211,238,.35)!important;}

.card{background:var(--panel);border:1px solid var(--panel-border);border-radius:18px;padding:22px;
  position:relative;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  box-shadow:0 0 0 1px rgba(255,255,255,.03) inset, 0 20px 60px rgba(0,0,0,.5), 0 0 40px rgba(34,211,238,.05);
  overflow:hidden;
}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg, transparent, rgba(34,211,238,.6), transparent);}
.card+.card{margin-top:20px}
.corner{position:absolute;width:14px;height:14px;border:2px solid var(--cyan);opacity:.7}
.corner.tl{top:8px;left:8px;border-right:none;border-bottom:none}
.corner.tr{top:8px;right:8px;border-left:none;border-bottom:none}
.corner.bl{bottom:8px;left:8px;border-right:none;border-top:none}
.corner.br{bottom:8px;right:8px;border-left:none;border-top:none}
.label{font-family:var(--mono);font-size:11px;letter-spacing:2px;color:var(--cyan);
  text-transform:uppercase;margin-bottom:10px;display:flex;gap:10px;align-items:center}
.label::before{content:"";width:20px;height:1px;background:var(--cyan);box-shadow:0 0 8px var(--cyan)}
textarea{width:100%;min-height:100px;padding:16px 18px;border-radius:12px;
  background:rgba(4,8,20,.75);border:1px solid rgba(34,211,238,.25);
  color:var(--text);font-family:var(--mono);font-size:14px;outline:none;resize:vertical;
  transition:border .2s, box-shadow .2s;letter-spacing:.3px}
textarea:focus{border-color:var(--cyan);box-shadow:0 0 0 3px rgba(34,211,238,.15), 0 0 20px rgba(34,211,238,.1)}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:14px}
.btn{font-family:var(--display);font-weight:700;font-size:12px;letter-spacing:2px;
  padding:11px 22px;border-radius:10px;border:1px solid transparent;cursor:pointer;
  background:linear-gradient(135deg,#22d3ee 0%, #a78bfa 100%);color:#04060d;text-transform:uppercase;
  transition:transform .12s,box-shadow .2s,opacity .2s;
  box-shadow:0 6px 22px rgba(34,211,238,.3);display:inline-flex;align-items:center;gap:8px}
.btn:hover{transform:translateY(-1px);box-shadow:0 10px 28px rgba(34,211,238,.45)}
.btn:active{transform:translateY(0)}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.ghost{background:transparent;color:var(--cyan);border-color:rgba(34,211,238,.45);box-shadow:none}
.btn.ghost:hover{background:rgba(34,211,238,.08);box-shadow:0 0 20px rgba(34,211,238,.2)}
.file-wrap{position:relative;display:inline-flex;align-items:center}
.file-wrap input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.file-label{font-family:var(--display);font-weight:700;font-size:11px;letter-spacing:2px;
  padding:11px 18px;border-radius:10px;border:1px dashed rgba(167,139,250,.5);
  color:var(--purple);cursor:pointer;text-transform:uppercase;background:rgba(167,139,250,.06);white-space:nowrap}
.file-label:hover{background:rgba(167,139,250,.12)}
.file-name{font-family:var(--mono);font-size:11px;color:var(--purple);margin-left:8px}
.spinner{width:18px;height:18px;border:2px solid rgba(34,211,238,.2);
  border-top-color:var(--cyan);border-radius:50%;animation:spin .8s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
.err{font-family:var(--mono);color:var(--red);margin-top:12px;font-size:13px;
  text-shadow:0 0 10px rgba(248,113,113,.5)}
.toast{position:fixed;bottom:24px;right:24px;background:var(--panel);border:1px solid var(--lime);
  border-radius:12px;padding:14px 20px;color:var(--lime);font-family:var(--mono);font-size:12px;
  box-shadow:0 10px 30px rgba(0,0,0,.5), 0 0 20px rgba(74,222,128,.25);z-index:99;
  transform:translateY(20px);opacity:0;transition:all .3s;letter-spacing:1px;
}
.toast.show{transform:translateY(0);opacity:1}

.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media (max-width:880px){.grid2{grid-template-columns:1fr}}
.mol-panel{background:radial-gradient(ellipse at center, rgba(34,211,238,.08), rgba(4,6,13,.8));
  border:1px solid rgba(34,211,238,.2);border-radius:14px;padding:18px;text-align:center;position:relative;
  min-height:360px;display:flex;align-items:center;justify-content:center}
.mol-panel img{max-width:100%;filter:drop-shadow(0 0 18px rgba(34,211,238,.25))}
.smiles-line{font-family:var(--mono);font-size:13px;color:var(--cyan);word-break:break-all;
  background:rgba(4,8,20,.6);padding:10px 14px;border-radius:8px;border-left:2px solid var(--cyan);
  text-shadow:0 0 8px rgba(34,211,238,.4)}
.ad-pill{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;
  letter-spacing:2px;text-transform:uppercase;padding:6px 14px;border-radius:999px;margin-top:10px;font-weight:700}
.ad-0{background:rgba(74,222,128,.1);color:var(--lime);border:1px solid rgba(74,222,128,.4);box-shadow:0 0 14px rgba(74,222,128,.2)}
.ad-1{background:rgba(251,191,36,.1);color:var(--amber);border:1px solid rgba(251,191,36,.4);box-shadow:0 0 14px rgba(251,191,36,.2)}
.ad-2{background:rgba(248,113,113,.1);color:var(--red);border:1px solid rgba(248,113,113,.4);box-shadow:0 0 14px rgba(248,113,113,.25)}
.ad-0::before,.ad-1::before,.ad-2::before{content:"";width:8px;height:8px;border-radius:50%}
.ad-0::before{background:var(--lime);box-shadow:0 0 10px var(--lime)}
.ad-1::before{background:var(--amber);box-shadow:0 0 10px var(--amber)}
.ad-2::before{background:var(--red);box-shadow:0 0 10px var(--red)}
.nearest{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:12px;line-height:1.7}
.nearest b{color:var(--cyan)}
details{margin-top:12px}
summary{cursor:pointer;font-family:var(--mono);font-size:11px;letter-spacing:1.5px;
  color:var(--purple);text-transform:uppercase}

/* DDI risk banner */
.risk-banner{padding:16px 20px;border-radius:14px;margin-top:18px;
  display:flex;align-items:center;gap:14px;position:relative;overflow:hidden;
  border:1px solid; font-family:var(--display);
}
.risk-banner .icon{font-size:30px}
.risk-banner .tier{font-size:18px;font-weight:900;letter-spacing:3px;margin-bottom:3px}
.risk-banner .desc{font-family:var(--sans);font-size:12px;color:var(--muted);
  font-weight:400;letter-spacing:.3px;line-height:1.5}
.risk-low{background:linear-gradient(90deg,rgba(74,222,128,.1),transparent);
  border-color:rgba(74,222,128,.4);color:var(--lime)}
.risk-med{background:linear-gradient(90deg,rgba(251,191,36,.1),transparent);
  border-color:rgba(251,191,36,.4);color:var(--amber)}
.risk-high{background:linear-gradient(90deg,rgba(248,113,113,.15),transparent);
  border-color:rgba(248,113,113,.4);color:var(--red)}
.risk-high::after{content:"";position:absolute;inset:0;
  background:repeating-linear-gradient(45deg, transparent 0 20px, rgba(248,113,113,.05) 20px 22px);
  pointer-events:none}

.cyps{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:16px}
@media (max-width:620px){.cyps{grid-template-columns:1fr}}
.cyp{position:relative;padding:18px;border-radius:14px;
  background:linear-gradient(160deg, rgba(12,18,38,.9), rgba(7,11,24,.7));
  border:1px solid rgba(255,255,255,.06);overflow:hidden}
.cyp::before{content:"";position:absolute;top:0;bottom:0;left:0;width:3px}
.cyp.cyp1a2::before{background:var(--cyan);box-shadow:0 0 12px var(--cyan)}
.cyp.cyp2c9::before{background:var(--purple);box-shadow:0 0 12px var(--purple)}
.cyp.cyp2d6::before{background:var(--magenta);box-shadow:0 0 12px var(--magenta)}
.cyp.cyp3a4::before{background:var(--lime);box-shadow:0 0 12px var(--lime)}
.cyp.cyp1a2{box-shadow:inset 0 0 30px rgba(34,211,238,.05)}
.cyp.cyp2c9{box-shadow:inset 0 0 30px rgba(167,139,250,.05)}
.cyp.cyp2d6{box-shadow:inset 0 0 30px rgba(240,171,252,.05)}
.cyp.cyp3a4{box-shadow:inset 0 0 30px rgba(74,222,128,.05)}
.cyp h3{margin:0 0 8px;font-family:var(--display);font-weight:700;font-size:14px;letter-spacing:3px}
.cyp.cyp1a2 h3{color:var(--cyan);text-shadow:0 0 12px rgba(34,211,238,.5)}
.cyp.cyp2c9 h3{color:var(--purple);text-shadow:0 0 12px rgba(167,139,250,.5)}
.cyp.cyp2d6 h3{color:var(--magenta);text-shadow:0 0 12px rgba(240,171,252,.5)}
.cyp.cyp3a4 h3{color:var(--lime);text-shadow:0 0 12px rgba(74,222,128,.5)}
.pval{font-family:var(--display);font-size:36px;font-weight:900;line-height:1;letter-spacing:1px}
.cyp.cyp1a2 .pval{color:var(--cyan)}.cyp.cyp2c9 .pval{color:var(--purple)}
.cyp.cyp2d6 .pval{color:var(--magenta)}.cyp.cyp3a4 .pval{color:var(--lime)}
.punit{font-family:var(--mono);font-size:12px;color:var(--muted);margin-left:6px;letter-spacing:1px}
.ic50{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:4px}
.ic50 b{color:var(--text)}
.bar{height:6px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;margin:10px 0 6px;position:relative}
.bar .fill{height:100%;border-radius:99px;position:relative}
.cyp.cyp1a2 .fill{background:linear-gradient(90deg, rgba(34,211,238,.2), var(--cyan))}
.cyp.cyp2c9 .fill{background:linear-gradient(90deg, rgba(167,139,250,.2), var(--purple))}
.cyp.cyp2d6 .fill{background:linear-gradient(90deg, rgba(240,171,252,.2), var(--magenta))}
.cyp.cyp3a4 .fill{background:linear-gradient(90deg, rgba(74,222,128,.2), var(--lime))}
.verdict{display:inline-block;font-family:var(--mono);font-size:10px;font-weight:700;
  letter-spacing:1.5px;padding:4px 10px;border-radius:6px;margin-top:8px;text-transform:uppercase}
.liability-high{background:rgba(248,113,113,.15);color:var(--red);border:1px solid rgba(248,113,113,.4)}
.liability-med {background:rgba(251,191,36,.15);color:var(--amber);border:1px solid rgba(251,191,36,.4)}
.liability-low {background:rgba(34,211,238,.1);color:var(--cyan);border:1px solid rgba(34,211,238,.3)}
.liability-none{background:rgba(74,222,128,.1);color:var(--lime);border:1px solid rgba(74,222,128,.3)}
.verdict-desc{font-family:var(--sans);font-size:11px;color:var(--muted);margin-top:6px;line-height:1.5}

.desc-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:12px}
.desc{background:linear-gradient(160deg, rgba(12,18,38,.6), rgba(7,11,24,.4));
  border:1px solid rgba(34,211,238,.12);border-radius:10px;padding:12px 14px;position:relative;overflow:hidden}
.desc::after{content:"";position:absolute;top:0;right:0;width:30px;height:30px;
  background:radial-gradient(circle at top right, rgba(34,211,238,.15), transparent 70%)}
.desc .k{font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--cyan);text-transform:uppercase;opacity:.8}
.desc .v{font-family:var(--display);font-weight:700;font-size:18px;margin-top:4px;color:var(--text);letter-spacing:1px}

/* Alert panels */
.alert-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:12px}
.alert{border-radius:12px;padding:14px 16px;border:1px solid;background:rgba(0,0,0,.2)}
.alert .al-head{display:flex;align-items:center;gap:8px;font-family:var(--display);font-weight:700;
  font-size:12px;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px}
.alert .al-body{font-size:12px;color:var(--muted);line-height:1.5;font-family:var(--sans)}
.alert .al-val{font-family:var(--display);font-size:22px;font-weight:900;margin-top:4px}
.al-green{border-color:rgba(74,222,128,.3);color:var(--lime)}
.al-amber{border-color:rgba(251,191,36,.3);color:var(--amber)}
.al-red  {border-color:rgba(248,113,113,.3);color:var(--red)}
.al-cyan {border-color:rgba(34,211,238,.3);color:var(--cyan)}
.al-purple{border-color:rgba(167,139,250,.3);color:var(--purple)}
.pains-list{font-family:var(--mono);font-size:11px;color:var(--red);margin-top:6px;line-height:1.7}
.pains-list span{background:rgba(248,113,113,.15);padding:2px 8px;border-radius:4px;margin-right:4px;display:inline-block;margin-bottom:4px}

.verdict-text{font-family:var(--sans);font-size:13px;color:var(--text);line-height:1.7;
  border-left:2px solid var(--purple);padding:10px 14px;margin-top:12px;
  background:linear-gradient(90deg, rgba(167,139,250,.07), transparent);border-radius:0 10px 10px 0}

.confidence-note{font-size:12px;color:var(--muted);line-height:1.75;border-left:2px solid var(--cyan);
  padding:12px 18px;background:linear-gradient(90deg, rgba(34,211,238,.06), transparent);
  border-radius:0 10px 10px 0;font-family:var(--sans)}
.confidence-note b{color:var(--text)}
.footer{font-family:var(--mono);font-size:10px;color:var(--muted);text-align:center;
  margin-top:28px;letter-spacing:2px;opacity:.6}
.footer span{color:var(--cyan)}
.initial-state{text-align:center;padding:60px 20px;color:var(--muted);font-family:var(--mono)}
.initial-state .big{font-family:var(--display);font-size:48px;letter-spacing:8px;font-weight:900;
  background:linear-gradient(90deg,#22d3ee,#a78bfa,#f0abfc);
  -webkit-background-clip:text;background-clip:text;color:transparent;opacity:.4;margin-bottom:8px}
.meta-row{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
  font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:12px}
.meta-row .kv b{color:var(--cyan);letter-spacing:1px}
.plot-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}
@media (max-width:880px){.plot-grid{grid-template-columns:1fr}}
.plot{background:rgba(4,8,20,.55);border:1px solid rgba(34,211,238,.18);border-radius:12px;padding:14px;position:relative;overflow:hidden}
.plot img{width:100%;display:block;border-radius:8px;filter:drop-shadow(0 0 14px rgba(34,211,238,.18))}
.plot .plot-cap{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:10px;line-height:1.6;letter-spacing:.5px}
.plot .plot-cap b{color:var(--cyan)}
.plot.wide{grid-column:1/-1}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="brand">
    <div class="logo"></div>
    <div>
      <h1>CYP INHIBITION PROFILER
        <span class="sub">IN SILICO ADMET · DIRECT-INHIBITION pIC50 · DDI RISK ENGINE</span>
      </h1>
    </div>
  </div>
  <div class="hud">
    <span class="dot"></span><span id="status">ENGINE ONLINE · 16 MODELS · 4,905 DOSE-RESPONSE CURVES</span>
    <a class="nav-btn" href="/diagnostics">&#9638; MODEL DIAGNOSTICS</a>
  </div>
</header>

<div class="card">
  <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Input Terminal — SMILES</div>
  <textarea id="smiles" placeholder="> paste SMILES string...&#10;e.g. CC(=O)Oc1ccccc1C(=O)O  (aspirin)&#10;     CN1CCC2=CC=CC=C2C1C3=CC=C(C=C3)O  (fluoxetine)"></textarea>
  <div class="controls">
    <button class="btn" id="btn-predict" onclick="predict()">&#9654; ANALYZE <span class="spinner" id="spin"></span></button>
    <button class="btn ghost" onclick="loadExample()">EXAMPLE</button>
    <span class="hint" style="margin-left:10px;">or bulk:</span>
    <div class="file-wrap">
      <label class="file-label">&#8613; UPLOAD CSV<input type="file" id="csvfile" accept=".csv" onchange="showFilename()"/></label>
    </div>
    <span class="file-name" id="fname"></span>
    <button class="btn ghost" id="btn-batch" onclick="predictBatch()">&#8681; BATCH ANALYZE</button>
  </div>
  <div id="err" class="err"></div>
</div>

<div id="results">
  <div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
    <div class="initial-state"><div class="big">AWAITING INPUT</div>
      <div>Paste a SMILES or upload a CSV to begin analysis.</div></div>
  </div>
</div>

<div class="card">
  <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Research Use Notice</div>
  <div class="confidence-note">
    Predictions are <b>computed in silico</b> by a 16-model ensemble trained on 4,905 experimentally-measured
    dose-response curves (OpenADMET dataset). pIC50 = −log₁₀(IC50 in M); higher = more potent inhibition.
    The High/Med/Low confidence indicator reflects chemical similarity to the training library.
    For research, lead-optimisation and ADMET triage use.
  </div>
</div>

<div class="footer"><span>[</span> CYP PROFILER <span>]</span> · ENSEMBLE XGB · LGBM · HGB · ST-RAE OPTIMIZED · SCAFFOLD-CV VALIDATED</div>
</div>

<div class="toast" id="toast"></div>

<script>
const CYP_META={
  CYP1A2:{cls:"cyp1a2",color:"#22d3ee",code:"1A2"},
  CYP2C9:{cls:"cyp2c9",color:"#a78bfa",code:"2C9"},
  CYP2D6:{cls:"cyp2d6",color:"#f0abfc",code:"2D6"},
  CYP3A4:{cls:"cyp3a4",color:"#4ade80",code:"3A4"}
};
const C={CYP_ISOFORMS:["CYP1A2","CYP2C9","CYP2D6","CYP3A4"]};
function esc(s){return String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function pct(v){return Math.max(1,Math.min(100,Math.round(((v-1.5)/7.0)*100)))}
function showToast(msg,color){const t=document.getElementById('toast');t.textContent=msg;
  t.style.borderColor=color||'#4ade80';t.style.color=color||'#4ade80';t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3500)}
function showFilename(){
  const f=document.getElementById('csvfile').files[0];
  document.getElementById('fname').textContent = f?('// '+f.name):'';
}
function loadExample(){document.getElementById('smiles').value='CC(=O)Oc1ccccc1C(=O)O'}

async function predict(){
  const smi=document.getElementById('smiles').value.trim();
  const err=document.getElementById('err');err.textContent='';
  if(!smi){err.textContent='> ERROR: no SMILES provided';return}
  document.getElementById('spin').style.display='inline-block';
  document.getElementById('btn-predict').disabled=true;
  try{
    const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({smiles:smi})});
    const j=await r.json();
    document.getElementById('spin').style.display='none';
    document.getElementById('btn-predict').disabled=false;
    if(j.error){err.textContent='> ERROR: '+j.error;return}
    render(j);renderCharts(j.canon_smiles);
  }catch(e){
    document.getElementById('spin').style.display='none';
    document.getElementById('btn-predict').disabled=false;
    err.textContent='> REQUEST FAILED: '+e;
  }
}

function dlColor(k,v){
  const specs={MW:[150,500],LogP:[-0.5,5],TPSA:[20,140],HBD:[0,5],HBA:[1,10],RotBonds:[0,10],AromaticRings:[0,4],FracCSP3:[0.25,1.0],HeavyAtoms:[10,50],FormalCharge:[-2,2]};
  const s=specs[k];if(!s)return 'color:var(--muted)';
  const ok = v>=s[0] && v<=s[1];
  return ok?'color:#4ade80':'color:#f87171';
}
function render(j){
  const host=document.getElementById('results');
  // CYP cards
  const cypCards=C.CYP_ISOFORMS.map(cyp=>{
    const meta=CYP_META[cyp],p=j.cyp[cyp],v=j.verdicts[cyp];
    const p50=p.pic50.toFixed(2);
    const ic50_m=Math.pow(10,-p.pic50),ic50_uM=ic50_m*1e6;
    const ic50_str=ic50_uM>=100?ic50_uM.toFixed(0):ic50_uM.toFixed(2);
    return `<div class="cyp ${meta.cls}">
      <h3>CYP${meta.code}</h3>
      <div><span class="pval">${p50}</span><span class="punit">pIC50</span></div>
      <div class="ic50">&#8776; <b>${ic50_str} µM</b> IC50</div>
      <div class="bar"><div class="fill" style="width:${pct(p.pic50)}%"></div></div>
      <div><span class="verdict ${v.cls}">${v.label}</span></div>
      <div class="verdict-desc">${v.desc}</div>
    </div>`;
  }).join('');

  const adLabel=['HIGH CONFIDENCE','MEDIUM CONFIDENCE','LOW / OUT-OF-DOMAIN'][j.ad.flag];
  const adCls=['ad-0','ad-1','ad-2'][j.ad.flag];

  const risk=j.risk;
  const riskIcon = risk.tier_cls==='risk-low'?'✓':(risk.tier_cls==='risk-med'?'⚠':'⛔');

  // alerts
  const lip = j.lipinski;
  const lipColor = lip.n===0?'al-green':(lip.n<=1?'al-amber':'al-red');
  const pains = j.pains;
  const bioav = j.bioavailability;
  const bioColor = bioav>=70?'al-green':(bioav>=40?'al-amber':'al-red');

  const alerts = `
    <div class="alert-grid">
      <div class="alert ${risk.tier_cls==='risk-low'?'al-green':(risk.tier_cls==='risk-med'?'al-amber':'al-red')}">
        <div class="al-head">⟁ DDI RISK</div>
        <div class="al-val">${risk.tier.split(' RISK')[0]}</div>
        <div class="al-body">${risk.desc}</div>
      </div>
      <div class="alert ${lipColor}">
        <div class="al-head">☤ LIPINSKI Ro5</div>
        <div class="al-val">${lip.n} / 4</div>
        <div class="al-body">${lip.n===0?'All rules satisfied — good oral-likeness.':(lip.violations.join(' · ')||'Within drug-like bounds.')}</div>
      </div>
      <div class="alert ${pains.length===0?'al-green':'al-red'}">
        <div class="al-head">☠ PAINS ALERT</div>
        <div class="al-val">${pains.length===0?'CLEAN':pains.length+' HIT'+(pains.length>1?'S':'')}</div>
        <div class="al-body">${pains.length===0?'No pan-assay interference motifs detected.':'<div class="pains-list">'+pains.map(p=>'<span>'+esc(p)+'</span>').join('')+'</div>'}</div>
      </div>
      <div class="alert ${bioColor}">
        <div class="al-head">◈ ORAL BIOAVAIL.</div>
        <div class="al-val">${bioav}<span style="font-size:14px;color:var(--muted)">/100</span></div>
        <div class="al-body">Heuristic score combining Ro5, TPSA, rotatable bonds, and Fsp³.</div>
      </div>
    </div>`;

  const desc=Object.entries(j.descriptors).map(([k,v])=>
    `<div class="desc"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`).join('');

  host.innerHTML=`
    <div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
      <div class="grid2">
        <div><div class="label">// Structure Render</div>
          <div class="mol-panel">
            ${j.mol_image?`<img src="data:image/png;base64,${j.mol_image}" alt="molecule"/>`:''}
          </div>
        </div>
        <div><div class="label">// Analysis Report</div>
          <div class="smiles-line">${esc(j.canon_smiles)}</div>
          <div style="margin-top:14px"><span class="${adCls}">${adLabel}</span>
            <span class="file-name" style="margin-left:10px">NN Tanimoto = <b>${j.ad.similarity.toFixed(3)}</b></span>
          </div>
          <details><summary>NEAREST TRAINING NEIGHBOUR</summary>
            <div class="nearest"><b>${esc(j.ad.nn_name)}</b><br/>
              <span style="font-family:var(--mono)">${esc(j.ad.nn_smiles)}</span></div>
          </details>
        </div>
      </div>
      <div class="risk-banner ${risk.tier_cls}">
        <div class="icon">${riskIcon}</div>
        <div><div class="tier">${risk.tier}</div><div class="desc">${risk.desc}</div></div>
      </div>
    </div>

    <div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
      <div class="label">// Multi-Parameter Risk Assessment</div>
      ${alerts}
      <div class="verdict-text">${esc(j.interpretation)}</div>
    </div>

    <div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
      <div class="label">// Isoform Profile — pIC50 Predictions</div>
      <div class="cyps">${cypCards}</div>
      <div class="meta-row" style="margin-top:14px">
        <span>Scale: <b style="color:var(--muted)">2 = weak</b> · <b style="color:var(--amber)">4 = threshold</b> · <b style="color:var(--lime)">6+ = potent</b></span>
        <span>16-model gradient-boosted ensemble · Scaffold-CV validated</span>
      </div>
    </div>

    <div class="card" id="plotsCard"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
      <div class="label">// Molecular Visualization — Interactive Analytics <span style="color:var(--muted);margin-left:10px;text-transform:none;letter-spacing:1px">(hover &amp; click · CYP cards link to bars)</span></div>
      <div class="plot-grid">
        <div class="chart-card wide"><div class="chart-title">⟁ ISOFORM INHIBITION PROFILE — hover for IC50 · components · percentile</div>
          <canvas id="ch-cyp"></canvas>
          <div class="chart-sub">Coloured bands: green=inactive (&lt;4), amber=weak (4–5), magenta=active (5–6), red=potent (≥6). Click a bar or CYP card to highlight.</div>
        </div>
        <div class="chart-card"><div class="chart-title">⟁ DRUG-LIKENESS PROFILE</div>
          <div style="padding:16px 8px">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-family:var(--mono);font-size:11px">
              ${Object.entries(j.descriptors).map(([k,v])=>"<div style=color:"+dlColor(k,v)+"><b style=color:var(--cyan);letter-spacing:1px>"+k.padEnd(11)+"</b><span style=float:right>"+v+"</span></div>").join('')}
            </div>
          </div>
          <div class="chart-sub">Live descriptors — Ro5 violations: <b style=color:${j.lipinski.n===0?'#4ade80':(j.lipinski.n<=1?'#fbbf24':'#f87171')}>${j.lipinski.n}/4</b> · Bioavail: <b style=color:#22d3ee>${j.bioavailability}/100</b> · PAINS: <b style=color:${j.pains.length===0?'#4ade80':'#f87171'}>${j.pains.length===0?'clean':j.pains.length}</b></div>
        </div>
        <div class="chart-card"><div class="chart-title">⟁ CHEMICAL-SPACE SIMILARITY</div>
          <canvas id="ch-sim"></canvas>
          <div class="chart-sub">Tanimoto distribution vs 4,905 training Mols · NN = <b style="color:#22d3ee">${j.ad.similarity.toFixed(3)}</b> · nearest: <b>${esc(j.ad.nn_name)}</b></div>
        </div>
        <div class="chart-card"><div class="chart-title">⟁ CYP1A2 DISTRIBUTION</div><canvas id="ch-a2"></canvas></div>
        <div class="chart-card"><div class="chart-title">⟁ CYP2C9 DISTRIBUTION</div><canvas id="ch-c9"></canvas></div>
        <div class="chart-card"><div class="chart-title">⟁ CYP2D6 DISTRIBUTION</div><canvas id="ch-d6"></canvas></div>
        <div class="chart-card"><div class="chart-title">⟁ CYP3A4 DISTRIBUTION</div><canvas id="ch-a4"></canvas></div>
      </div>
    </div>

    <div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
      <div class="label">// Molecular Descriptors</div>
      <div class="desc-grid">${desc}</div>
    </div>
  `;
}

async function predictBatch(){
  const f=document.getElementById('csvfile').files[0];
  const err=document.getElementById('err');
  if(!f){err.textContent='> BATCH ERROR: select a CSV file with a SMILES column first.';return}
  err.textContent='';
  document.getElementById('spin').style.display='inline-block';
  document.getElementById('btn-batch').disabled=true;
  try{
    const fd=new FormData();fd.append('file',f);
    const r=await fetch('/api/predict_batch',{method:'POST',body:fd});
    document.getElementById('spin').style.display='none';
    document.getElementById('btn-batch').disabled=false;
    if(!r.ok){const t=await r.text();err.textContent='> BATCH FAILED: '+t;return}
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download='CYP_INHIBITION_PREDICTIONS.csv';a.click();
    URL.revokeObjectURL(url);
    showToast('BATCH COMPLETE · CSV DOWNLOADED','#4ade80');
  }catch(e){
    document.getElementById('spin').style.display='none';
    document.getElementById('btn-batch').disabled=false;
    err.textContent='> BATCH FAILED: '+e;
  }
}
document.getElementById('smiles').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))predict()});
</script>
<div class="cursor-glow" id="curGlow"></div>
<div class="cursor-cross" id="curCross"></div>
<script>
// === CURSOR / PARALLAX (BIOWATCH-style) ===
const glow=document.getElementById('curGlow'), cross=document.getElementById('curCross');
let mx=window.innerWidth/2, my=window.innerHeight/2, tx=mx, ty=my;
document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;});
function raf(){
  tx+=(mx-tx)*0.18; ty+=(my-ty)*0.18;
  glow.style.transform=`translate(${tx}px,${ty}px) translate(-50%,-50%)`;
  cross.style.backgroundPosition=`${tx}px 0, 0 ${ty}px`;
  // parallax drift on body::before grid is done via CSS var
  document.body.style.setProperty('--mx',(mx/window.innerWidth-0.5).toFixed(3));
  document.body.style.setProperty('--my',(my/window.innerHeight-0.5).toFixed(3));
  requestAnimationFrame(raf);
}
requestAnimationFrame(raf);
document.addEventListener('mousedown',()=>glow.style.width=glow.style.height='50px');
document.addEventListener('mouseup',  ()=>glow.style.width=glow.style.height='32px');

// === INTERACTIVE CHARTS ===
let _charts=[];
function destroyCharts(){_charts.forEach(c=>c.destroy());_charts=[];}
Chart.defaults.color='#6b88a6';
Chart.defaults.font.family="'JetBrains Mono',monospace";
Chart.defaults.font.size=10;
Chart.defaults.borderColor='rgba(34,211,238,0.12)';
const CYPC={CYP1A2:'#22d3ee',CYP2C9:'#a78bfa',CYP2D6:'#f0abfc',CYP3A4:'#4ade80'};
const CYPL={CYP1A2:'1A2',CYP2C9:'2C9',CYP2D6:'2D6',CYP3A4:'3A4'};

function mkTooltip(neon){return {
  backgroundColor:'rgba(11,19,38,0.92)',borderColor:neon,borderWidth:1,
  titleColor:neon,bodyColor:'#cde8ff',padding:10,
  titleFont:{family:"'Orbitron',sans-serif",size:11,weight:'700'},
  displayColors:false,cornerRadius:8,boxPadding:4,
}}

async function renderCharts(smi){
  destroyCharts();
  const r=await fetch('/api/mol_profile',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({smiles:smi})});
  const d=await r.json();
  if(d.error)return;

  // 1) CYP BAR chart (interactive; click a bar highlights the card, hover shows IC50 + class + components)
  const cyps=['CYP1A2','CYP2C9','CYP2D6','CYP3A4'];
  const vals=cyps.map(c=>d.cyps[c].pic50);
  const ctx1=document.getElementById('ch-cyp');
  const ch=new Chart(ctx1,{
    type:'bar',
    data:{labels:cyps.map(c=>'CYP'+CYPL[c]),
      datasets:[{data:vals,backgroundColor:cyps.map(c=>CYPC[c]+'cc'),borderColor:cyps.map(c=>CYPC[c]),borderWidth:1.5,borderRadius:6,hoverBackgroundColor:cyps.map(c=>CYPC[c])}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:900,easing:'easeOutCubic'},
      plugins:{legend:{display:false},tooltip:{...mkTooltip('#22d3ee'),
        callbacks:{label:(ctx)=>{const c=cyps[ctx.dataIndex];const p=d.cyps[c];
          return [`pIC50 = ${p.pic50.toFixed(2)}`,`IC50 ≈ ${p.ic50_uM} µM`,`Percentile: ${p.percentile}%`,
          ...Object.entries(p.components).map(([k,v])=>k+': '+v.toFixed(2))]}}}},
      scales:{y:{beginAtZero:true,max:8.2,grid:{color:'rgba(34,211,238,0.1)'},ticks:{color:'#6b88a6'},
        title:{display:true,text:'predicted pIC50',color:'#cde8ff'}},
              x:{grid:{display:false},ticks:{color:'#cde8ff',font:{weight:'700',size:12}}}},
      onClick:(e,els)=>{if(!els.length)return;highlightCyp(cyps[els[0].index]);}
    },
    plugins:[{id:'zones',beforeDatasetsDraw(ch){
      const {ctx,chartArea,scales:{y}}=ch;
      const bands=[[0,4,'rgba(74,222,128,.08)',[4,'INACTIVE','#4ade80']],
                   [4,5,'rgba(251,191,36,.10)',[5,'WEAK','#fbbf24']],
                   [5,6,'rgba(240,171,252,.12)',[6,'ACTIVE','#f0abfc']],
                   [6,8.2,'rgba(248,113,113,.14)',null]];
      bands.forEach(([lo,hi,col,lbl])=>{
        ctx.fillStyle=col;
        ctx.fillRect(chartArea.left,y.getPixelForValue(hi),chartArea.right-chartArea.left,y.getPixelForValue(lo)-y.getPixelForValue(hi));
        ctx.strokeStyle=col.replace(/,[\d.]+\)/,',0.6)').replace('rgba','rgb').replace('rgb','rgba').replace(/\)$/,',0.6)');
        ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(chartArea.left,y.getPixelForValue(lo));ctx.lineTo(chartArea.right,y.getPixelForValue(lo));ctx.stroke();ctx.setLineDash([]);
        if(lbl){ctx.fillStyle=lbl[2];ctx.font='10px JetBrains Mono';ctx.textAlign='right';ctx.fillText(lbl[1],chartArea.right-8,y.getPixelForValue(lbl[0])-6);}
      });
    }}]
  });_charts.push(ch);

  // 2) Similarity histogram
  const edges=d.sim_edges, hist=d.sim_hist;
  const labels=edges.slice(0,-1).map((e,i)=>(e+0.01).toFixed(2));
  const ctx2=document.getElementById('ch-sim');
  const ch2=new Chart(ctx2,{type:'bar',
    data:{labels,datasets:[{data:hist,backgroundColor:'rgba(167,139,250,0.7)',borderColor:'#a78bfa',borderWidth:0,borderRadius:2,hoverBackgroundColor:'#f0abfc'}]},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:800},
      plugins:{legend:{display:false},tooltip:{...mkTooltip('#a78bfa'),
        callbacks:{title:(c)=>'Tanimoto '+c[0].label+'–'+(parseFloat(c[0].label)+0.02).toFixed(2),
          label:(c)=>c.parsed.y+' training compounds ('+((c.parsed.y/4905)*100).toFixed(1)+'%)'}}},
      scales:{x:{ticks:{color:'#6b88a6',maxTicksLimit:6,callback:(v)=>labels[v]},grid:{display:false},title:{display:true,text:'Tanimoto to train (Morgan r=2,2048)',color:'#cde8ff'}},
              y:{grid:{color:'rgba(167,139,250,0.1)'},ticks:{color:'#6b88a6'},title:{display:true,text:'count',color:'#cde8ff'}}}},
    plugins:[{id:'ann',afterDatasetsDraw(ch){const {ctx,scales:{x,y},chartArea}=ch;
      [['0.3','#f87171','OOD'],['0.5','#fbbf24','LOW']].forEach(([v,c])=>{
        const px=x.getPixelForValue(parseFloat(v)*50);
        ctx.save();ctx.strokeStyle=c;ctx.setLineDash([4,3]);ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(px,chartArea.top);ctx.lineTo(px,chartArea.bottom);ctx.stroke();ctx.restore();
      });
      // NN line
      const bin=Math.min(49,Math.floor(d.nn_similarity*50));
      const nnpx=x.getPixelForValue(bin)+(x.getPixelForValue(1)-x.getPixelForValue(0))/2;
      ctx.save();ctx.strokeStyle='#22d3ee';ctx.lineWidth=2.5;ctx.shadowColor='#22d3ee';ctx.shadowBlur=12;ctx.beginPath();ctx.moveTo(nnpx,chartArea.top);ctx.lineTo(nnpx,chartArea.bottom);ctx.stroke();ctx.restore();
      ctx.fillStyle='#22d3ee';ctx.font='bold 10px JetBrains Mono';ctx.textAlign='left';
      ctx.fillText('NN = '+d.nn_similarity.toFixed(2)+'   ≥0.5: '+d.n_analogs_ge_05+'   ≥0.3: '+d.n_analogs_ge_03,nnpx+6,chartArea.top+14);
    }}]
  });_charts.push(ch2);

  // 3) 2x2 training distribution charts
  const pos={'CYP1A2':'ch-a2','CYP2C9':'ch-c9','CYP2D6':'ch-d6','CYP3A4':'ch-a4'};
  cyps.forEach(c=>{
    const p=d.cyps[c], edges=p.edges, hist=p.hist;
    const labs=edges.slice(0,-1).map((e,i)=>((e+edges[i+1])/2).toFixed(1));
    const ctx=document.getElementById(pos[c]);
    const chh=new Chart(ctx,{type:'bar',
      data:{labels:labs,datasets:[{data:hist,backgroundColor:'rgba(42,64,112,0.85)',borderColor:'#2a4070',borderRadius:1,hoverBackgroundColor:CYPC[c]}]},
      options:{responsive:true,maintainAspectRatio:false,animation:{duration:900,delay:150},
        plugins:{legend:{display:false},tooltip:{...mkTooltip(CYPC[c]),callbacks:{label:(ctx)=>ctx.parsed.y+' train molecules at pIC50≈'+ctx.label}}},
        scales:{x:{ticks:{color:'#6b88a6',maxTicksLimit:5},grid:{display:false},title:{display:true,text:'pIC50',color:'#cde8ff'}},
                y:{grid:{color:'rgba(34,211,238,0.08)'},ticks:{color:'#6b88a6',maxTicksLimit:4}}}},
      plugins:[{id:'vline',afterDatasetsDraw(ch){const {ctx,scales:{x,y},chartArea}=ch;
        // find bin index closest to prediction
        let bi=0,bd=99;
        edges.forEach((e,i)=>{const m=(e+(edges[i+1]||e))/2;const dd=Math.abs(m-p.pic50);if(dd<bd){bd=dd;bi=i}});
        const px=x.getPixelForValue(bi)+(x.getPixelForValue(1)-x.getPixelForValue(0))/2;
        ctx.save();ctx.strokeStyle=CYPC[c];ctx.lineWidth=2.5;ctx.shadowColor=CYPC[c];ctx.shadowBlur=10;ctx.beginPath();ctx.moveTo(px,chartArea.top);ctx.lineTo(px,chartArea.bottom);ctx.stroke();ctx.restore();
        ctx.fillStyle=CYPC[c];ctx.font='bold 10px JetBrains Mono';ctx.textAlign='left';
        ctx.fillText(p.pic50.toFixed(2)+' ('+p.percentile.toFixed(0)+'th %ile)',px+4,chartArea.top+14);
      }}]
    });_charts.push(chh);
  });

  // Click a CYP card -> highlight its chart
  document.querySelectorAll('.cyp').forEach(el=>{
    el.style.cursor='pointer';
    el.onclick=()=>{
      const c=Array.from(el.classList).find(cl=>cl.startsWith('cyp')&&cl!=='cyps');
      const map={'cyp1a2':'CYP1A2','cyp2c9':'CYP2C9','cyp2d6':'CYP2D6','cyp3a4':'CYP3A4'};
      highlightCyp(map[c]);
    };
  });
}
function highlightCyp(cyp){
  if(!cyp)return;
  document.querySelectorAll('.cyp').forEach(el=>el.classList.remove('active-cyp'));
  document.querySelector('.'+cyp.toLowerCase()).classList.add('active-cyp');
  const card=document.getElementById('plotsCard');
  card.scrollIntoView({behavior:'smooth',block:'start'});
  // Flash the chart
  const bar=_charts[0];
  const idx=['CYP1A2','CYP2C9','CYP2D6','CYP3A4'].indexOf(cyp);
  if(bar && idx>=0){bar.setActiveElements([{datasetIndex:0,index:idx}]);bar.tooltip.setActiveElements([{datasetIndex:0,index:idx}],{x:0,y:0});bar.update();}
}
</script>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


DIAG_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CYP PROFILER // Model Diagnostics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg-0:#04060d;--cyan:#22d3ee;--magenta:#f0abfc;--purple:#a78bfa;--lime:#4ade80;
 --amber:#fbbf24;--red:#f87171;--text:#e2f3ff;--muted:#6f8aa8;--panel:rgba(12,18,38,.78);
 --mono:'JetBrains Mono',ui-monospace,monospace;--display:'Orbitron','Inter',sans-serif;--sans:'Inter',system-ui,sans-serif;}
*{box-sizing:border-box}html,body{margin:0;padding:0}
body{min-height:100vh;color:var(--text);font-family:var(--sans);
  background:radial-gradient(1200px 800px at 12% -10%,rgba(167,139,250,.18),transparent 60%),
    radial-gradient(1000px 700px at 100% 0%,rgba(34,211,238,.16),transparent 60%),
    radial-gradient(900px 600px at 50% 110%,rgba(240,171,252,.12),transparent 60%),
    linear-gradient(180deg,#04060d 0%,#070b18 50%,#04060d 100%);position:relative;overflow-x:hidden;}
body::before{content:"";position:fixed;inset:0;
  background-image:linear-gradient(rgba(34,211,238,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.06) 1px,transparent 1px);
  background-size:48px 48px;mask-image:radial-gradient(ellipse at center,rgba(0,0,0,.9),transparent 75%);
  -webkit-mask-image:radial-gradient(ellipse at center,rgba(0,0,0,.9),transparent 75%);pointer-events:none;z-index:0;
  animation:drift 40s linear infinite;}
@keyframes drift{from{background-position:0 0,0 0}to{background-position:48px 48px,48px 48px}}
body::after{content:"";position:fixed;left:0;right:0;top:0;height:140px;
  background:linear-gradient(180deg,rgba(34,211,238,.07),transparent);pointer-events:none;z-index:1;
  animation:scan 6s linear infinite;mix-blend-mode:screen;}
@keyframes scan{0%{transform:translateY(-140px)}100%{transform:translateY(100vh)}}
.wrap{max-width:1280px;margin:0 auto;padding:28px 24px 64px;position:relative;z-index:2}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px}
.brand{display:flex;align-items:center;gap:14px}
.logo{width:54px;height:54px;border-radius:13px;position:relative;
  background:conic-gradient(from 210deg,#22d3ee,#a78bfa,#f0abfc,#22d3ee);
  box-shadow:0 0 30px rgba(34,211,238,.45),inset 0 0 18px rgba(0,0,0,.6);}
.logo::before{content:"";position:absolute;inset:4px;border-radius:10px;background:radial-gradient(circle at 30% 30%,rgba(255,255,255,.2),rgba(6,10,22,.9));}
.logo::after{content:"CYP";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-family:var(--display);font-weight:900;font-size:15px;letter-spacing:1px;color:#eaffff;text-shadow:0 0 12px rgba(34,211,238,.8);}
h1{font-family:var(--display);font-weight:900;font-size:24px;margin:0;letter-spacing:2px;
  background:linear-gradient(90deg,#22d3ee 0%,#a78bfa 50%,#f0abfc 100%);-webkit-background-clip:text;background-clip:text;color:transparent;text-shadow:0 0 28px rgba(34,211,238,.25);}
h1 .sub{display:block;font-family:var(--mono);font-weight:500;font-size:11px;letter-spacing:3px;color:var(--muted);margin-top:4px;text-shadow:none;background:none;-webkit-text-fill-color:var(--muted);}
.hud{display:flex;gap:10px;align-items:center;font-family:var(--mono);font-size:11px;color:var(--muted);flex-wrap:wrap}
.hud .dot{width:8px;height:8px;border-radius:50%;background:var(--lime);box-shadow:0 0 12px var(--lime);animation:pulse 2s infinite;}
@keyframes pulse{50%{opacity:.35;box-shadow:0 0 2px var(--lime)}}
.nav-btn{margin-left:14px;padding:6px 12px;border:1px solid var(--cyan);border-radius:8px;color:var(--cyan);text-decoration:none;
  font-family:var(--mono);font-size:10px;letter-spacing:1.5px;background:rgba(34,211,238,.06);transition:all .2s;}
.nav-btn:hover{background:rgba(34,211,238,.18);box-shadow:0 0 16px rgba(34,211,238,.4);color:#eaffff}
.card{background:var(--panel);border:1px solid rgba(56,189,248,.22);border-radius:18px;padding:22px;position:relative;
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  box-shadow:0 0 0 1px rgba(255,255,255,.03) inset,0 20px 60px rgba(0,0,0,.5),0 0 40px rgba(34,211,238,.05);overflow:hidden;}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(34,211,238,.6),transparent);}
.card+.card{margin-top:20px}
.corner{position:absolute;width:14px;height:14px;border:2px solid var(--cyan);opacity:.7}
.corner.tl{top:8px;left:8px;border-right:none;border-bottom:none}.corner.tr{top:8px;right:8px;border-left:none;border-bottom:none}
.corner.bl{bottom:8px;left:8px;border-right:none;border-top:none}.corner.br{bottom:8px;right:8px;border-left:none;border-top:none}
.label{font-family:var(--mono);font-size:11px;letter-spacing:2px;color:var(--cyan);text-transform:uppercase;margin-bottom:10px;display:flex;gap:10px;align-items:center}
.label::before{content:"";width:20px;height:1px;background:var(--cyan);box-shadow:0 0 8px var(--cyan)}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
@media(max-width:800px){.metric-grid{grid-template-columns:repeat(2,1fr)}}
.metric{border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px;
  background:linear-gradient(160deg,rgba(12,18,38,.8),rgba(7,11,24,.6));position:relative;overflow:hidden}
.metric::after{content:"";position:absolute;top:0;left:0;right:0;height:2px}
.metric.c1::after{background:#22d3ee;box-shadow:0 0 12px #22d3ee}
.metric.c2::after{background:#a78bfa;box-shadow:0 0 12px #a78bfa}
.metric.c3::after{background:#f0abfc;box-shadow:0 0 12px #f0abfc}
.metric.c4::after{background:#4ade80;box-shadow:0 0 12px #4ade80}
.metric h3{margin:0;font-family:var(--display);font-size:12px;letter-spacing:3px;color:var(--muted);font-weight:700}
.metric.c1 h3{color:#22d3ee}.metric.c2 h3{color:#a78bfa}.metric.c3 h3{color:#f0abfc}.metric.c4 h3{color:#4ade80}
.metric .v{font-family:var(--display);font-weight:900;font-size:30px;margin-top:6px;letter-spacing:1px}
.metric .sub{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:4px;letter-spacing:1px}
figure{margin:0;padding:0;text-align:center}
figure img{max-width:100%;border-radius:12px;border:1px solid rgba(34,211,238,.15);
  box-shadow:0 10px 40px rgba(0,0,0,.5);}
figcaption{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:10px;letter-spacing:1px;line-height:1.6}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px;margin-top:10px}
th,td{padding:10px 14px;text-align:center;border-bottom:1px solid rgba(34,211,238,.1)}
th{color:var(--cyan);font-weight:700;letter-spacing:1.5px;text-transform:uppercase;font-size:10px;
  border-bottom:1px solid rgba(34,211,238,.3);background:rgba(34,211,238,.05)}
td{color:var(--text)}
td.model{text-align:left;color:var(--purple);letter-spacing:1px}
.best{color:var(--lime);font-weight:700;text-shadow:0 0 8px rgba(74,222,128,.5)}
.bad{color:var(--red)}
.weights-bar{height:8px;border-radius:6px;background:rgba(255,255,255,.06);overflow:hidden;display:flex;margin-top:6px}
.weights-bar span{height:100%}
.w-xgb{background:#f0abfc}.w-lgbm{background:#22d3ee}.w-hgb{background:#a78bfa}.w-fp{background:#fbbf24}
.note{font-size:12px;color:var(--muted);line-height:1.75;border-left:2px solid var(--cyan);padding:10px 14px;
  background:linear-gradient(90deg,rgba(34,211,238,.06),transparent);border-radius:0 10px 10px 0;margin-top:12px;font-family:var(--sans)}
.footer{font-family:var(--mono);font-size:10px;color:var(--muted);text-align:center;margin-top:28px;letter-spacing:2px;opacity:.6}
.footer span{color:var(--cyan)}
a.back{color:var(--cyan);text-decoration:none;font-family:var(--mono);font-size:11px;letter-spacing:1.5px}
a.back:hover{text-shadow:0 0 8px var(--cyan)}
</style></head><body>
<div class="wrap">
<header>
  <div class="brand"><div class="logo"></div>
    <div><h1>CYP INHIBITION PROFILER<span class="sub">MODEL DIAGNOSTICS · VALIDATION DASHBOARD</span></h1></div>
  </div>
  <div class="hud"><span class="dot"></span><span>5-FOLD SCAFFOLD CV · 16 MODELS · 750 BLIND COMPOUNDS</span>
    <a class="nav-btn" href="/">&#9664; BACK TO PROFILER</a>
  </div>
</header>

<div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Ensemble Performance — 5-fold Scaffold CV (ST-RAE ↓)</div>
  <div class="metric-grid">
    <div class="metric c1"><h3>CYP1A2</h3><div class="v">{{ m['CYP1A2'] }}</div><div class="sub">ST-RAE · R²={{ r2['CYP1A2'] }} · ρ={{ rho['CYP1A2'] }}</div></div>
    <div class="metric c2"><h3>CYP2C9</h3><div class="v">{{ m['CYP2C9'] }}</div><div class="sub">ST-RAE · R²={{ r2['CYP2C9'] }} · ρ={{ rho['CYP2C9'] }}</div></div>
    <div class="metric c3"><h3>CYP2D6</h3><div class="v">{{ m['CYP2D6'] }}</div><div class="sub">ST-RAE · R²={{ r2['CYP2D6'] }} · ρ={{ rho['CYP2D6'] }}</div></div>
    <div class="metric c4"><h3>CYP3A4</h3><div class="v">{{ m['CYP3A4'] }}</div><div class="sub">ST-RAE · R²={{ r2['CYP3A4'] }} · ρ={{ rho['CYP3A4'] }}</div></div>
  </div>
  <div class="note"><b>Metric:</b> Macro-averaged Soft-Threshold RAE (MA-ST-RAE). Lower is better; 1.0 = predicting the training mean.
  Scaffold split ensures no test molecule shares a Bemis-Murcko scaffold with its training fold,
  giving a more honest estimate of prospective performance than a random split.
  CYP3A4 is the strongest endpoint; CYP2D6 is the hardest (lowest inter-lab consensus + noisy singletons).</div>
</div>

<div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Model Comparison — ST-RAE per isoform</div>
  <figure><img src="data:image/png;base64,{{ figs['perf'] }}" alt="model performance"/>
    <figcaption>XGB on fingerprints + descriptors dominates every isoform; the non-negative ST-RAE-optimal blend (green) adds a small consistent lift over any single model. Red dashed line = trivial mean-predictor.</figcaption>
  </figure>
</div>

<div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Out-of-Fold Predicted vs Experimental</div>
  <figure><img src="data:image/png;base64,{{ figs['oof'] }}" alt="OOF pred vs true"/>
    <figcaption>Per-CYP scatter of ensemble OOF predictions (5-fold scaffold CV) against experimental pIC50. Red line = identity. Classic regression-to-mean visible at the extremes — expected for noisy biochemical data and mitigated by ST-RAE loss weighting.</figcaption>
  </figure>
</div>

<div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Train vs Blind-Test pIC50 Distributions</div>
  <figure><img src="data:image/png;base64,{{ figs['dist'] }}" alt="distributions"/>
    <figcaption>Predicted distributions for the 750 blind-test compounds overlaid on the training set. All four CYPs stay well within the training support — no pathological extrapolation.</figcaption>
  </figure>
</div>

<div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Applicability Domain — Tanimoto to Nearest Training Neighbour</div>
  <figure><img src="data:image/png;base64,{{ figs['ad'] }}" alt="applicability domain"/>
    <figcaption>Morgan chiral radius-2 / 2048-bit fingerprint. Median similarity ≈0.58; only 1/750 test compounds below the 0.3 OOD cutoff. Predictions outside 0.3–0.5 band carry the LOW CONFIDENCE pill on the main page.</figcaption>
  </figure>
</div>

<div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Ensemble Blend Weights (non-negative, ST-RAE-optimal)</div>
  <table>
    <thead><tr><th style="text-align:left">Component</th><th>CYP1A2</th><th>CYP2C9</th><th>CYP2D6</th><th>CYP3A4</th></tr></thead>
    <tbody>{% for mname,short,cls in [('XGBoost · FP+Descr','xgb_both','w-xgb'),('LightGBM · FP+Descr','lgbm_both','w-lgbm'),('Hist-Grad-Boost · Descriptors','hgb_descr','w-hgb'),('LightGBM · FP only','lgbm_fp','w-fp')] %}
      <tr><td class="model">{{ mname }}<div class="weights-bar">
        {% for cyp in ['CYP1A2','CYP2C9','CYP2D6','CYP3A4'] %}<span class="{{ cls }}" style="width:{{ (weights[cyp][short]*100)|round(1) }}%"></span>{% endfor %}
      </div></td>
      {% for cyp in ['CYP1A2','CYP2C9','CYP2D6','CYP3A4'] %}<td>{{ "%.2f"|format(weights[cyp][short]) }}</td>{% endfor %}
      </tr>{% endfor %}
    </tbody>
  </table>
  <div class="note">Weights fit independently per CYP via non-negative least-squares against OOF predictions, constrained to sum to 1.
  XGBoost on combined fingerprints + descriptors is the dominant contributor everywhere; descriptor-only HGB and FP-only LGBM add orthogonal signal.</div>
</div>

<div class="card"><span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
  <div class="label">// Pipeline Summary</div>
  <div class="note" style="font-family:var(--mono);color:var(--text);font-size:12px;line-height:2.2">
    <span style="color:var(--cyan)">FEATURES</span> · Morgan chiral r=2 / 2048-bit FP + 202 RDKit 2D descriptors · z-scored, median-imputed<br/>
    <span style="color:var(--purple)">BASE MODELS</span> · XGBoost, LightGBM, Histogram-Gradient-Boost (sklearn) on (FP+descr), (FP), (descr) → 4 per CYP = 16<br/>
    <span style="color:var(--magenta)">SPLITTING</span> · 5-fold GroupKFold on Bemis-Murcko scaffolds (no scaffold leakage across folds)<br/>
    <span style="color:var(--lime)">ENSEMBLE</span> · Per-CYP non-negative NNLS blend optimized on out-of-fold ST-RAE<br/>
    <span style="color:var(--amber)">CALIBRATION</span> · Predictions clipped to training per-CYP [0.5%, 99.5%] interval to suppress runaway extrapolation<br/>
    <span style="color:var(--red)">AD</span> · Tanimoto (Morgan FP) to nearest training neighbour; flag 0 / 1 / 2 with thresholds 0.5 / 0.3<br/>
    <span style="color:var(--cyan)">TARGET</span> · pIC50 = −log₁₀(IC₅0/M) for direct inhibition (1A2, 2C9, 2D6, 3A4); no TDI / Emax / single-conc contamination<br/>
    <span style="color:var(--purple)">SEED</span> · Fixed seeds (42) throughout; artifacts cached in <code>models/pipeline_artifacts.pkl</code>
  </div>
</div>

<div class="footer"><span>[</span> CYP PROFILER <span>]</span> · Research Use · Scaffold-CV validated · 4,905 training curves</div>
</div></body></html>"""


@app.route("/diagnostics")
def diagnostics():
    figs_in = _neon_plots()

    # OOF scatter: rebuild from saved figures (light PNG) OR regenerate dark on the fly.
    # Use the existing light PNG as base for speed.  We'll also regenerate a dark OOF scatter now
    # because the cached one is light-themed and would clash with the UI.
    oof_b64 = _build_dark_oof_scatter()

    ens_strae = {"CYP1A2":0.903,"CYP2C9":0.772,"CYP2D6":0.995,"CYP3A4":0.546}
    ens_r2    = {"CYP1A2":0.26,"CYP2C9":0.37,"CYP2D6":0.15,"CYP3A4":0.59}
    ens_rho   = {"CYP1A2":0.52,"CYP2C9":0.59,"CYP2D6":0.39,"CYP3A4":0.77}

    return render_template_string(DIAG_HTML,
        figs={"perf":figs_in["perf"],"dist":figs_in["dist"],"ad":figs_in["ad"],"oof":oof_b64},
        m={k:f"{v:.3f}" for k,v in ens_strae.items()},
        r2=ens_r2, rho=ens_rho, weights=_WEIGHTS)


# ---------- Per-molecule plots ----------
def _mol_plot_cyp_bars(cyp_preds, out=None):
    """Bar chart of predicted pIC50 per CYP, with liability zones shaded."""
    fig, ax = plt.subplots(figsize=(7.2,3.6), dpi=140)
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL_BG)
    cyps = C.CYP_ISOFORMS
    colors = [NEON_CYAN, NEON_VIO, NEON_PINK, NEON_GRN]
    vals = [cyp_preds[c]["pic50"] for c in cyps]
    # background bands
    ax.axhspan(0, 4, color=NEON_GRN, alpha=0.06)
    ax.axhspan(4, 5, color=NEON_AMB, alpha=0.08)
    ax.axhspan(5, 6, color=NEON_PINK, alpha=0.09)
    ax.axhspan(6, 8.5, color=NEON_RED, alpha=0.10)
    ax.axhline(4, color=NEON_AMB, lw=0.8, ls=":", alpha=0.6)
    ax.axhline(5, color=NEON_PINK, lw=0.8, ls=":", alpha=0.6)
    ax.axhline(6, color=NEON_RED, lw=0.8, ls=":", alpha=0.7)
    bars = ax.bar(range(len(cyps)), vals, color=colors, edgecolor="#0b1326", linewidth=1.2, width=0.58)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.08, f"{v:.2f}", ha="center",
                color=TEXT_CLR, fontsize=11, fontweight="bold", family="DejaVu Sans Mono")
    ax.set_xticks(range(len(cyps)))
    ax.set_xticklabels(["CYP1A2","CYP2C9","CYP2D6","CYP3A4"], fontsize=10, fontweight="bold", color=TEXT_CLR)
    ax.tick_params(axis='y', colors=MUTED)
    ax.set_ylabel("predicted pIC₅₀", color=TEXT_CLR)
    ax.set_ylim(0, 8.2)
    ax.set_xlim(-0.6, len(cyps)-0.4)
    ax.set_title("ISOFORM INHIBITION PROFILE", color=NEON_CYAN, fontweight="bold", fontsize=12, pad=10)
    for spine in ax.spines.values(): spine.set_color(MUTED)
    # zone labels (inside the plot area)
    ax.text(len(cyps)-1-0.32, 2.0, "INACTIVE",  color=NEON_GRN,  fontsize=8, alpha=0.8, ha="right", family="DejaVu Sans Mono")
    ax.text(len(cyps)-1-0.32, 4.5, "WEAK",      color=NEON_AMB,  fontsize=8, alpha=0.8, ha="right", family="DejaVu Sans Mono")
    ax.text(len(cyps)-1-0.32, 5.5, "ACTIVE",    color=NEON_PINK, fontsize=8, alpha=0.8, ha="right", family="DejaVu Sans Mono")
    ax.text(len(cyps)-1-0.32, 7.3, "POTENT",    color=NEON_RED,  fontsize=8, alpha=0.9, ha="right", family="DejaVu Sans Mono")
    fig.tight_layout()
    return _save_b64(fig, out)


def _mol_plot_radar(desc, out=None):
    """Drug-likeness radar: normalise each descriptor to a 0-1 'ideal oral drug' window."""
    # (label, value, ideal_min, ideal_max, display_max)
    axes_spec = [
        ("MW",        desc["MW"],          150, 450, 700),
        ("LogP",      desc["LogP"],        -0.5, 4.5, 7.0),
        ("TPSA",      desc["TPSA"],        20,  120, 200),
        ("HBD",       desc["HBD"],         0,   3,   6),
        ("HBA",       desc["HBA"],         1,   7,   12),
        ("RotBonds",  desc["RotBonds"],    0,   6,   12),
        ("AromRings", desc["AromaticRings"],0,  2,   5),
        ("Fsp³",      desc["FracCSP3"],    0.3, 0.8, 1.0),
    ]
    n = len(axes_spec)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    def score(v, lo, hi, dmax):
        if lo <= v <= hi: return 1.0
        if v < lo: return max(0.0, v/lo) if lo>0 else max(0.0, 1.0 - (lo-v)/dmax)
        return max(0.0, 1.0 - (v-hi)/(dmax-hi))
    vals = [score(v,lo,hi,dmax) for _,v,lo,hi,dmax in axes_spec]
    vals += vals[:1]

    fig = plt.figure(figsize=(5.2,5.0), dpi=140)
    fig.patch.set_facecolor(DARK_BG)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(PANEL_BG)
    ideal = [1.0]*n + [1.0]
    ax.fill(angles, ideal, color=NEON_GRN, alpha=0.08)
    ax.plot(angles, ideal, color=NEON_GRN, lw=0.8, ls="--", alpha=0.5)
    ax.fill(angles, vals, color=NEON_CYAN, alpha=0.22)
    ax.plot(angles, vals, color=NEON_CYAN, lw=2.2)
    ax.scatter(angles[:-1], vals[:-1], s=40, color=NEON_CYAN, edgecolors="#fff", zorder=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a[0] for a in axes_spec], color=TEXT_CLR, fontsize=9,
                       family="DejaVu Sans Mono", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["","0.5","","1.0"], color=MUTED, fontsize=7)
    ax.grid(color=GRID_CLR, lw=0.7)
    ax.spines["polar"].set_color(MUTED)
    ax.set_title("DRUG-LIKENESS FINGERPRINT", color=NEON_CYAN, fontweight="bold", fontsize=12, pad=18)
    fig.tight_layout()
    return _save_b64(fig, out)


def _mol_plot_similarity_hist(mol, out=None):
    """Histogram of Tanimoto similarity from this mol to all TRAIN_MOLS; mark NN and AD thresholds."""
    from rdkit import DataStructs
    fp = _MORGAN_GEN.GetFingerprint(mol)
    sims_all = np.array([DataStructs.TanimotoSimilarity(fp, tfp) for tfp in TRAIN_FPS], dtype=np.float64)
    nn = float(sims_all.max())
    # how many training compounds are *less similar* than the nearest? 100% by definition,
    # so instead report how many training compounds have similarity >= 0.5 / 0.3
    n_ge_05 = int((sims_all >= 0.5).sum())
    n_ge_03 = int((sims_all >= 0.3).sum())
    n_ge_nn = int((sims_all >= nn - 1e-9).sum())  # ties at NN

    fig, ax = plt.subplots(figsize=(6.4,3.6), dpi=140)
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values(): spine.set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.hist(sims_all, bins=50, color=NEON_VIO, alpha=0.75, edgecolor="#0e1a2f", label=f"all train ({len(TRAIN_FPS):,})")
    ax.axvline(0.3, color=NEON_RED, lw=1.5, ls="--", label="OOD (0.3)")
    ax.axvline(0.5, color=NEON_AMB, lw=1.5, ls="--", label="low-conf (0.5)")
    ax.axvline(nn,  color=NEON_CYAN, lw=2.4, ls="-", label=f"nearest = {nn:.2f}")
    ax.set_xlabel("Tanimoto similarity to training set (Morgan r=2, 2048)", color=TEXT_CLR)
    ax.set_ylabel("count", color=TEXT_CLR)
    ax.set_title("CHEMICAL-SPACE SIMILARITY PROFILE", color=NEON_CYAN, fontweight="bold", fontsize=12, pad=10)
    ax.legend(facecolor=PANEL_BG, edgecolor=MUTED, labelcolor=TEXT_CLR, fontsize=8, loc="upper right")
    ax.set_xlim(0,1.05)
    # Annotation near the NN line
    note = (f"NN Tanimoto = {nn:.2f}\n"
            f"≥0.5: {n_ge_05} analogs\n"
            f"≥0.3: {n_ge_03} in-domain")
    ax.text(nn+0.02, ax.get_ylim()[1]*0.92, note,
            color=NEON_CYAN, fontsize=8, family="DejaVu Sans Mono", va="top",
            bbox=dict(facecolor=PANEL_BG, edgecolor=NEON_CYAN, boxstyle="round,pad=0.35", alpha=0.92))
    fig.tight_layout()
    return _save_b64(fig, out)


def _mol_plot_pic50_vs_train(cyp_preds, out=None):
    """Where does this molecule's pIC50 fall in each CYP's training distribution?"""
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.0),dpi=140)
    fig.patch.set_facecolor(DARK_BG)
    colors = [NEON_CYAN, NEON_VIO, NEON_PINK, NEON_GRN]
    for ax,cyp,col in zip(axes.ravel(), C.CYP_ISOFORMS, colors):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values(): spine.set_color(MUTED)
        ax.tick_params(colors=MUTED)
        y = TRAIN_PIC50[cyp]
        ax.hist(y, bins=32, color="#2a4070", alpha=0.75, edgecolor="#1a2a4a", label="train")
        v = cyp_preds[cyp]["pic50"]
        ax.axvline(v, color=col, lw=2.5)
        ax.axvspan(v-0.05, v+0.05, color=col, alpha=0.3)
        pct = float((y <= v).mean())*100
        ax.text(v, ax.get_ylim()[1]*0.92, f"  {v:.2f}\n  {pct:.0f}th %ile", color=col,
                fontsize=9, fontweight="bold", va="top", family="DejaVu Sans Mono")
        ax.set_title(cyp, color=col, fontweight="bold", fontsize=11, pad=6)
        ax.set_xlabel("pIC50", color=TEXT_CLR); ax.set_ylabel("count", color=TEXT_CLR)
    fig.suptitle("PREDICTION vs TRAINING DISTRIBUTION", color=NEON_CYAN,
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95])
    return _save_b64(fig, out)


def _save_b64(fig, out=None):
    if out is None: out = io.BytesIO()
    fig.savefig(out, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    if isinstance(out, io.BytesIO):
        out.seek(0); return base64.b64encode(out.read()).decode()
    return None


def build_molecule_plots(mol, cyp_preds, desc):
    """Return dict of four base64 PNG figures for the analysed molecule."""
    # Apply dark theme rcParams (idempotent)
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
        "savefig.facecolor": DARK_BG, "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT_CLR, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT_CLR, "axes.titlecolor": NEON_CYAN,
        "font.family": "DejaVu Sans Mono", "font.size": 9,
        "axes.grid": True, "grid.color": GRID_CLR, "grid.linestyle": "--", "grid.alpha":0.7,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return {
        "cyp_bars":    _mol_plot_cyp_bars(cyp_preds),
        "radar":       _mol_plot_radar(desc),
        "similarity":  _mol_plot_similarity_hist(mol),
        "distribution":_mol_plot_pic50_vs_train(cyp_preds),
    }


def _build_dark_oof_scatter():
    """Dark-themed OOF pred-vs-true scatter — color-tint the existing light PNG via PIL."""
    try:
        from PIL import Image as PILImage, ImageEnhance
        src = os.path.join(APP_DIR,"figures","oof_pred_vs_true.png")
        img = PILImage.open(src).convert("RGBA")
        # Darken: replace near-white pixels with dark theme colour, blues become cyan
        import numpy as np
        arr = np.array(img).astype(np.float32)
        r,g,b,a = arr[...,0],arr[...,1],arr[...,2],arr[...,3]
        # background pixels (high luminance) -> dark
        lum = (0.299*r+0.587*g+0.114*b)/255.0
        is_bg = lum > 0.92
        is_grid = (lum > 0.7) & (np.abs(r-g) < 15) & (np.abs(g-b) < 15) & ~is_bg
        is_blue = (b > r+15) & (b > g-5) & ~is_bg & ~is_grid
        is_red = (r > 150) & (g < 100) & (b < 100) & ~is_bg & ~is_grid
        is_text = (lum < 0.2) & ~is_bg & ~is_grid & ~is_blue & ~is_red
        # remap
        arr[is_bg] = [6,10,22,255]
        arr[is_grid] = [34*0.4,211*0.4,238*0.4,255]  # dim cyan grid
        # blue dots -> neon cyan
        factor = 0.7 + 0.3*(lum[is_blue,None])
        arr[is_blue] = np.clip(np.array([34,211,238,255])*factor,0,255)
        # red diagonal -> neon magenta
        arr[is_red] = [240,171,252,255]
        # text/axes -> muted cyan
        arr[is_text] = [205,232,255,255]
        out = PILImage.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO(); out.save(buf,format="PNG"); buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        # Fallback: just embed the original
        try:
            with open(os.path.join(APP_DIR,"figures","oof_pred_vs_true.png"),"rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return ""


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    smi = (data or {}).get("smiles","")
    mol,fp,desc,canon,ok,err = featurize_one(smi)
    if not ok: return jsonify({"error":err})
    cyp_preds = predict_pic50(mol,fp,desc)
    flag,sim,nn_smi,nn_name = ad_flag(mol)
    lip_n, lip_v = lipinski_check(mol)
    pains = pains_hits(mol)
    bioav = bioavailability_score(mol, lip_n)
    verdicts = {}
    for cyp in C.CYP_ISOFORMS:
        label,cls,desc_t = liability_verdict(cyp_preds[cyp]["pic50"])
        verdicts[cyp] = {"label":label,"cls":cls,"desc":desc_t}
    risk_tier, risk_cls, risk_desc = ddi_risk_tier(cyp_preds)

    # Plain-English interpretation
    potent = [c for c in C.CYP_ISOFORMS if cyp_preds[c]["pic50"]>=6]
    active = [c for c in C.CYP_ISOFORMS if 5<=cyp_preds[c]["pic50"]<6]
    weak   = [c for c in C.CYP_ISOFORMS if 4<=cyp_preds[c]["pic50"]<5]
    parts = []
    if potent:
        parts.append(f"Predicted as a <b>potent inhibitor</b> of {', '.join(potent)} (pIC50 ≥ 6), suggesting strong clinical DDI liability via those isoforms.")
    if active:
        parts.append(f"Moderate activity predicted for {', '.join(active)} (pIC50 5–6), warranting monitoring.")
    if not potent and not active:
        parts.append("No strong CYP inhibition predicted across the four major isoforms (all pIC50 < 5), suggesting a low immediate DDI risk profile.")
    if lip_n > 1:
        parts.append(f"Note {lip_n} Lipinski rule-of-five violation(s), which may limit oral bioavailability.")
    elif lip_n == 0:
        parts.append("Lipinski Ro5 profile is clean, consistent with good oral drug-likeness.")
    if pains:
        parts.append(f"⚠ <b>PAINS warning:</b> {len(pains)} pan-assay interference motif(s) detected — biochemical assay hits may be promiscuous.")
    if bioav < 40:
        parts.append("Oral bioavailability proxy score is low; consider structural optimisation of polarity and flexibility.")
    if flag == 2:
        parts.append("The molecule sits outside the well-represented region of training chemistry — treat quantitative predictions as low-confidence.")
    interpretation = " ".join(parts)
    desc_map = descriptors(mol)
    plots = build_molecule_plots(mol, cyp_preds, desc_map)

    return jsonify({
        "canon_smiles":canon, "mol_image":mol_to_png_b64(mol),
        "cyp":cyp_preds, "verdicts":verdicts,
        "descriptors":desc_map,
        "lipinski":{"n":lip_n,"violations":lip_v},
        "pains":pains, "bioavailability":bioav,
        "risk":{"tier":risk_tier,"tier_cls":risk_cls,"desc":risk_desc},
        "interpretation":interpretation,
        "ad":{"flag":flag,"similarity":sim,"nn_smi":nn_smi,"nn_name":nn_name},
        "plots":plots,
    })


@app.route("/api/mol_profile", methods=["POST"])
def api_mol_profile():
    """Return rich per-molecule data for interactive charts:
       - similarity histogram (50 bins to 1.0)
       - training pIC50 distributions per CYP (32 bins)
       - ensemble component breakdown per CYP
       - percentile of prediction in each training CYP
    """
    from rdkit import DataStructs
    data = request.get_json(force=True)
    smi = (data or {}).get("smiles","")
    mol,fp,desc,canon,ok,err = featurize_one(smi)
    if not ok: return jsonify({"error":err})
    cyp_preds = predict_pic50(mol,fp,desc)

    # Similarity histogram vs all training FPs
    fp_mol = _MORGAN_GEN.GetFingerprint(mol)
    sims = np.fromiter((DataStructs.TanimotoSimilarity(fp_mol,tfp) for tfp in TRAIN_FPS),
                       dtype=np.float64, count=len(TRAIN_FPS))
    hist_sim, edges_sim = np.histogram(sims, bins=50, range=(0.0,1.0))

    # Per-CYP training distribution + percentile + components
    cyp_data = {}
    for cyp in C.CYP_ISOFORMS:
        col = C.TARGET_COLS[cyp]
        y = _train_df[col].dropna().values
        hist, edges = np.histogram(y, bins=32, range=(1.5,8.0))
        v = cyp_preds[cyp]["pic50"]
        pct = float((y <= v).mean())*100
        comps = {}
        for k,val in cyp_preds[cyp]["components"].items():
            model_label = {"xgb_both":"XGB (FP+Desc)","lgbm_both":"LGBM (FP+Desc)",
                           "lgbm_fp":"LGBM (FP)","hgb_descr":"HGB (Descr)"}.get(k,k)
            comps[model_label] = float(val)
        cyp_data[cyp] = {
            "pic50": float(v),
            "percentile": round(pct,1),
            "ic50_uM": float(round((10**-v)*1e6,3)),
            "hist": hist.astype(int).tolist(),
            "edges": edges.tolist(),
            "components": comps,
            "weight": _WEIGHTS[cyp],
        }

    return jsonify({
        "canon_smiles": canon,
        "nn_similarity": float(sims.max()),
        "n_analogs_ge_05": int((sims>=0.5).sum()),
        "n_analogs_ge_03": int((sims>=0.3).sum()),
        "sim_hist": hist_sim.astype(int).tolist(),
        "sim_edges": edges_sim.tolist(),
        "cyps": cyp_data,
    })


@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    if "file" not in request.files:
        return jsonify({"error":"no file"}),400
    f=request.files["file"]
    try: df=pd.read_csv(f)
    except Exception as e: return jsonify({"error":str(e)}),400
    if "SMILES" not in df.columns:
        return jsonify({"error":"CSV must have a SMILES column."}),400
    rows=[]
    for _,row in df.iterrows():
        smi=row["SMILES"]
        rec={c:row[c] for c in df.columns if c!="SMILES"}
        rec["SMILES"]=smi
        mol,fp,d,canon,ok,_=featurize_one(smi)
        if not ok:
            for cyp in C.CYP_ISOFORMS: rec[C.TARGET_COLS[cyp]]=np.nan
            rec["DDI_risk"]="ERROR"; rec["AD_flag"]=-1; rec["nearest_train_tanimoto"]=np.nan
        else:
            preds=predict_pic50(mol,fp,d)
            for cyp in C.CYP_ISOFORMS: rec[C.TARGET_COLS[cyp]]=preds[cyp]["pic50"]
            tier,_,_=ddi_risk_tier(preds)
            rec["DDI_risk"]=tier
            fl,s,_,_=ad_flag(mol)
            rec["AD_flag"]=fl; rec["nearest_train_tanimoto"]=round(s,4)
        rows.append(rec)
    buf=io.StringIO(); pd.DataFrame(rows).to_csv(buf,index=False); buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),mimetype="text/csv",
                     as_attachment=True,download_name="CYP_INHIBITION_PREDICTIONS.csv")


if __name__ == "__main__":
    print(f"[profiler] {len(_RESULTS)} models loaded across {len(C.CYP_ISOFORMS)} isoforms.")
    print(f"[profiler] Train cache: {len(TRAIN_MOLS)} molecules, {len(TRAIN_FPS)} fingerprints.")
    host=os.environ.get("HOST","0.0.0.0"); port=int(os.environ.get("PORT","5000"))
    print(f"[profiler] Online at http://{host}:{port}")
    app.run(host=host,port=port,debug=False)
