# CYP INHIBITION PROFILER

> **Research-grade in silico CYP450 direct-inhibition profiler** — a 16-model gradient-boosted ensemble trained on 4,905 dose-response curves from the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test), wrapped in a cyberpunk/neon Flask web UI with interactive charts, DDI risk tiering, Lipinski / PAINS / bioavailability triage, and per-CYP applicability-domain flags.

![CYP Inhibition Profiler screenshot](screenshot.png)

## Live demo

A public tunnel is usually running during development at https://cyp-inhibition-profiler (see Releases / check with the author). To run it locally, see [Quick start](#quick-start).

## Features

- **Direct-inhibition pIC50 predictions** for the four major drug-metabolising isoforms: **CYP1A2, CYP2C9, CYP2D6, CYP3A4**
- **16-model ensemble** — XGBoost, LightGBM, Histogram-Gradient-Boost on Morgan chiral fingerprints (r=2, 2048-bit), 202 RDKit 2D descriptors, and both combined; blended with per-CYP non-negative ST-RAE-optimal weights
- **5-fold GroupKFold on Bemis-Murcko scaffolds** — no scaffold leakage across folds; honest prospective-performance estimate
- **Interactive analytics for every molecule** (Chart.js, dark-themed, neon cyberpunk palette):
  - Isoform bar chart with liability bands (INACTIVE / WEAK / ACTIVE / POTENT) — hover for IC50 + percentile + per-component model breakdown
  - Chemical-space Tanimoto similarity histogram vs all 4,905 training compounds with AD thresholds
  - Per-CYP training-distribution histograms showing the molecule's percentile in each isoform's activity population
  - Live descriptor grid with colour-coded drug-likeness
- **Overall DDI risk tier** — LOW RISK / MONITOR / HIGH DDI RISK banner with breathing pulse on high
- **Multi-parameter risk panel**:
  - Lipinski Rule-of-Five violations (MW/LogP/HBD/HBA)
  - PAINS pan-assay interference alerts (via RDKit `FilterCatalog`)
  - Heuristic oral-bioavailability score (0–100)
- **Applicability domain**: nearest-training-neighbour Tanimoto (Morgan FP) with HIGH/MEDIUM/LOW confidence pill; nearest-neighbour name + SMILES
- **Mouse-reactive UI**: neon glow cursor, cross-hair, parallax grid drift, card lift-on-hover, click-to-link between CYP cards and chart bars, animated bar entry
- **Single-SMILES and batch CSV** endpoints (`/api/predict`, `/api/predict_batch`)
- **Model diagnostics dashboard** at `/diagnostics` with OOF predicted-vs-experimental scatter, train/test distributions, model-comparison chart, ensemble-weight table

## Ensemble performance (5-fold scaffold CV)

| CYP | ST-RAE ↓ | R² | Spearman ρ |
|---|---|---|---|
| CYP1A2 | 0.903 | 0.26 | 0.52 |
| CYP2C9 | 0.772 | 0.37 | 0.59 |
| CYP2D6 | 0.995 | 0.15 | 0.39 |
| CYP3A4 | **0.546** | **0.59** | **0.77** |

**Macro ST-RAE ≈ 0.80** (improvement over best single-model XGB-both macro ST-RAE 0.832). Lower is better; 1.0 = predicting the training mean. CYP3A4 is the strongest endpoint; CYP2D6 is the hardest (low inter-lab reproducibility). See [`results/baseline_summary.csv`](results/baseline_summary.csv) for per-model numbers.

Submission: `FINAL_CYP_CHALLENGE_PREDICTIONS.csv` — 750 blinded test compounds, exact 6-column format required by the challenge (`SMILES, Molecule_Name, CYP1A2/2C9/2D6/3A4_pIC50_direct_inhibition`). Only 1 of 750 falls outside the 0.3 Tanimoto AD threshold.

## Quick start

```bash
# 1) Clone
git clone https://github.com/ademmanlincoln07-crypto/cyp-inhibition-profiler.git
cd cyp-inhibition-profiler

# 2) Install dependencies
pip install -r requirements.txt

# 3) Download the challenge raw data (only the inhibition train file is needed for the app)
#    Place cyp-challenge-TRAIN_inhibition.csv under data/raw/
#    from https://huggingface.co/datasets/openadmet/cyp-challenge-train-test
#    (Other raw tracks are optional; see data/raw/README.md)

# 4) Run the web app
python app.py
# Open http://127.0.0.1:5000
```

The first run will preprocess the training SMILES and cache fingerprints to `data/processed/` (~20 MB); subsequent starts are instant.

## Project structure

```
cyp-challenge/
├── app.py                     # Flask app + cyberpunk UI + interactive Chart.js dashboards
├── requirements.txt
├── DATASET_AUDIT_REPORT.md    # Pre-training data QA
├── README.md
├── data/
│   └── raw/                   # Official challenge CSVs (download from HuggingFace)
├── models/pipeline_artifacts.pkl   # Trained 16-model ensemble + feature builders
├── results/                   # Baseline summaries, ensemble weights, blind predictions
├── figures/                   # Diagnostic figures (OOF scatter, AD, distributions)
└── src/                       # Pipeline source
    ├── data_loader.py
    ├── preprocessing.py
    ├── features.py
    ├── models.py              # XGB / LGBM / HGB trainers
    ├── ensemble.py            # NNLS blend, ST-RAE metric
    ├── splits.py              # Scaffold GroupKFold
    ├── applicability_domain.py
    ├── metrics.py
    ├── multitask_model.py
    ├── submission.py
    ├── constants.py
    └── train_pipeline_final.py
```

## API

### `POST /api/predict` — single SMILES
```json
{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
```
Returns: canonical SMILES, molecule render (base64 PNG), per-CYP pIC50 + ensemble component breakdowns, liability verdicts, DDI risk tier, Lipinski/PAINS/bioavailability, nearest-training-neighbour AD info, **plus raw chart data** for the four interactive plots.

### `POST /api/predict_batch` — CSV
Upload a multipart file with a `SMILES` column; returns a CSV with predictions + `DDI_risk`, `AD_flag`, `nearest_train_tanimoto` columns appended.

### `POST /api/mol_profile` — raw chart data
Used by the front-end charts; returns histogram bins, percentiles, component weights for a given SMILES.

### `GET /diagnostics`
Static dashboard with overall model-performance metrics and global validation figures.

## Disclaimers

Predictions are computed **in silico** by a gradient-boosted ensemble. They are intended for **research, lead-optimisation and ADMET triage** — not as a substitute for experimental assays or clinical guidance. pIC50 values can be noisy; the confidence pill reflects chemical similarity to the training set, not prediction accuracy per se.

## Tags / Topics

`drug-discovery` `admet` `cyp450` `cytochrome-p450` `machine-learning` `cheminformatics` `rdkit` `xgboost` `lightgbm` `ddi-prediction` `pharmacokinetics` `bioavailability` `pains` `lipinski` `scaffold-cv` `cyberpunk-ui` `flask` `openadmet`

## License

MIT © the contributors. Training data © OpenADMET challenge organisers (see `data/raw/README.md`).
