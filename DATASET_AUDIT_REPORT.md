# OpenADMET CYP Inhibition Blind Challenge — Dataset Audit Report

**Audit date:** 2026-09-17
**Auditor:** Agentic pipeline (RDKit + pandas / numpy)

---

## 1. Files audited

| File | Size | Rows | Cols | Role |
|---|---|---|---|---|
| `cyp-challenge-TRAIN_inhibition.csv` | 638 KB | **4,905** | 18 | Primary direct-inhibition (regression) training set |
| `cyp-challenge-TEST-BLINDED.csv` | 44 KB | **750** | 2 | Blinded test set (labels witheld) |
| `cyp-challenge-TRAIN_TDI.csv` | 1.2 MB | **6,145** | 36 | Time-dependent inhibition (TDI) classification + paired pIC50s |
| `cyp-challenge-single-concentration-TRAIN.csv` | 3.5 MB | **17,504** | 12 | Single-concentration primary screen (long format) |
| `cyp-challenge-TRAIN_Emax.csv` | 1.1 MB | **6,146** | 30 | Max-effect (Emax) vs positive control, both assay conditions |

---

## 2. Primary direct-inhibition training set (`TRAIN_inhibition.csv`)

### 2.1 Columns

Identifier / structure:
- `Molecule_Name` — unique OCNT-style identifier (e.g., `OCNT-0000422`), 4,905 unique values
- `SMILES` — isomeric SMILES strings (all RDKit-valid, see §2.5)

Primary regression targets (pIC50 units, log10 molar):
- `CYP1A2_pIC50_direct_inhibition`
- `CYP2C9_pIC50_direct_inhibition`
- `CYP2D6_pIC50_direct_inhibition`
- `CYP3A4_pIC50_direct_inhibition`

Per-target uncertainty metadata (95 % credible intervals from Bayesian dose-response curve fit):
- `*_conf_high` — upper bound of 95 % CI
- `*_conf_low`  — lower bound of 95 % CI
- `*_std`       — posterior std

### 2.2 Missingness pattern (primary targets)

| Target | Non-missing n | Missing | % labelled |
|---|---|---|---|
| CYP1A2_pIC50_direct_inhibition | 1,412 | 3,493 | 28.8 % |
| CYP2C9_pIC50_direct_inhibition | 1,285 | 3,620 | 26.2 % |
| CYP2D6_pIC50_direct_inhibition | 1,493 | 3,412 | 30.4 % |
| CYP3A4_pIC50_direct_inhibition | 2,335 | 2,570 | 47.6 % |

Missingness is **structured** (dose-response curves were only collected on a subset of compounds per isoform) — **not** MCAR. CYP3A4 has the most dose-response data, consistent with its status as the most clinically important isoform. CI/STD columns are missing exactly when the primary target is missing.

### 2.3 Target distributions (pIC50)

| Target | n | min | max | mean | median | std |
|---|---|---|---|---|---|---|
| CYP1A2 | 1,412 | 1.906 | 7.949 | 4.955 | 5.133 | 1.031 |
| CYP2C9 | 1,285 | 2.098 | 7.473 | 4.581 | 4.621 | 0.782 |
| CYP2D6 | 1,493 | 1.947 | 7.535 | 4.784 | 4.726 | 0.916 |
| CYP3A4 | 2,335 | 1.909 | 7.187 | 4.096 | 4.266 | 1.093 |

Target is **already on pIC50 scale** (i.e. –log10 IC50 in M). Values below pIC50 ≈ 4 correspond to IC50 > 100 µM — at/below the assay detection limit, which is why the official metric downweights them.

### 2.4 Posterior uncertainty (std)

| Target | median std | max std |
|---|---|---|
| CYP1A2 | 0.084 | 0.826 |
| CYP2C9 | 0.137 | 0.817 |
| CYP2D6 | 0.069 | 0.942 |
| CYP3A4 | 0.095 | 0.819 |

These credible intervals are **used by the official ST-RAE metric**: predictions landing inside the CI score zero error.

### 2.5 Structure quality

- Empty SMILES: **0**
- Invalid SMILES (RDKit parse fail): **0**
- Duplicate rows: **0**
- Duplicate raw SMILES: **0**
- Duplicate canonical SMILES: **0**

All 4,905 molecules parse cleanly in RDKit; no deduplication needed.

### 2.6 Identifier–SMILES mapping
`Molecule_Name` ↔ `SMILES` is a perfect 1:1 bijection (4,905 unique pairs). Names are stable OCNT identifiers that must be preserved in submissions.

---

## 3. Blinded test set (`TEST-BLINDED.csv`)

| Property | Value |
|---|---|
| Rows | 750 |
| Columns | `Molecule_Name`, `SMILES` |
| Labels | **None** (truly blinded) |
| Invalid SMILES | 0 |
| Duplicate SMILES | 0 |
| Duplicate rows | 0 |

All 750 test molecules parse cleanly. Identifiers must be preserved in exact original order for submission.

---

## 4. TDI training set (`TRAIN_TDI.csv`)

- 6,145 compounds × 36 columns.
- Classification labels (only for isoforms with TDI data):
  - `CYP2D6_is_TDI`: 1,173 False / 324 True / 4,648 NaN (TDI shift assessed only on a subset)
  - `CYP3A4_is_TDI`: 2,820 False / 764 True / 2,561 NaN
- Regression endpoints present for **both** assay conditions (`*_pIC50_TDI_condition` and `*_pIC50_direct_inhibition`) for all four isoforms, with matching CI/STD columns.
- TDI label definition: IC50 shift > 2-fold after NADPH pre-incubation (per challenge blog).
- For the **TDI classification track** of the challenge, only CYP2D6 and CYP3A4 are scored.

---

## 5. Single-concentration primary screen (`single-concentration-TRAIN.csv`)

- 17,504 rows (**long format**: 4,376 compounds × 4 CYPs)
- Columns include `OCNT_Batch`, `enzyme`, `plate_id`, `concentration_M` (fixed at 50 µM = 5e-5 M), `log2fc_estimate`, `log2fc_std_error`, `p_value`, `log2fc_fdr`, `log2fc_median`, `cohens_d`.
- Endpoint: log2 fold-change in signal vs control (negative ⇒ inhibition).
- Compound coverage exactly matches the 4,376-compound primary-screen library (all 4 enzymes per compound).
- These data are a **single-concentration primary screen**, not full dose-response curves. They are useful for:
  1. Pre-training / auxiliary features
  2. Weakly-supervised approaches
  3. Applicability-domain coverage
- They are **not** direct pIC50 measurements and should not be naively concatenated with the inhibition DRC data.

---

## 6. Emax training set (`TRAIN_Emax.csv`)

- 6,146 compounds × 30 columns.
- Contains `*_EmaxVsPosCtrl_TDI_condition` and `*_EmaxVsPosCtrl_direct_inhibition` for all four CYPs (maximal effect vs positive control), with CI columns.
- Also contains `CYP{1A2,2C9,2D6,3A4}_is_TDI` — TDI labels extended to all four isoforms (CYP1A2 and CYP2C9 have many NaNs, low positive rate).
- Emax is a continuous endpoint but **not** the primary regression target; useful for multi-task auxiliary heads only.

---

## 7. Train / test contamination & cross-set overlap

| Comparison | Overlapping canonical SMILES |
|---|---|
| `train_inhibition` ∩ `test_blinded` | **0** ✓ |
| `train_tdi` ∩ `test_blinded` | **0** ✓ |
| `train_single` ∩ `test_blinded` | **0** ✓ |
| `train_emax` ∩ `test_blinded` | **0** ✓ |
| `train_inhibition` ⊂ `train_tdi` | 4,905 (all inhibition compounds are in TDI set) |
| `train_inhibition` ⊂ `train_emax` | 4,905 |
| `train_single` ⊂ `train_inhibition` (almost) | 4,375 / 4,376 |
| `train_tdi` ⊂ `train_emax` | 6,145 (TDI set is essentially a subset of Emax set) |

**No test-set leakage.** The training sets are nested supersets of one another (the primary-screen library is 4,376 compounds, and 1,769 additional compounds were added for full DRC profiling → total 6,145). 750 blinded compounds are entirely distinct.

---

## 8. Submission requirements (from official challenge contract)

Two independent tracks, submitted as separate files:

### 8.1 Direct-inhibition regression track (primary focus)
CSV/Parquet, exactly **750 rows** in the original test order, with exactly these columns (case-sensitive):

```
SMILES
Molecule_Name
CYP1A2_pIC50_direct_inhibition
CYP2C9_pIC50_direct_inhibition
CYP2D6_pIC50_direct_inhibition
CYP3A4_pIC50_direct_inhibition
```

All predictions must be finite floats — no NaN / inf / -inf.

### 8.2 TDI classification track (secondary)
Exactly 750 rows with:

```
SMILES
Molecule_Name
CYP2D6_is_TDI
CYP3A4_is_TDI
```

Values must be Boolean (True/False or 1/0).

### 8.3 Primary metric (regression track)
**Macro-Soft-Threshold Relative Absolute Error (MA-ST-RAE)** across the four isoforms:
- Error per compound = distance from prediction to nearest bound of the true 95 % credible interval (0 if prediction lands inside the interval).
- Denominator: same error for the constant-mean predictor ⇒ ST-RAE = 1.0 ⇔ no better than predicting the mean.
- Compounds with pIC50 < 4 are downweighted (assay detection floor).
- 1,000 bootstrap resamples for confidence intervals on the leaderboard.
- Secondary metrics: MAE, R², Spearman ρ, Kendall τ.

### 8.4 Primary metric (TDI track)
**Matthews Correlation Coefficient (MCC)** per isoform, macro-averaged.

---

## 9. Key findings & implications for modelling

1. **Targets are already pIC50; no unit transform needed.** Predictions will be submitted on the same scale.
2. **Missing labels are structured not random.** Each CYP model must be trained on its own labelled subset (n ≈ 1,285–2,335). Multi-task learning can leverage partial overlap but must use masked loss.
3. **No invalid/duplicate structures** — no rows discarded during cleaning; we only standardize tautomers/salts where justified and record all decisions.
4. **Test set is chemically disjoint from training** (by scaffold/SMILES identity) — scaffold-split CV will give a more honest estimate of leaderboard performance than random splits.
5. **CI metadata matters.** The official metric uses credible intervals; we should calibrate predictions (and optionally use uncertainty-aware losses) rather than optimise RMSE blindly.
6. **Auxiliary data** (single-conc log2fc, TDI labels, Emax) provide extra coverage for the same chemistry — we can use them as auxiliary features or multi-task regularisers but must not mix endpoints casually.
7. **Submission must preserve original order, identifiers, and SMILES** of the 750-row blinded file.

---

## 10. Recommended next phases

Proceed to Phase 2 (SMILES/RDKit preprocessing) with a clean canonical-SMILES + descriptor/fingerprint caching step, then Phase 3–5 (fingerprints, descriptors, baselines) with nested scaffold CV optimised against ST-RAE as the model-selection metric.
