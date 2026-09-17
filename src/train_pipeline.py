#!/usr/bin/env python3
"""
End-to-end training pipeline for the OpenADMET CYP Challenge.

Order of operations:
  1. Load + preprocess train (inhibition) + test
  2. Build fingerprints + descriptors
  3. Scaffold-CV baselines: LightGBM, XGBoost, RF, HGB on FP / descriptors / both
  4. Pick the best single model per CYP (by ST-RAE in nested scaffold CV)
  5. Train multi-task MLP (shared encoder + 4 heads)
  6. Build weight-optimised ensemble of top per-CYP models + multi-task
  7. Refit final models on all training data
  8. Predict on blinded test, build + validate submission
  9. Applicability-domain flags, diagnostics, figures, saved artifacts

Reproducible via fixed SEED.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from src import constants as C
from src.data_loader import (
    load_train_inhibition, load_test_blinded,
)
from src.preprocessing import process_smiles
from src.features import build_features_for_df
from src.models import cv_cyp, FeatureBuilder, get_model
from src.multitask_model import (
    cv_multitask, fit_final_multitask, predict as predict_mt,
)
from src.metrics import regression_metrics, soft_threshold_abs_error
from src.ensemble import blend_predictions
from src.submission import build_regression_submission
from src.applicability_domain import nearest_neighbor_similarity, flag_out_of_domain

SEED = C.SEED
DEVICE = "cuda" if False else "cpu"  # CPU is fine for the MLP here


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    t_overall = time.time()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    os.makedirs(C.FIGURES_DIR, exist_ok=True)

    print("=" * 80)
    print("OpenADMET CYP Challenge — End-to-end Training Pipeline")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    print("\n[1/8] Loading & preprocessing data...")
    train = load_train_inhibition()
    test  = load_test_blinded()
    print(f"  train inhibition: {train.shape}, test blinded: {test.shape}")

    # SMILES standardisation (largest fragment only; no uncharging to avoid
    # perturbing pKa-sensitive chemistry)
    train_prep = process_smiles(train, name="train_inhibition", keep_largest_fragment=True)
    test_prep  = process_smiles(test,  name="test_blinded",    keep_largest_fragment=True)
    df_tr = train_prep.df
    df_te = test_prep.df

    # Combined dataset for feature caching (we build fingerprints/descriptors on
    # train+test together, which is safe because fingerprints/descriptors are
    # deterministic and do not learn statistics; only scaling/filtering must
    # avoid leakage, which we handle in FeatureBuilder inside folds).
    df_tr["_split"] = "train"
    df_te["_split"] = "test"
    # Pad test with NaN target columns so we can concat
    for col in df_tr.columns:
        if col not in df_te.columns:
            df_te[col] = np.nan
    df_all = pd.concat([df_tr, df_te[df_tr.columns]], ignore_index=True)

    # ------------------------------------------------------------------
    # 2. Feature generation
    # ------------------------------------------------------------------
    print("\n[2/8] Building fingerprints + descriptors (cached)...")
    feats = build_features_for_df(
        df_all, mol_col="mol",
        fp_radius=2, fp_nbits=2048, fp_use_chirality=True,
        include_descriptors=True,
        cache_tag="all_morgan_chiral_2048", use_cache=True,
    )
    fp_all = feats["fp"]
    descr_all = feats["descr_raw"]
    descr_names_all = feats["descr_names_raw"]
    print(f"  FP shape: {fp_all.shape}")
    print(f"  Descriptors (after global constant removal): {descr_all.shape[1]}")

    # Split back into train / test
    is_train = (df_all["_split"] == "train").values
    is_test  = (df_all["_split"] == "test").values
    df_train_only = df_all[is_train].reset_index(drop=True)
    df_test_only  = df_all[is_test].reset_index(drop=True)
    fp_tr   = fp_all[is_train]
    fp_te   = fp_all[is_test]
    descr_tr = descr_all[is_train]
    descr_te = descr_all[is_test]

    # ------------------------------------------------------------------
    # 3. Baseline screen (scaffold CV) — abbreviated for tractability
    # ------------------------------------------------------------------
    print("\n[3/8] Running nested scaffold-CV baselines...")
    # We evaluate 3 model families x 2 feature sets = 6 combos per CYP,
    # prioritising the combinations known to work well on ECFP data:
    # - LightGBM on (fp, both)
    # - XGBoost on (fp, both)
    # - RandomForest on (fp) as a robust baseline
    # - HistGBM on descriptors (best for descriptor-only)
    # - Ridge on descriptors (linear baseline)
    model_specs = [
        ("lgbm", "fp"),
        ("lgbm", "both"),
        ("xgb",  "fp"),
        ("xgb",  "both"),
        ("rf",   "fp"),
        ("hgb",  "descr"),
        ("ridge","descr"),
    ]

    cyp_results = {}  # (cyp, model, features) -> CYPResult
    summary_rows = []
    for cyp in C.CYP_ISOFORMS:
        for mn, fs in model_specs:
            try:
                res = cv_cyp(
                    cyp, df_train_only, fp_tr, descr_tr, descr_names_all,
                    model_name=mn, feature_set=fs,
                    n_outer=5, split_type="scaffold",
                    n_trials_inner=20, seed=SEED,
                )
                cyp_results[(cyp, mn, fs)] = res
                summary_rows.append({
                    "CYP": cyp, "model": mn, "features": fs,
                    "ST_RAE": res.oof_metrics["ST_RAE"],
                    "RMSE":   res.oof_metrics["RMSE"],
                    "R2":     res.oof_metrics["R2"],
                    "MAE":    res.oof_metrics["MAE"],
                    "Spearman": res.oof_metrics["Spearman"],
                    "time_sec": res.training_time_sec,
                })
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  !! {cyp}/{mn}/{fs} failed: {e}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(C.RESULTS_DIR, "baseline_summary.csv"), index=False)
    print("\nBaseline CV summary (sorted by macro ST-RAE):")
    macro = summary_df.groupby(["model", "features"])[["ST_RAE", "RMSE", "R2", "Spearman"]].mean().reset_index()
    macro = macro.sort_values("ST_RAE")
    print(macro.to_string(index=False))
    macro.to_csv(os.path.join(C.RESULTS_DIR, "baseline_macro_summary.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. Multi-task MLP
    # ------------------------------------------------------------------
    print("\n[4/8] Multi-task neural network (scaffold-CV)...")
    mt_cv = cv_multitask(
        df_train_only, fp_tr, descr_tr, descr_names_all,
        feature_set="both", n_outer=5, split_type="scaffold",
        seed=SEED, device=DEVICE,
        enc_hidden=768, enc_latent=384, head_hidden=96,
        dropout=0.25,
        lr=1e-3, weight_decay=1e-4,
        batch_size=128, max_epochs=120, patience=20,
    )
    _save_pickle(mt_cv, os.path.join(C.RESULTS_DIR, "multitask_cv.pkl"))

    # ------------------------------------------------------------------
    # 5. Choose per-CYP best and build ensemble
    # ------------------------------------------------------------------
    print("\n[5/8] Selecting best models per CYP + building ensemble...")
    best_per_cyp = {}
    for cyp in C.CYP_ISOFORMS:
        sub = summary_df[summary_df["CYP"] == cyp].sort_values("ST_RAE")
        best = sub.iloc[0]
        best_per_cyp[cyp] = (best["model"], best["features"])
        print(f"  {cyp}: best single = {best['model']} + {best['features']} "
              f"| ST-RAE = {best['ST_RAE']:.3f}, RMSE = {best['RMSE']:.3f}, "
              f"R2 = {best['R2']:.3f}, ρ = {best['Spearman']:.3f}")

    # Build OOF blend: best per-CYP single model + multi-task OOF
    # (We also include the second-best if it is close & different feature set,
    #  for diversity.)
    oof_blend = {}
    weights_per_cyp = {}
    for j, cyp in enumerate(C.CYP_ISOFORMS):
        target = C.TARGET_COLS[cyp]
        cl_col = C.CI_LOW_COLS[cyp]; ch_col = C.CI_HIGH_COLS[cyp]
        # find labelled rows for this CYP
        labelled = df_train_only[target].notna().values
        y_true = df_train_only.loc[labelled, target].values.astype(float)
        cl = df_train_only.loc[labelled, cl_col].values.astype(float)
        ch = df_train_only.loc[labelled, ch_col].values.astype(float)
        y_mean = float(y_true.mean())

        preds = {}
        # best single model
        bmn, bfs = best_per_cyp[cyp]
        best_res = cyp_results[(cyp, bmn, bfs)]
        preds[f"best_{bmn}_{bfs}"] = best_res.oof_preds
        # second-best with complementary feature set (if available)
        sub = summary_df[summary_df["CYP"] == cyp].sort_values("ST_RAE")
        for _, row in sub.iloc[1:].iterrows():
            key = f"{row['model']}_{row['features']}"
            r = cyp_results[(cyp, row["model"], row["features"])]
            preds[key] = r.oof_preds
            if len(preds) >= 3:
                break
        # multi-task OOF
        mt_oof_for_cyp = mt_cv["oof"][:, j]
        mt_mask = np.isfinite(mt_cv["Y"][:, j])
        # mt_cv is over the "any label" subset. We need to align to labelled
        # order for this CYP.
        df_lab_any = mt_cv["df_labelled"]
        # Build index map: position in any-label subset -> position in cyp-labelled subset
        cyp_names = df_train_only.loc[labelled, "Molecule_Name"].values
        any_names = df_lab_any["Molecule_Name"].values
        name_to_pos_in_cyp = {name: i for i, name in enumerate(cyp_names)}
        mt_oof_aligned = np.full(len(y_true), np.nan)
        for pos, name in enumerate(any_names):
            if name in name_to_pos_in_cyp and np.isfinite(mt_oof_for_cyp[pos]):
                mt_oof_aligned[name_to_pos_in_cyp[name]] = mt_oof_for_cyp[pos]
        # Some CYP labels in mt_cv may be missing (if a compound only had other CYPs)
        mt_finite = np.isfinite(mt_oof_aligned)
        if mt_finite.sum() > 0.9 * len(y_true):
            # Fill any remaining NaNs with the CYP mean (harmless because ensemble
            # weights will down-weight if it doesn't help).
            mt_oof_aligned[~mt_finite] = y_mean
            preds["multitask"] = mt_oof_aligned

        blend, w = blend_predictions(preds, y_true, cl, ch, y_mean)
        oof_blend[cyp] = (blend, preds, w, y_true, cl, ch)
        weights_per_cyp[cyp] = w
        blend_m = regression_metrics(blend, y_true, cl, ch, y_train=np.array([y_mean]))
        best_m = best_res.oof_metrics
        print(f"\n  {cyp} ensemble: ST-RAE = {blend_m['ST_RAE']:.3f} "
              f"(best single = {best_m['ST_RAE']:.3f})")
        print(f"    weights: {w}")

    _save_json(weights_per_cyp, os.path.join(C.RESULTS_DIR, "ensemble_weights.json"))

    # ------------------------------------------------------------------
    # 6. Refit final models on full training data
    # ------------------------------------------------------------------
    print("\n[6/8] Refitting final models on all training data...")
    final_models = {}
    for cyp in C.CYP_ISOFORMS:
        mn, fs = best_per_cyp[cyp]
        # The last-fit final_model inside cv_cyp was already trained on all labels; reuse it
        res = cyp_results[(cyp, mn, fs)]
        final_models[(cyp, "best")] = res
        print(f"  {cyp}: best={mn}/{fs} ready (n={res.n_labelled})")

    # Fit final multi-task model on all data
    print("  Fitting final multi-task MLP on all labelled data...")
    mt_final_res, mt_fb_final, _ = fit_final_multitask(
        df_train_only, fp_tr, descr_tr, descr_names_all,
        feature_set="both", device=DEVICE, seed=SEED,
        enc_hidden=768, enc_latent=384, head_hidden=96,
        dropout=0.25, lr=1e-3, weight_decay=1e-4,
        batch_size=128, max_epochs=150, patience=25,
    )

    # ------------------------------------------------------------------
    # 7. Test-set prediction & ensemble
    # ------------------------------------------------------------------
    print("\n[7/8] Generating blinded test predictions...")

    def predict_from_cypresult(res: "CYPResult", fp_te, descr_te):
        fb = res.final_feature_builder
        X_te = fb.transform(fp_te, descr_te)
        return res.final_model.predict(X_te).astype(np.float64)

    test_preds_by_model = {}
    for cyp in C.CYP_ISOFORMS:
        mn, fs = best_per_cyp[cyp]
        res = cyp_results[(cyp, mn, fs)]
        test_preds_by_model[f"best_{cyp}"] = predict_from_cypresult(res, fp_te, descr_te)

    # multi-task predictions on test
    # Need to transform test through a FeatureBuilder fit on ALL training data
    # (multi-task's final feature builder was fit on the any-label subset, but
    # we need features for ALL test compounds. Simpler: re-fit a fresh
    # FeatureBuilder on the full training descriptor matrix.)
    fb_fulltrain = FeatureBuilder(feature_set="both").fit(fp_tr, descr_tr)
    X_te_all = fb_fulltrain.transform(fp_te, descr_te)
    mt_test_preds = predict_mt(mt_final_res.model, X_te_all, device=DEVICE)
    # Also generate from cyp-specific models' own feature builders for the blend
    # We need: for each cyp, the blend was over models with their own feature
    # spaces — replicate those predictions on the test set.
    test_component_preds = {cyp: {} for cyp in C.CYP_ISOFORMS}
    for cyp in C.CYP_ISOFORMS:
        target = C.TARGET_COLS[cyp]
        labelled = df_train_only[target].notna().values
        for key in weights_per_cyp[cyp].keys():
            if key.startswith("best_") or key in ("lgbm_fp", "lgbm_both", "xgb_fp", "xgb_both", "rf_fp", "hgb_descr", "ridge_descr"):
                # Parse the model/feature from key
                if key.startswith("best_"):
                    parts = key[len("best_"):].split("_", 1)
                    if len(parts) != 2:
                        continue
                    mn_cand, fs_cand = parts[0], parts[1]
                else:
                    mn_cand, fs_cand = key.split("_", 1)
                if (cyp, mn_cand, fs_cand) not in cyp_results:
                    continue
                r = cyp_results[(cyp, mn_cand, fs_cand)]
                test_component_preds[cyp][key] = predict_from_cypresult(r, fp_te, descr_te)
            elif key == "multitask":
                j = C.CYP_ISOFORMS.index(cyp)
                test_component_preds[cyp]["multitask"] = mt_test_preds[:, j]

    final_preds = {}
    for j, cyp in enumerate(C.CYP_ISOFORMS):
        w = weights_per_cyp[cyp]
        blend = np.zeros(len(df_test_only), dtype=np.float64)
        w_sum = 0.0
        for key, wi in w.items():
            if key in test_component_preds[cyp]:
                blend += wi * test_component_preds[cyp][key]
                w_sum += wi
        if w_sum > 0:
            blend /= w_sum
        col = C.TARGET_COLS[cyp]
        final_preds[col] = blend
        # Clip to training range (extrapolation guard)
        y_train = df_train_only[C.TARGET_COLS[cyp]].dropna().values
        lo = float(np.percentile(y_train, 0.5))
        hi = float(np.percentile(y_train, 99.5))
        final_preds[col] = np.clip(blend, lo - 0.5, hi + 0.5)

    # ------------------------------------------------------------------
    # Build submission
    # ------------------------------------------------------------------
    out_csv = os.path.join(C.RESULTS_DIR, "FINAL_CYP_CHALLENGE_PREDICTIONS.csv")
    sub_df = build_regression_submission(df_test_only, final_preds, out_csv)
    print(sub_df.head())

    # ------------------------------------------------------------------
    # 8. Diagnostics, figures, applicability domain, artifacts
    # ------------------------------------------------------------------
    print("\n[8/8] Diagnostics + figures + applicability domain...")

    # 8.1 Predicted-vs-true (OOF) for each CYP
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for i, cyp in enumerate(C.CYP_ISOFORMS):
        ax = axes[i // 2, i % 2]
        blend, _, _, y_true, cl, ch = oof_blend[cyp]
        ax.scatter(y_true, blend, s=10, alpha=0.5, color="#1f77b4")
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=1)
        ax.set_xlabel(f"Experimental {cyp} pIC50")
        ax.set_ylabel(f"Predicted {cyp} pIC50 (OOF blend)")
        m = regression_metrics(blend, y_true, cl, ch, y_train=np.array([y_true.mean()]))
        ax.set_title(f"{cyp}  ST-RAE={m['ST_RAE']:.3f}  R²={m['R2']:.2f}  ρ={m['Spearman']:.2f}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.FIGURES_DIR, "oof_pred_vs_true.png"), dpi=150)
    plt.close()

    # 8.2 Applicability domain
    train_mols = df_train_only["mol"].tolist()
    test_mols  = df_test_only["mol"].tolist()
    flags, sims = flag_out_of_domain(train_mols, test_mols)
    print(f"  AD flags: 0=high ({(flags==0).sum()}), 1=med ({(flags==1).sum()}), 2=low ({(flags==2).sum()})")
    print(f"  Nearest-neighbour Tanimoto: min={sims.min():.2f}, "
          f"median={np.median(sims):.2f}, mean={sims.mean():.2f}, max={sims.max():.2f}")

    plt.figure(figsize=(7, 4))
    plt.hist(sims, bins=40, color="#2ca02c", alpha=0.75, edgecolor="black", lw=0.3)
    plt.xlabel("Tanimoto to nearest training neighbour (Morgan r=2, 2048 bits)")
    plt.ylabel("Count")
    plt.title("Test-set applicability domain: similarity to training set")
    plt.axvline(0.3, color="red",   ls="--", label="low-AD threshold (0.3)")
    plt.axvline(0.5, color="orange", ls="--", label="med-AD threshold (0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(C.FIGURES_DIR, "applicability_domain.png"), dpi=150)
    plt.close()

    # Attach AD info to a copy of the submission (not in the leaderboard
    # submission, but useful for inspection)
    ad_df = sub_df.copy()
    ad_df["nearest_train_tanimoto"] = sims
    ad_df["AD_flag"] = flags
    ad_df.to_csv(os.path.join(C.RESULTS_DIR, "FINAL_CYP_CHALLENGE_PREDICTIONS_with_AD.csv"), index=False)

    # 8.3 Save CV results + model objects
    _save_pickle(cyp_results, os.path.join(C.MODELS_DIR, "cyp_specific_models.pkl"))
    torch_save_path = os.path.join(C.MODELS_DIR, "multitask_final.pt")
    import torch
    torch.save(mt_final_res.model.state_dict(), torch_save_path)
    _save_pickle({
        "feature_builder_fulltrain": fb_fulltrain,
        "weights_per_cyp": weights_per_cyp,
        "best_per_cyp": best_per_cyp,
        "summary_df": summary_df,
        "mt_config": mt_final_res.config,
    }, os.path.join(C.MODELS_DIR, "pipeline_artifacts.pkl"))

    # 8.4 Test-prediction distributions
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, cyp in enumerate(C.CYP_ISOFORMS):
        col = C.TARGET_COLS[cyp]
        ax = axes[i // 2, i % 2]
        ax.hist(final_preds[col], bins=30, alpha=0.7, color="#9467bd", edgecolor="black", lw=0.3)
        ax.hist(df_train_only[col].dropna(), bins=30, alpha=0.3, color="#1f77b4", edgecolor="black", lw=0.3, label="train")
        ax.set_title(f"{cyp} — predicted test vs train distribution")
        ax.set_xlabel("pIC50"); ax.set_ylabel("Count")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(C.FIGURES_DIR, "test_vs_train_distributions.png"), dpi=150)
    plt.close()

    elapsed = time.time() - t_overall
    print(f"\n=== Pipeline complete in {elapsed/60:.1f} minutes ===")
    print(f"Submission: {out_csv}")
    print(f"Artifacts:  models/, results/, figures/")


if __name__ == "__main__":
    import torch
    main()
