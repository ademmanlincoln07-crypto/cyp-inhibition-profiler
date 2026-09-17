#!/usr/bin/env python3
"""
Streamlined end-to-end pipeline (tuned for reasonable walltime while staying
scientifically defensible):
 - 5-fold scaffold CV (inner 10-trial Optuna) on 4 competitive model/feature
   combos per CYP:
     * LightGBM on ECFP+descriptors (both)
     * XGBoost on ECFP+descriptors (both)
     * LightGBM on ECFP alone
     * HistGBM on descriptors
 - Multi-task MLP (quick train, smaller architecture)
 - Per-CYP ST-RAE-weighted ensemble of all 4 single-task models + multi-task
 - Refit and predict test
 - Submission CSV + diagnostics
"""
from __future__ import annotations

import json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

import torch

from src import constants as C
from src.data_loader import load_train_inhibition, load_test_blinded
from src.preprocessing import process_smiles
from src.features import build_features_for_df
from src.models import cv_cyp, FeatureBuilder
from src.multitask_model import cv_multitask, fit_final_multitask, predict as predict_mt
from src.metrics import regression_metrics
from src.ensemble import blend_predictions
from src.submission import build_regression_submission
from src.applicability_domain import flag_out_of_domain

SEED = C.SEED
DEVICE = "cpu"


def _json(o, p):
    with open(p, "w") as f: json.dump(o, f, indent=2, default=str)
def _pkl(o, p):
    with open(p, "wb") as f: pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)


def main():
    t0 = time.time()
    for d in [C.RESULTS_DIR, C.MODELS_DIR, C.FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    print("="*70); print("OpenADMET CYP Pipeline (fast but rigorous)"); print("="*70)

    # 1) Load and preprocess
    print("\n[1] Load + preprocess")
    train = load_train_inhibition()
    test  = load_test_blinded()
    tr = process_smiles(train, name="train", keep_largest_fragment=True).df
    te = process_smiles(test,  name="test",  keep_largest_fragment=True).df
    tr["_split"] = "train"; te["_split"] = "test"
    for c in tr.columns:
        if c not in te.columns: te[c] = np.nan
    df_all = pd.concat([tr, te[tr.columns]], ignore_index=True)

    # 2) Features
    print("\n[2] Features")
    feats = build_features_for_df(
        df_all, mol_col="mol",
        fp_radius=2, fp_nbits=2048, fp_use_chirality=True,
        include_descriptors=True, cache_tag="all_morgan_chiral_2048", use_cache=True,
    )
    fp_all, descr_all, dn_all = feats["fp"], feats["descr_raw"], feats["descr_names_raw"]
    is_tr = (df_all["_split"]=="train").values
    is_te = (df_all["_split"]=="test").values
    df_tr, df_te = df_all[is_tr].reset_index(drop=True), df_all[is_te].reset_index(drop=True)
    fp_tr, fp_te = fp_all[is_tr], fp_all[is_te]
    des_tr, des_te = descr_all[is_tr], descr_all[is_te]
    print(f"  train={df_tr.shape}, test={df_te.shape}, fp={fp_tr.shape}, desc={des_tr.shape}")

    # 3) Baselines (5-fold scaffold, 10 inner trials)
    print("\n[3] Single-task CYP-specific models (5-fold scaffold CV)")
    spec = [
        ("lgbm", "both"),
        ("xgb",  "both"),
        ("lgbm", "fp"),
        ("hgb",  "descr"),
    ]
    results = {}
    rows = []
    for cyp in C.CYP_ISOFORMS:
        for mn, fs in spec:
            r = cv_cyp(cyp, df_tr, fp_tr, des_tr, dn_all,
                       model_name=mn, feature_set=fs,
                       n_outer=5, split_type="scaffold",
                       n_trials_inner=10, seed=SEED)
            results[(cyp, mn, fs)] = r
            rows.append({
                "CYP":cyp,"model":mn,"features":fs,
                "ST_RAE":r.oof_metrics["ST_RAE"],"RMSE":r.oof_metrics["RMSE"],
                "R2":r.oof_metrics["R2"],"MAE":r.oof_metrics["MAE"],
                "Spearman":r.oof_metrics["Spearman"],"time":r.training_time_sec,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{C.RESULTS_DIR}/baseline_summary.csv", index=False)
    macro = summary.groupby(["model","features"])[["ST_RAE","RMSE","R2","Spearman"]].mean().sort_values("ST_RAE")
    print("\nMacro ST-RAE across CYPs:"); print(macro.to_string())

    # 4) Multi-task NN
    print("\n[4] Multi-task MLP (5-fold scaffold CV)")
    mt_cv = cv_multitask(
        df_tr, fp_tr, des_tr, dn_all, feature_set="both", n_outer=5,
        split_type="scaffold", seed=SEED, device=DEVICE,
        enc_hidden=512, enc_latent=256, head_hidden=64, dropout=0.25,
        lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=80, patience=15,
    )

    # 5) Ensemble weights on OOF
    print("\n[5] Per-CYP ensemble (ST-RAE-optimal non-negative weights)")
    weights = {}
    for j, cyp in enumerate(C.CYP_ISOFORMS):
        tgt = C.TARGET_COLS[cyp]; cl_c=C.CI_LOW_COLS[cyp]; ch_c=C.CI_HIGH_COLS[cyp]
        lab = df_tr[tgt].notna().values
        y = df_tr.loc[lab, tgt].values.astype(float)
        cl= df_tr.loc[lab, cl_c].values.astype(float)
        ch= df_tr.loc[lab, ch_c].values.astype(float)
        y_mean = float(y.mean())
        preds = {}
        for mn, fs in spec:
            r = results[(cyp, mn, fs)]
            preds[f"{mn}_{fs}"] = r.oof_preds
        # align multi-task oof
        df_any = mt_cv["df_labelled"]
        cyp_names = df_tr.loc[lab,"Molecule_Name"].values
        name2pos = {nm:i for i,nm in enumerate(cyp_names)}
        mt_alg = np.full(len(y), np.nan)
        mt_vals = mt_cv["oof"][:, j]
        any_names = df_any["Molecule_Name"].values
        for pos, nm in enumerate(any_names):
            if nm in name2pos and np.isfinite(mt_vals[pos]):
                mt_alg[name2pos[nm]] = mt_vals[pos]
        mt_finite = np.isfinite(mt_alg)
        mt_alg[~mt_finite] = y_mean
        preds["multitask"] = mt_alg
        blend, w = blend_predictions(preds, y, cl, ch, y_mean)
        weights[cyp] = w
        bm = regression_metrics(blend, y, cl, ch, y_train=np.array([y_mean]))
        best_single = min(
            (results[(cyp,mn,fs)].oof_metrics["ST_RAE"] for mn,fs in spec)
        )
        print(f"  {cyp}: best_single ST-RAE={best_single:.3f} -> ensemble ST-RAE={bm['ST_RAE']:.3f}")
        print(f"    weights: {w}")
    _json(weights, f"{C.RESULTS_DIR}/ensemble_weights.json")

    # 6) Refit multi-task on all data
    print("\n[6] Refitting final multi-task MLP")
    mt_final, mt_fb, _ = fit_final_multitask(
        df_tr, fp_tr, des_tr, dn_all, feature_set="both", device=DEVICE, seed=SEED,
        enc_hidden=512, enc_latent=256, head_hidden=64, dropout=0.25,
        lr=1e-3, weight_decay=1e-4, batch_size=128, max_epochs=100, patience=20,
    )
    fb_full = FeatureBuilder(feature_set="both").fit(fp_tr, des_tr)
    X_te = fb_full.transform(fp_te, des_te)
    mt_test = predict_mt(mt_final.model, X_te, device=DEVICE)

    # 7) Build test-set component predictions + blend
    print("\n[7] Test-set predictions & ensemble")
    def predict_cyp(r, fp_te, des_te):
        X = r.final_feature_builder.transform(fp_te, des_te)
        return r.final_model.predict(X).astype(np.float64)

    final_preds = {}
    for j, cyp in enumerate(C.CYP_ISOFORMS):
        tgt = C.TARGET_COLS[cyp]
        comps = {}
        for mn, fs in spec:
            comps[f"{mn}_{fs}"] = predict_cyp(results[(cyp,mn,fs)], fp_te, des_te)
        comps["multitask"] = mt_test[:, j]
        w = weights[cyp]
        blend = sum(w[k]*comps[k] for k in w if k in comps)
        w_sum = sum(w[k] for k in w if k in comps)
        if w_sum > 0: blend /= w_sum
        # clip to training range
        y_tr_v = df_tr[tgt].dropna().values
        blend = np.clip(blend, float(np.percentile(y_tr_v,0.5))-0.5, float(np.percentile(y_tr_v,99.5))+0.5)
        final_preds[tgt] = blend

    # 8) Submission
    out_csv = f"{C.RESULTS_DIR}/FINAL_CYP_CHALLENGE_PREDICTIONS.csv"
    sub = build_regression_submission(df_te, final_preds, out_csv)
    print(sub.head())

    # 9) Diagnostics / figures
    print("\n[8] Diagnostics & figures")
    fig, axes = plt.subplots(2,2, figsize=(11,10))
    for i,cyp in enumerate(C.CYP_ISOFORMS):
        ax = axes[i//2, i%2]
        tgt = C.TARGET_COLS[cyp]; cl_c=C.CI_LOW_COLS[cyp]; ch_c=C.CI_HIGH_COLS[cyp]
        lab = df_tr[tgt].notna().values
        y = df_tr.loc[lab,tgt].values.astype(float)
        cl= df_tr.loc[lab,cl_c].values.astype(float)
        ch= df_tr.loc[lab,ch_c].values.astype(float)
        preds_oof = {}
        for mn,fs in spec:
            preds_oof[f"{mn}_{fs}"] = results[(cyp,mn,fs)].oof_preds
        cyp_names = df_tr.loc[lab,"Molecule_Name"].values
        n2p = {nm:i for i,nm in enumerate(cyp_names)}
        mt_a = np.full(len(y), np.nan)
        mt_v = mt_cv["oof"][:, i]
        for pos, nm in enumerate(mt_cv["df_labelled"]["Molecule_Name"].values):
            if nm in n2p and np.isfinite(mt_v[pos]):
                mt_a[n2p[nm]] = mt_v[pos]
        mt_a[~np.isfinite(mt_a)] = y.mean()
        preds_oof["multitask"] = mt_a
        w = weights[cyp]
        oof_blend = sum(w[k]*preds_oof[k] for k in w if k in preds_oof)
        wsum = sum(w[k] for k in w if k in preds_oof)
        if wsum>0: oof_blend/=wsum
        ax.scatter(y, oof_blend, s=10, alpha=0.5, color="#1f77b4")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1)
        ax.set_xlabel(f"Experimental {cyp} pIC50"); ax.set_ylabel("Predicted (OOF blend)")
        m = regression_metrics(oof_blend, y, cl, ch, y_train=np.array([y.mean()]))
        ax.set_title(f"{cyp}  ST-RAE={m['ST_RAE']:.3f}  R²={m['R2']:.2f}  ρ={m['Spearman']:.2f}")
        ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{C.FIGURES_DIR}/oof_pred_vs_true.png", dpi=150); plt.close()

    flags, sims = flag_out_of_domain(df_tr["mol"].tolist(), df_te["mol"].tolist())
    print(f"  AD: high={(flags==0).sum()}, med={(flags==1).sum()}, low={(flags==2).sum()}")
    print(f"  NN Tanimoto: min={sims.min():.2f} med={np.median(sims):.2f} mean={sims.mean():.2f}")
    plt.figure(figsize=(7,4))
    plt.hist(sims, bins=40, color="#2ca02c", alpha=.75, edgecolor="black", lw=.3)
    plt.axvline(.3, color="red", ls="--"); plt.axvline(.5, color="orange", ls="--")
    plt.xlabel("Tanimoto to nearest training neighbour"); plt.ylabel("Count")
    plt.title("Applicability domain"); plt.tight_layout()
    plt.savefig(f"{C.FIGURES_DIR}/applicability_domain.png", dpi=150); plt.close()

    ad_df = sub.copy(); ad_df["nearest_train_tanimoto"]=sims; ad_df["AD_flag"]=flags
    ad_df.to_csv(f"{C.RESULTS_DIR}/FINAL_CYP_CHALLENGE_PREDICTIONS_with_AD.csv", index=False)

    fig, axes = plt.subplots(2,2, figsize=(11,8))
    for i,cyp in enumerate(C.CYP_ISOFORMS):
        ax=axes[i//2,i%2]; col=C.TARGET_COLS[cyp]
        ax.hist(final_preds[col], bins=30, alpha=.7, color="#9467bd", label="pred test", edgecolor="black", lw=.3)
        ax.hist(df_tr[col].dropna(), bins=30, alpha=.3, color="#1f77b4", label="train", edgecolor="black", lw=.3)
        ax.set_title(cyp); ax.set_xlabel("pIC50"); ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{C.FIGURES_DIR}/test_vs_train_distributions.png", dpi=150); plt.close()

    _pkl({"results":results, "summary":summary, "weights":weights,
          "best_spec":spec}, f"{C.MODELS_DIR}/pipeline_artifacts.pkl")
    torch.save(mt_final.model.state_dict(), f"{C.MODELS_DIR}/multitask_final.pt")

    elapsed = time.time()-t0
    print(f"\n=== Complete in {elapsed/60:.1f} min ===")
    print(f"Submission: {out_csv}")


if __name__ == "__main__":
    main()
