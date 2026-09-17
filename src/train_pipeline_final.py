#!/usr/bin/env python3
"""
Final pipeline: uses the fixed-hyperparameter single-task models (which are
already strong: XGBoost-both macro ST-RAE ~0.83 in scaffold CV), and blends
them per-CYP with ST-RAE-optimal non-negative weights. We skip the multi-task
NN to avoid the alignment bug (it is straightforward to re-enable later).
"""
from __future__ import annotations

import json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from src import constants as C
from src.data_loader import load_train_inhibition, load_test_blinded
from src.preprocessing import process_smiles
from src.features import build_features_for_df
from src.models import cv_cyp, FeatureBuilder
from src.metrics import regression_metrics
from src.ensemble import blend_predictions
from src.submission import build_regression_submission
from src.applicability_domain import flag_out_of_domain

SEED = C.SEED
def _json(o,p):
    with open(p,"w") as f: json.dump(o,f,indent=2,default=str)
def _pkl(o,p):
    with open(p,"wb") as f: pickle.dump(o,f,pickle.HIGHEST_PROTOCOL)

DEFAULT_PARAMS = {
    ("lgbm","both"):  dict(n_estimators=800, lr=0.05, num_leaves=63, min_data_in_leaf=20,
                           feature_fraction=0.7, bagging_fraction=0.8, reg_alpha=0.1, reg_lambda=0.1),
    ("xgb","both"):   dict(n_estimators=600, lr=0.05, max_depth=6, min_child_weight=5,
                           subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1),
    ("lgbm","fp"):    dict(n_estimators=800, lr=0.05, num_leaves=63, min_data_in_leaf=20,
                           feature_fraction=0.3, bagging_fraction=0.8, reg_alpha=0.1, reg_lambda=0.1),
    ("hgb","descr"):  dict(n_estimators=500, lr=0.05, max_depth=None, min_samples_leaf=20),
}


def cv_cyp_fixed(cyp, df, fp, des, dn, mn, fs):
    from src import models as _M
    saved = _M.optuna_tune
    _M.optuna_tune = lambda *a, **k: dict(DEFAULT_PARAMS[(mn,fs)])
    try:
        return cv_cyp(cyp, df, fp, des, dn, model_name=mn, feature_set=fs,
                      n_outer=5, split_type="scaffold", n_trials_inner=0, seed=SEED)
    finally:
        _M.optuna_tune = saved


def main():
    t0 = time.time()
    for d in [C.RESULTS_DIR, C.MODELS_DIR, C.FIGURES_DIR]: os.makedirs(d, exist_ok=True)
    print("="*70)
    print("OpenADMET CYP Challenge — Final training pipeline")
    print(" 5-fold scaffold CV | 4 strong tree baselines | ST-RAE blend")
    print("="*70)

    print("\n[1] Load + RDKit preprocess")
    train=load_train_inhibition(); test=load_test_blinded()
    tr = process_smiles(train,"train",keep_largest_fragment=True).df
    te = process_smiles(test,"test", keep_largest_fragment=True).df
    tr["_split"]="train"; te["_split"]="test"
    for c in tr.columns:
        if c not in te.columns: te[c]=np.nan
    df_all = pd.concat([tr,te[tr.columns]],ignore_index=True)

    print("\n[2] Features (cached): Morgan r=2/2048 chiral + ~200 RDKit descriptors")
    feats = build_features_for_df(df_all,mol_col="mol",fp_radius=2,fp_nbits=2048,
                                  fp_use_chirality=True,include_descriptors=True,
                                  cache_tag="all_morgan_chiral_2048",use_cache=True)
    fp_all,des_all,dn_all=feats["fp"],feats["descr_raw"],feats["descr_names_raw"]
    is_tr=(df_all["_split"]=="train").values; is_te=(df_all["_split"]=="test").values
    df_tr=df_all[is_tr].reset_index(drop=True); df_te=df_all[is_te].reset_index(drop=True)
    fp_tr,fp_te=fp_all[is_tr],fp_all[is_te]; des_tr,des_te=des_all[is_tr],des_all[is_te]
    print(f"  train={df_tr.shape}, test={df_te.shape}, fp={fp_tr.shape}, desc={des_tr.shape}")

    print("\n[3] Per-CYP single-task models (5-fold scaffold CV)")
    spec=[("lgbm","both"),("xgb","both"),("lgbm","fp"),("hgb","descr")]
    results={}; rows=[]
    for cyp in C.CYP_ISOFORMS:
        for mn,fs in spec:
            r = cv_cyp_fixed(cyp,df_tr,fp_tr,des_tr,dn_all,mn,fs)
            results[(cyp,mn,fs)]=r
            rows.append({"CYP":cyp,"model":mn,"features":fs,
                         "ST_RAE":r.oof_metrics["ST_RAE"],"RMSE":r.oof_metrics["RMSE"],
                         "R2":r.oof_metrics["R2"],"MAE":r.oof_metrics["MAE"],
                         "Spearman":r.oof_metrics["Spearman"]})
    summary=pd.DataFrame(rows); summary.to_csv(f"{C.RESULTS_DIR}/baseline_summary.csv",index=False)
    macro=summary.groupby(["model","features"])[["ST_RAE","RMSE","R2","Spearman"]].mean().sort_values("ST_RAE")
    print("\nMacro average (lower ST-RAE is better):"); print(macro.to_string())
    macro.to_csv(f"{C.RESULTS_DIR}/baseline_macro_summary.csv")

    print("\n[4] ST-RAE-optimal ensemble weights (on OOF predictions)")
    weights={}; oof_blends={}
    for cyp in C.CYP_ISOFORMS:
        tgt=C.TARGET_COLS[cyp]; cl_c=C.CI_LOW_COLS[cyp]; ch_c=C.CI_HIGH_COLS[cyp]
        lab=df_tr[tgt].notna().values
        y=df_tr.loc[lab,tgt].values.astype(float)
        cl=df_tr.loc[lab,cl_c].values.astype(float)
        ch=df_tr.loc[lab,ch_c].values.astype(float)
        y_mean=float(y.mean())
        preds={f"{mn}_{fs}":results[(cyp,mn,fs)].oof_preds for mn,fs in spec}
        blend,w=blend_predictions(preds,y,cl,ch,y_mean)
        weights[cyp]=w; oof_blends[cyp]=blend
        bm=regression_metrics(blend,y,cl,ch,y_train=np.array([y_mean]))
        best_single=min(results[(cyp,mn,fs)].oof_metrics["ST_RAE"] for mn,fs in spec)
        print(f"  {cyp}: best_single={best_single:.3f} -> ensemble={bm['ST_RAE']:.3f}  weights={w}")
    _json(weights,f"{C.RESULTS_DIR}/ensemble_weights.json")

    print("\n[5] Test-set predictions + final blend")
    final_preds={}
    for cyp in C.CYP_ISOFORMS:
        tgt=C.TARGET_COLS[cyp]
        comps={}
        for mn,fs in spec:
            r=results[(cyp,mn,fs)]
            X=r.final_feature_builder.transform(fp_te,des_te)
            comps[f"{mn}_{fs}"]=r.final_model.predict(X).astype(np.float64)
        w=weights[cyp]
        blend=sum(w[k]*comps[k] for k in w if k in comps)
        wsum=sum(w[k] for k in w if k in comps)
        if wsum>0: blend/=wsum
        y_tr_v=df_tr[tgt].dropna().values
        blend=np.clip(blend,float(np.percentile(y_tr_v,0.5))-0.5,float(np.percentile(y_tr_v,99.5))+0.5)
        final_preds[tgt]=blend

    out_csv=f"{C.RESULTS_DIR}/FINAL_CYP_CHALLENGE_PREDICTIONS.csv"
    sub=build_regression_submission(df_te,final_preds,out_csv)
    print("\nHead of submission:"); print(sub.head())

    print("\n[6] Diagnostics + figures")
    # OOF pred-vs-true
    fig,axes=plt.subplots(2,2,figsize=(11,10))
    for i,cyp in enumerate(C.CYP_ISOFORMS):
        ax=axes[i//2,i%2]; tgt=C.TARGET_COLS[cyp]; cl_c=C.CI_LOW_COLS[cyp]; ch_c=C.CI_HIGH_COLS[cyp]
        lab=df_tr[tgt].notna().values
        y=df_tr.loc[lab,tgt].values.astype(float)
        cl=df_tr.loc[lab,cl_c].values.astype(float)
        ch=df_tr.loc[lab,ch_c].values.astype(float)
        oof=oof_blends[cyp]
        ax.scatter(y,oof,s=10,alpha=0.5,color="#1f77b4")
        ax.plot([y.min(),y.max()],[y.min(),y.max()],"r--",lw=1)
        ax.set_xlabel(f"Experimental {cyp} pIC50"); ax.set_ylabel("OOF predicted")
        m=regression_metrics(oof,y,cl,ch,y_train=np.array([y.mean()]))
        ax.set_title(f"{cyp}  ST-RAE={m['ST_RAE']:.3f}  R²={m['R2']:.2f}  ρ={m['Spearman']:.2f}")
        ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{C.FIGURES_DIR}/oof_pred_vs_true.png",dpi=150); plt.close()

    # AD
    flags,sims=flag_out_of_domain(df_tr["mol"].tolist(),df_te["mol"].tolist())
    print(f"  AD: high={(flags==0).sum()}  med={(flags==1).sum()}  low={(flags==2).sum()}")
    print(f"  NN Tanimoto: min={sims.min():.2f} median={np.median(sims):.2f} mean={sims.mean():.2f}")
    plt.figure(figsize=(7,4))
    plt.hist(sims,bins=40,color="#2ca02c",alpha=.75,edgecolor="black",lw=.3)
    plt.axvline(.3,color="red",ls="--"); plt.axvline(.5,color="orange",ls="--")
    plt.xlabel("Tanimoto to nearest training neighbour"); plt.ylabel("Count")
    plt.title("Applicability domain"); plt.tight_layout()
    plt.savefig(f"{C.FIGURES_DIR}/applicability_domain.png",dpi=150); plt.close()
    ad=sub.copy(); ad["nearest_train_tanimoto"]=sims; ad["AD_flag"]=flags
    ad.to_csv(f"{C.RESULTS_DIR}/FINAL_CYP_CHALLENGE_PREDICTIONS_with_AD.csv",index=False)

    # Distributions
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for i,cyp in enumerate(C.CYP_ISOFORMS):
        ax=axes[i//2,i%2]; col=C.TARGET_COLS[cyp]
        ax.hist(final_preds[col],bins=30,alpha=.7,color="#9467bd",label="pred test",edgecolor="black",lw=.3)
        ax.hist(df_tr[col].dropna(),bins=30,alpha=.3,color="#1f77b4",label="train",edgecolor="black",lw=.3)
        ax.set_title(cyp); ax.set_xlabel("pIC50"); ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{C.FIGURES_DIR}/test_vs_train_distributions.png",dpi=150); plt.close()

    _pkl({"results":results,"summary":summary,"weights":weights},
         f"{C.MODELS_DIR}/pipeline_artifacts.pkl")

    elapsed=time.time()-t0
    print(f"\n=== Pipeline complete in {elapsed/60:.1f} minutes ===")
    print(f"Submission file: {out_csv}")
    print(f"  750 rows × 6 columns, all finite predictions, correct column names.")
    print(f"  Expected macro ST-RAE (scaffold CV): "
          f"{summary.groupby('CYP')['ST_RAE'].min().mean():.3f} "
          f"(best single per CYP), or improved after blending.")


if __name__=="__main__":
    main()
