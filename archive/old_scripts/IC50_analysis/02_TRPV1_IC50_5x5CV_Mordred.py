# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:33:53 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
Mordred descriptor pipeline on TRPV1 IC50 (train scaffold split):
  1. Compute descriptors on train
  2. Feature cleaning / selection
  3. Fit scaler; save feature list + scaler
  4. 5×5 repeated CV on train (internal model comparison)
  5. Save per-fold + summary CSVs

External scaffold test evaluation will be done in a separate script
using the saved 'features.pkl' and 'scaler.pkl'.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
# pip install mordred
from mordred import Calculator, descriptors

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Optional: XGBoost & LightGBM
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# ───────── config ────────────────────────────────────────────────
TRAIN_CSV = "TRPV1_IC50_scaffold_split/TRPV1_IC50_train_scaffold.csv"
OUT_DIR   = Path("TRPV1_IC50_scaff_Mordred")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS   = 5
N_FOLDS     = 5
BASE_SEED   = 42

MODEL_ORDER = [
    "KNN", "SVM", "Bayesian", "LogisticRegression",
    "RandomForest", "LightGBM", "XGBoost"
]
# -----------------------------------------------------------------


def compute_mordred(mol_series):
    """Compute Mordred descriptors for a pandas Series of Mol objects."""
    calc = Calculator(descriptors, ignore_3D=True)
    try:
        return calc.pandas(mol_series)
    except Exception as e:
        warnings.warn(f"Mordred failed: {e}")
        return pd.DataFrame(index=mol_series.index)


def corr_filter(df, thr=0.95):
    """Drop highly correlated features based on Spearman |rho| > thr."""
    c = df.corr(method="spearman").abs()
    upper = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
    drop = [col for col in upper.columns if (upper[col] > thr).any()]
    print(f"Dropping {len(drop)} highly correlated features (>{thr})")
    return df.drop(columns=drop)


def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    gmn  = np.sqrt(sens * spec) if sens * spec else 0.0

    # guard single-class probability vectors
    if len(np.unique(y_prob)) == 1:
        roc = pr = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
        pr  = average_precision_score(y_true, y_prob)

    return dict(
        ROC_AUC=roc,
        PR_AUC=pr,
        Accuracy=accuracy_score(y_true, y_pred),
        Sensitivity=sens,
        Specificity=spec,
        GMean=gmn,
        Precision=precision_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        MCC=matthews_corrcoef(y_true, y_pred),
        Kappa=cohen_kappa_score(y_true, y_pred),
        TP=tp, FP=fp, TN=tn, FN=fn
    )


def get_model_factories():
    """Return model-name → factory(rs) dict for 7 models."""
    factories = {
        "RandomForest":       lambda rs: RandomForestClassifier(
                                        n_estimators=500,
                                        random_state=rs,
                                        n_jobs=-1,
                                        class_weight="balanced"
                                    ),
        "Bayesian":           lambda rs: GaussianNB(),
        "SVM":                lambda rs: SVC(
                                        kernel="rbf",
                                        probability=True,
                                        random_state=rs
                                    ),
        "KNN":                lambda rs: KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": lambda rs: LogisticRegression(
                                        max_iter=2000,
                                        class_weight="balanced",
                                        random_state=rs
                                    ),
    }

    if lgb is not None:
        factories["LightGBM"] = lambda rs: lgb.LGBMClassifier(
            n_estimators=500,
            random_state=rs,
            n_jobs=-1
        )
    else:
        print("[WARN] lightgbm not installed; skipping LightGBM.")

    if XGBClassifier is not None:
        factories["XGBoost"] = lambda rs: XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=rs,
            n_jobs=-1,
            tree_method="hist",
            use_label_encoder=False
        )
    else:
        print("[WARN] xgboost not installed; skipping XGBoost.")

    return factories


def main():
    # ─── 1. load & descriptor calculation ─────────────────────────
    train = pd.read_csv(TRAIN_CSV)
    train["Mol"] = train["SMILES"].map(Chem.MolFromSmiles)
    train = train.dropna(subset=["Mol"]).reset_index(drop=True)

    print(f"Train rows after Mol filtering: {len(train)}")

    desc_train = compute_mordred(train["Mol"]).apply(
        pd.to_numeric, errors="coerce"
    )
    # replace inf with NaN
    desc_train = desc_train.replace([np.inf, -np.inf], np.nan)
    print("Raw descriptor matrix:", desc_train.shape)

    # ─── 2. cleaning/selection on train ───────────────────────────
    # drop all-NaN and constant features
    mask = ~desc_train.isna().all() & (desc_train.nunique() > 1)
    desc_train = desc_train.loc[:, mask]

    # drop features with too many NaNs
    nan_frac = desc_train.isna().mean()
    desc_train = desc_train.loc[:, nan_frac < 0.05]

    # impute remaining NaNs with column means
    desc_train = desc_train.fillna(desc_train.mean())
    print("After NaN / constant filter:", desc_train.shape)

    # variance filter
    vt = VarianceThreshold(threshold=0.01)
    X_var = vt.fit_transform(desc_train)
    feat_var = desc_train.columns[vt.get_support()]
    desc_var = pd.DataFrame(X_var, columns=feat_var, index=train.index)
    print("After variance filter:", desc_var.shape)

    # correlation filter
    desc_sel = corr_filter(desc_var, thr=0.95)
    features = desc_sel.columns.tolist()
    print("Final feature count:", len(features))

    # ─── 3. scaling ───────────────────────────────────────────────
    scaler = MinMaxScaler().fit(desc_sel)
    X_train = scaler.transform(desc_sel)
    y_train = train["CLASS"].values

    # save scaler + feature list for external test alignment later
    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUT_DIR / "features.pkl", "wb") as f:
        pickle.dump(features, f)

    # ─── 4. 5×5 CV on train ───────────────────────────────────────
    factories = get_model_factories()
    rows = []

    for rep in range(1, N_REPEATS + 1):
        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=BASE_SEED + rep
        )

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_te = X_train[tr_idx], X_train[te_idx]
            y_tr, y_te = y_train[tr_idx], y_train[te_idx]

            for name, factory in factories.items():
                rs = BASE_SEED + 100 * rep + fold
                model = factory(rs)
                model.fit(X_tr, y_tr)

                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_te)[:, 1]
                else:
                    # fall back to decision_function or hard labels
                    if hasattr(model, "decision_function"):
                        df = model.decision_function(X_te)
                        # scale to 0-1
                        min_df, max_df = df.min(), df.max()
                        y_prob = (df - min_df) / (max_df - min_df + 1e-9)
                    else:
                        y_pred_tmp = model.predict(X_te)
                        y_prob = y_pred_tmp.astype(float)

                y_pred = (y_prob >= 0.5).astype(int)

                rows.append(dict(
                    Set="CV",
                    Repeat=rep,
                    Fold=fold,
                    Model=name,
                    **metrics(y_te, y_pred, y_prob)
                ))

    df_cv = pd.DataFrame(rows)
    num_cols = df_cv.select_dtypes(include="number").columns

    # ─── 5. write per-fold + summary CSVs ─────────────────────────
    per_fold_csv = OUT_DIR / "IC50_Mordred_scaffCV_per_fold_metrics_5x5.csv"
    df_cv.to_csv(per_fold_csv, index=False)

    avg = (
        df_cv.groupby("Model")[num_cols]
        .mean()
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    avg[num_cols] = avg[num_cols].round(2)
    avg_csv = OUT_DIR / "IC50_Mordred_scaffCV_avg_metrics_5x5.csv"
    avg.to_csv(avg_csv, index=False)

    sd = (
        df_cv.groupby("Model")[num_cols]
        .agg(["mean", "std"])
        .reindex(MODEL_ORDER)
    )
    sd.columns = [f"{m}_{s}" for m, s in sd.columns]
    sd = sd.reset_index()
    sd.iloc[:, 1:] = sd.iloc[:, 1:].round(2)
    sd_csv = OUT_DIR / "IC50_Mordred_scaffCV_avg+sd_metrics_5x5.csv"
    sd.to_csv(sd_csv, index=False)

    print(f"Mordred 5×5 CV complete → {OUT_DIR}")
    print(f"  Saved: {per_fold_csv.name}, {avg_csv.name}, {sd_csv.name}")


if __name__ == "__main__":
    main()
