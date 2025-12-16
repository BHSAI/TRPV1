# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 13:50:52 2025

@author: mabdulhameed
"""

#!/usr/bin/env python

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

# Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

# Optional: XGBoost & LightGBM
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# ───────────────── GLOBAL RNG ─────────────────
BASE_RANDOM_STATE = 42
random.seed(BASE_RANDOM_STATE)
np.random.seed(BASE_RANDOM_STATE)

# ───────────────── CONFIG ─────────────────

TRAIN_CSV = "TRPV1_IC50_scaffold_split/TRPV1_IC50_train_scaffold.csv"

OUT_FOLDERS = {
    "RDKITfp": "TRPV1_IC50_RDKITfp",
    "MACCS":   "TRPV1_IC50_MACCS",
    "Morgan":  "TRPV1_IC50_Morgan",
}

MODEL_ORDER = [
    "KNN", "SVM", "Bayesian", "LogisticRegression",
    "RandomForest", "LightGBM", "XGBoost",
]

SMILES_COL = "SMILES"   # change if needed
TARGET_COL = "CLASS"    # change if needed

N_REPEATS = 5
N_FOLDS   = 5

# ─────────────── Fingerprint generators (mols → fp) ───────────────
rdkit_gen  = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp_rdk(mol: Chem.Mol) -> np.ndarray:
    """2048-bit RDKit FP from a Mol."""
    arr = np.zeros(2048, dtype=np.int8)
    if mol is not None:
        DataStructs.ConvertToNumpyArray(rdkit_gen.GetFingerprint(mol), arr)
    return arr


def fp_morgan(mol: Chem.Mol) -> np.ndarray:
    """2048-bit Morgan FP (radius 2) from a Mol."""
    arr = np.zeros(2048, dtype=np.int8)
    if mol is not None:
        DataStructs.ConvertToNumpyArray(morgan_gen.GetFingerprint(mol), arr)
    return arr


def fp_maccs(mol: Chem.Mol) -> np.ndarray:
    """
    166-bit MACCS: original is 167 bits, bit-0 unused.
    We drop bit-0 and keep bits 1..166.
    """
    tmp = np.zeros(167, dtype=np.int8)
    arr = np.zeros(166, dtype=np.int8)
    if mol is not None:
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), tmp)
        arr[:] = tmp[1:]
    return arr


FEATURIZERS = {
    "RDKITfp": fp_rdk,
    "MACCS":   fp_maccs,
    "Morgan":  fp_morgan,
}

# ───────────────── MODEL FACTORIES ─────────────────


def get_model_factories():
    """
    Return a dict of model-name -> factory(rs) producing a fresh estimator
    with given random_state where applicable.
    """
    models = {
        "RandomForest":       lambda rs: RandomForestClassifier(
                                          n_estimators=500,
                                          class_weight="balanced",
                                          n_jobs=-1,
                                          random_state=rs),
        "Bayesian":           lambda rs: BernoulliNB(alpha=1.0),
        "SVM":                lambda rs: SVC(
                                          kernel="rbf",
                                          probability=True,
                                          random_state=rs),
        "KNN":                lambda rs: KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": lambda rs: LogisticRegression(
                                          max_iter=2000,
                                          class_weight="balanced",
                                          random_state=rs),
    }

    if lgb is not None:
        models["LightGBM"] = lambda rs: lgb.LGBMClassifier(
            n_estimators=500,
            random_state=rs,
            n_jobs=-1,
        )
    else:
        print("[WARN] lightgbm not installed; skipping LightGBM.")

    if XGBClassifier is not None:
        models["XGBoost"] = lambda rs: XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=rs,
            n_jobs=-1,
            tree_method="hist",
            use_label_encoder=False,
        )
    else:
        print("[WARN] xgboost not installed; skipping XGBoost.")

    return models


# ───────────────── METRICS helper ─────────────────


def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    gmean = np.sqrt(sens * spec) if sens * spec else 0.0

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
        GMean=gmean,
        Precision=precision_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        MCC=matthews_corrcoef(y_true, y_pred),
        Kappa=cohen_kappa_score(y_true, y_pred),
        TP=tp, FP=fp, TN=tn, FN=fn
    )


# ───────────────── DATA LOADING ─────────────────


def load_data(csv_path: str):
    """Load data, drop invalid SMILES/labels, return df, mols, y."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).copy()

    mols = []
    keep_idx = []
    for i, smi in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            keep_idx.append(i)

    df = df.iloc[keep_idx].reset_index(drop=True)
    y = df[TARGET_COL].astype(int).to_numpy()

    return df, mols, y


# ───────────────── PROBABILITY HELPER ─────────────────


def get_positive_proba(clf, X):
    """
    Return predicted probability for the positive class (label=1),
    robust to cases where only one class appears in training.
    """
    proba = clf.predict_proba(X)
    classes = clf.classes_

    if len(classes) == 2:
        idx = np.where(classes == 1)[0]
        if len(idx) == 0:
            # positive class unseen in training
            return np.zeros(proba.shape[0], dtype=float)
        return proba[:, idx[0]]

    elif len(classes) == 1:
        # only one class learned
        if classes[0] == 1:
            return np.ones(proba.shape[0], dtype=float)
        else:
            return np.zeros(proba.shape[0], dtype=float)

    else:
        raise ValueError("Unexpected number of classes in clf.classes_")


# ───────────────── CV LOOP FOR ONE FP ─────────────────


def run_cv_for_fp(fp_name: str, X: np.ndarray, y: np.ndarray, model_factories: dict):
    """Run 5×5 repeated stratified CV for one fingerprint matrix."""
    rows = []

    for repeat in range(1, N_REPEATS + 1):
        skf = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=BASE_RANDOM_STATE + repeat,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, factory in model_factories.items():
                # per-fold seed for reproducibility
                rs = BASE_RANDOM_STATE + 100 * repeat + fold_idx
                clf = factory(rs)

                clf.fit(X_train, y_train)

                y_prob = get_positive_proba(clf, X_test)
                y_pred = (y_prob >= 0.5).astype(int)

                metrics = calc_metrics(y_test, y_pred, y_prob)

                row = {
                    "Set": "CV",
                    "FP": fp_name,
                    "Repeat": repeat,
                    "Fold": fold_idx,
                    "Model": model_name,
                }
                row.update(metrics)
                rows.append(row)

    per_fold_df = pd.DataFrame(rows)
    return per_fold_df


# ───────────────── SUMMARY TABLES ─────────────────


def save_summary_tables(per_fold_df: pd.DataFrame, fp_name: str, out_dir: Path):
    non_num_cols = ["Set", "FP", "Repeat", "Fold", "Model"]
    numeric_cols = [c for c in per_fold_df.columns if c not in non_num_cols]

    present_models = [m for m in MODEL_ORDER if m in per_fold_df["Model"].unique()]

    # per-fold CSV
    per_fold_csv = out_dir / f"IC50_{fp_name}_rand_per_fold_metrics_5x5.csv"
    per_fold_df.to_csv(per_fold_csv, index=False)

    # mean metrics
    avg_df = (
        per_fold_df.groupby("Model")[numeric_cols]
        .mean()
        .reindex(present_models)
        .reset_index()
    )
    avg_df[numeric_cols] = avg_df[numeric_cols].round(2)
    avg_csv = out_dir / f"IC50_{fp_name}_rand_avg_metrics_5x5.csv"
    avg_df.to_csv(avg_csv, index=False)

    # mean ± SD
    mean_sd = (
        per_fold_df.groupby("Model")[numeric_cols]
        .agg(["mean", "std"])
        .reindex(present_models)
    )
    mean_sd.columns = [f"{m}_{stat}" for m, stat in mean_sd.columns]
    mean_sd = mean_sd.reset_index()
    mean_sd.iloc[:, 1:] = mean_sd.iloc[:, 1:].round(2)
    sd_csv = out_dir / f"IC50_{fp_name}_rand_avg+sd_metrics_5x5.csv"
    mean_sd.to_csv(sd_csv, index=False)

    print(f"  Saved: {per_fold_csv.name}, {avg_csv.name}, {sd_csv.name}")


# ───────────────── MAIN ─────────────────


def main():
    df, mols, y = load_data(TRAIN_CSV)
    print(f"Loaded {len(df)} molecules from {TRAIN_CSV}")

    model_factories = get_model_factories()
    print(f"Models used: {list(model_factories.keys())}")

    for fp_name, folder in OUT_FOLDERS.items():
        print(f"\n=== Fingerprint: {fp_name} ===")

        # featurise once from mols
        featurize = FEATURIZERS[fp_name]
        X = np.stack([featurize(mol) for mol in mols])
        print(f"FP matrix shape: {X.shape}")

        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        per_fold_df = run_cv_for_fp(fp_name, X, y, model_factories)
        save_summary_tables(per_fold_df, fp_name, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
