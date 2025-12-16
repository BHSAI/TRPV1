# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 16:58:29 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
07_TRPV1_EC50_MorganLR_external_AD_metrics.py

Final EC50 model (Morgan + LogisticRegression):

  • Train on full scaffold-train set
  • Evaluate on external scaffold-split test set:
      - overall metrics
      - metrics inside SDC-based applicability domain
      - metrics outside SDC-based applicability domain

  AD definition:
    - Compute SDC for all training compounds vs training set
    - Choose SDC threshold so that 95% of training compounds are inside AD
      (i.e., SDC >= threshold)
    - Apply same threshold to external set
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.linear_model import LogisticRegression
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

# ───────────── CONFIG ─────────────

TRAIN_CSV = "TRPV1_EC50_scaffold_split/TRPV1_EC50_train_scaffold.csv"
TEST_CSV  = "TRPV1_EC50_scaffold_split/TRPV1_EC50_exttest_scaffold.csv"

SMILES_COL = "SMILES"
TARGET_COL = "CLASS"

OUT_METRICS_CSV = Path("TRPV1_EC50_MorganLR_external_AD_metrics.csv")

BASE_SEED = 42

# ───────────── HELPERS ─────────────

def load_labelled_mols(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[SMILES_COL, TARGET_COL]).copy()
    df["Mol"] = df[SMILES_COL].map(Chem.MolFromSmiles)
    df = df.dropna(subset=["Mol"]).reset_index(drop=True)
    y = df[TARGET_COL].astype(int).to_numpy()
    return df, y

# RDKit Morgan generator → ExplicitBitVect
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def morgan_fp(mol):
    """Return RDKit ExplicitBitVect for a Mol."""
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)

def compute_morgan_fps(mols):
    return [morgan_fp(m) for m in mols]

# Standard metrics you already use
def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    gmean = np.sqrt(sens * spec) if sens * spec else 0.0

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
        TP=tp, FP=fp, TN=tn, FN=fn,
    )

# ───────────── SDC / AD ─────────────

def compute_sdc_scores(train_fps, query_fps, eps=1e-6):
    """
    Compute SDC for each query fingerprint given list of training fps.

    SDC = sum_i exp(-3 * TD_i) / (1 - TD_i),
    where TD_i = 1 - TanimotoSimilarity.
    """
    train_fps_clean = [fp for fp in train_fps if fp is not None]
    if len(train_fps_clean) == 0:
        raise ValueError("No valid training fingerprints for SDC.")

    sdc_scores = np.zeros(len(query_fps), dtype=float)

    for idx, qfp in enumerate(query_fps):
        if qfp is None:
            sdc_scores[idx] = np.nan
            continue

        sims = DataStructs.BulkTanimotoSimilarity(qfp, train_fps_clean)
        td = 1.0 - np.asarray(sims, dtype=float)

        td = np.clip(td, eps, 1.0 - eps)
        contrib = np.exp(-3.0 * td) / (1.0 - td)
        sdc_scores[idx] = contrib.sum()

    return sdc_scores

def choose_sdc_threshold(train_sdc, coverage=0.95):
    """
    Choose SDC threshold so that `coverage` fraction of *training* compounds
    are inside AD (SDC >= threshold).
    """
    train_sdc = np.asarray(train_sdc)
    train_sdc = train_sdc[~np.isnan(train_sdc)]
    # If coverage=0.95 → take 5th percentile as threshold
    q = 1.0 - coverage
    return float(np.quantile(train_sdc, q))

def metrics_for_region(label, y_true, y_prob, mask, thr=0.5):
    """Return full metrics dict for a region (overall / inside / outside)."""
    mask = np.asarray(mask)
    if mask.sum() == 0:
        row = {k: np.nan for k in [
            "ROC_AUC","PR_AUC","Accuracy","Sensitivity","Specificity",
            "GMean","Precision","F1","MCC","Kappa","TP","FP","TN","FN"
        ]}
        row.update({"region": label, "n": 0, "coverage": 0.0})
        return row

    y_t = np.asarray(y_true)[mask]
    y_p = np.asarray(y_prob)[mask]
    y_hat = (y_p >= thr).astype(int)

    m = calc_metrics(y_t, y_hat, y_p)
    m["region"] = label
    m["n"] = int(mask.sum())
    m["coverage"] = float(mask.mean())
    return m

# ───────────── MAIN ─────────────

def main():
    # 1. Load train + external set
    train_df, y_train = load_labelled_mols(TRAIN_CSV)
    test_df,  y_test  = load_labelled_mols(TEST_CSV)

    print(f"Train rows: {len(train_df)}; External rows: {len(test_df)}")
    print(f"Train actives={np.sum(y_train==1)}, inactives={np.sum(y_train==0)}")

    # 2. Morgan fingerprints
    train_fps = compute_morgan_fps(train_df["Mol"])
    test_fps  = compute_morgan_fps(test_df["Mol"])

    # Convert Morgan bitvectors to numpy arrays for LR
    def fps_to_array(fps):
        arr = np.zeros((len(fps), 2048), dtype=np.int8)
        tmp = np.zeros(2048, dtype=int)
        for i, fp in enumerate(fps):
            if fp is None:
                continue
            DataStructs.ConvertToNumpyArray(fp, tmp)
            arr[i, :] = tmp
        return arr

    X_train = fps_to_array(train_fps)
    X_test  = fps_to_array(test_fps)

    # 3. Train final Morgan + LogisticRegression model
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=BASE_SEED,
    )
    clf.fit(X_train, y_train)

    # 4. Predictions on external test
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # 5. SDC computation
    train_sdc = compute_sdc_scores(train_fps, train_fps)
    test_sdc  = compute_sdc_scores(train_fps, test_fps)

    sdc_thr = choose_sdc_threshold(train_sdc, coverage=0.95)
    print(f"SDC threshold (95% train coverage): {sdc_thr:.3f}")

    in_AD_test = test_sdc >= sdc_thr

    # 6. Metrics for overall / inside / outside
    rows = []
    rows.append(metrics_for_region("overall",   y_test, y_prob, np.ones_like(in_AD_test, dtype=bool)))
    rows.append(metrics_for_region("inside_AD", y_test, y_prob, in_AD_test))
    rows.append(metrics_for_region("outside_AD",y_test, y_prob, ~in_AD_test))

    res_df = pd.DataFrame(rows)

    # Optional: order columns similar to your bar plot CSV
    col_order = [
        "region", "n", "coverage",
        "Sensitivity", "Specificity", "GMean",
        "ROC_AUC", "MCC", "Accuracy", "PR_AUC",
        "Precision", "F1", "Kappa",
        "TP", "FP", "TN", "FN",
    ]
    col_order = [c for c in col_order if c in res_df.columns]
    res_df = res_df[col_order]

    res_df.to_csv(OUT_METRICS_CSV, index=False)
    print(f"\nSaved EC50 Morgan+LR external + AD metrics → {OUT_METRICS_CSV}")
    print(res_df)

if __name__ == "__main__":
    main()
