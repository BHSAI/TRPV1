# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 14:10:14 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
08_TRPV1_IC50_MorganLR_SDC_AD.py

Applicability Domain (AD) analysis for the final TRPV1 IC50 model
(Morgan fingerprints + Logistic Regression) using the SDC measure.

Scope:
  - Train Morgan+LR on the full scaffold-train IC50 set.
  - Compute SDC for train and external test compounds.
  - Define AD threshold so that 95% of training compounds are "inside AD".
  - Summarize external test performance inside vs outside AD.

Outputs:
  - TRPV1_IC50_MorganLR_SDC_train.csv
      ID / SMILES / CLASS / SDC / in_AD
  - TRPV1_IC50_MorganLR_SDC_test.csv
      ID / SMILES / CLASS / SDC / in_AD / y_true / y_proba / y_pred
  - TRPV1_IC50_MorganLR_AD_summary_scaffold.csv
      region / n / coverage / accuracy / roc_auc / pr_auc
"""

#!/usr/bin/env python
"""
08_TRPV1_IC50_MorganLR_SDC_AD.py

Applicability Domain (AD) analysis for the final TRPV1 IC50 model
(Morgan fingerprints + Logistic Regression) using the SDC measure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

# ───────────────── CONFIG ─────────────────

TRAIN_CSV = "TRPV1_IC50_scaffold_split/TRPV1_IC50_train_scaffold.csv"
TEST_CSV  = "TRPV1_IC50_scaffold_split/TRPV1_IC50_exttest_scaffold.csv"

SMILES_COL = "SMILES"
TARGET_COL = "CLASS"

RADIUS = 2
NBITS  = 2048

COVERAGE = 0.95          # 95% of training compounds inside AD
THRESHOLD_PROB = 0.5     # classification threshold for probabilities

OUT_DIR = Path("TRPV1_IC50_SDC_AD_MorganLR")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ───────── 1. Fingerprints ─────────

def compute_morgan_fps(mols, radius=2, nBits=2048):
    """Return a list of RDKit ExplicitBitVect Morgan fingerprints."""
    fps = []
    for m in mols:
        if m is None:
            fps.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)
            fps.append(fp)
    return fps


def fps_to_array(fps, nBits=2048):
    """
    Convert a list of ExplicitBitVect fingerprints to a 2D numpy array
    (n_samples × nBits) for use with scikit-learn.
    """
    X = np.zeros((len(fps), nBits), dtype=np.int8)
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        arr = np.zeros((nBits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X


# ───────── 2. SDC computation ─────────

def compute_sdc_scores(train_fps, query_fps, eps=1e-6):
    """
    Compute SDC for each query fingerprint given a list of training fingerprints.

    SDC = sum_i exp(-3 * TD_i) / (1 - TD_i)
    where TD_i = 1 - TanimotoSimilarity(query, train_i).
    """
    train_fps_clean = [fp for fp in train_fps if fp is not None]
    n_train = len(train_fps_clean)
    if n_train == 0:
        raise ValueError("No valid training fingerprints for SDC.")

    sdc_scores = np.zeros(len(query_fps), dtype=float)

    for idx, qfp in enumerate(query_fps):
        if qfp is None:
            sdc_scores[idx] = np.nan
            continue

        # Bulk Tanimoto similarity to all training fps → list of floats
        sims = np.array(
            DataStructs.BulkTanimotoSimilarity(qfp, train_fps_clean),
            dtype=float
        )

        # Convert to distances
        td = 1.0 - sims
        td = np.clip(td, eps, 1.0 - eps)

        contrib = np.exp(-3.0 * td) / (1.0 - td)
        sdc_scores[idx] = contrib.sum()

    return sdc_scores


# ───────── 3. AD threshold from training SDC ─────────

def choose_sdc_threshold(train_sdc, coverage=0.95):
    """
    Choose SDC threshold so that `coverage` fraction of training compounds
    are considered inside AD (SDC >= threshold).
    """
    train_sdc = np.asarray(train_sdc, dtype=float)
    train_sdc = train_sdc[~np.isnan(train_sdc)]
    if train_sdc.size == 0:
        raise ValueError("No valid SDC scores in training set.")
    threshold = float(np.quantile(train_sdc, 1.0 - coverage))
    return threshold


# ───────── 4. Performance summary inside vs outside AD ─────────

def summarize_performance_by_AD(y_true, y_proba, in_AD_mask, threshold_prob=0.5):
    """
    Summarize performance inside vs outside AD region.

    y_true      : array-like of true labels (0/1)
    y_proba     : array-like of predicted probabilities for positive class
    in_AD_mask  : boolean mask (True = inside AD)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    in_AD_mask = np.asarray(in_AD_mask)

    results = []

    for label, mask in [("inside_AD", in_AD_mask), ("outside_AD", ~in_AD_mask)]:
        if mask.sum() == 0:
            results.append({
                "region": label,
                "n": 0,
                "coverage": 0.0,
                "accuracy": np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
            })
            continue

        y_t = y_true[mask]
        y_p = y_proba[mask]
        y_hat = (y_p >= threshold_prob).astype(int)

        row = {
            "region": label,
            "n": int(mask.sum()),
            "coverage": float(mask.mean()),
            "accuracy": float(accuracy_score(y_t, y_hat)),
        }

        try:
            row["roc_auc"] = float(roc_auc_score(y_t, y_p))
        except ValueError:
            row["roc_auc"] = np.nan

        try:
            row["pr_auc"] = float(average_precision_score(y_t, y_p))
        except ValueError:
            row["pr_auc"] = np.nan

        results.append(row)

    return pd.DataFrame(results)


# ───────── 5. Data loading ─────────

def load_df_and_mols(csv_path: str):
    """
    Load CSV, keep rows with valid SMILES, return:
      df (cleaned), mols (list of Mol), y (labels)
    """
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


# ───────── 6. Main pipeline ─────────

def main():
    train_df, train_mols, y_train = load_df_and_mols(TRAIN_CSV)
    test_df,  test_mols,  y_test  = load_df_and_mols(TEST_CSV)

    print(f"Train rows (valid mols): {len(train_df)}")
    print(f"Test rows  (valid mols): {len(test_df)}")

    train_fps = compute_morgan_fps(train_mols, radius=RADIUS, nBits=NBITS)
    test_fps  = compute_morgan_fps(test_mols,  radius=RADIUS, nBits=NBITS)

    X_train = fps_to_array(train_fps, nBits=NBITS)
    X_test  = fps_to_array(test_fps,  nBits=NBITS)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (proba_test >= THRESHOLD_PROB).astype(int)

    print("Computing SDC scores for train...")
    train_sdc = compute_sdc_scores(train_fps, train_fps)
    print("Computing SDC scores for test...")
    test_sdc  = compute_sdc_scores(train_fps, test_fps)

    sdc_threshold = choose_sdc_threshold(train_sdc, coverage=COVERAGE)
    print(f"SDC threshold (coverage={COVERAGE:.2f}): {sdc_threshold:.3f}")

    in_AD_train = train_sdc >= sdc_threshold
    in_AD_test  = test_sdc  >= sdc_threshold

    ad_summary_df = summarize_performance_by_AD(
        y_true=y_test,
        y_proba=proba_test,
        in_AD_mask=in_AD_test,
        threshold_prob=THRESHOLD_PROB,
    )
    print("\nAD performance summary (external scaffold test):")
    print(ad_summary_df)

    train_out = train_df.copy()
    train_out["SDC"]   = train_sdc
    train_out["in_AD"] = in_AD_train.astype(int)
    train_out.to_csv(OUT_DIR / "TRPV1_IC50_MorganLR_SDC_train.csv", index=False)

    test_out = test_df.copy()
    test_out["SDC"]     = test_sdc
    test_out["in_AD"]   = in_AD_test.astype(int)
    test_out["y_true"]  = y_test
    test_out["y_proba"] = proba_test
    test_out["y_pred"]  = y_pred_test
    test_out.to_csv(OUT_DIR / "TRPV1_IC50_MorganLR_SDC_test.csv", index=False)

    ad_summary_df.to_csv(
        OUT_DIR / "TRPV1_IC50_MorganLR_AD_summary_scaffold.csv",
        index=False,
    )

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'TRPV1_IC50_MorganLR_SDC_train.csv'}")
    print(f"  {OUT_DIR / 'TRPV1_IC50_MorganLR_SDC_test.csv'}")
    print(f"  {OUT_DIR / 'TRPV1_IC50_MorganLR_AD_summary_scaffold.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
