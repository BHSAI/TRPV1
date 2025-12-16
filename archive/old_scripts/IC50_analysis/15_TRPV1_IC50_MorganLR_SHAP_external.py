# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 21:22:38 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
08_TRPV1_IC50_MorganLR_SHAP_external.py

SHAP analysis for the final TRPV1 IC50 champion model:

  • Model: LogisticRegression (balanced) on 2048-bit Morgan fingerprints (radius 2)
  • Train on scaffold-split training set
  • Explain predictions on the external scaffold test set
  • Output:
      - SHAP beeswarm summary plot (top 20 bits)
      - SHAP bar plot (top 20 bits by mean |SHAP|)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sklearn.linear_model import LogisticRegression

import shap
import joblib  # optional, just to save the model if you want


# ───────────── CONFIG ─────────────

TRAIN_CSV = "TRPV1_IC50_scaffold_split/TRPV1_IC50_train_scaffold.csv"
TEST_CSV  = "TRPV1_IC50_scaffold_split/TRPV1_IC50_exttest_scaffold.csv"

OUT_DIR   = Path("TRPV1_IC50_shap_Morgan")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PKL = OUT_DIR/"MorganLR_scaffold_morgan.pkl"

RADIUS = 2
N_BITS = 2048
SEED   = 42

SMILES_COL = "SMILES"
TARGET_COL = "CLASS"


# ───────────── Morgan featurizer ─────────────

mgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

def featurize_smiles_to_morgan_df(smiles_series: pd.Series) -> pd.DataFrame:
    """
    Convert a pandas Series of SMILES into a DataFrame of
    2048-bit Morgan fingerprints (radius 2).
    Columns: Bit_0 ... Bit_2047
    """
    arrs = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(N_BITS, dtype=int)
        if mol is not None:
            fp = mgen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
        arrs.append(arr)

    X = pd.DataFrame(arrs, columns=[f"Bit_{i}" for i in range(N_BITS)])
    return X


# ───────────── main ─────────────

def main():
    # 1) Load train / test
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)

    # Drop rows with missing SMILES / CLASS
    df_tr = df_tr.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)
    df_te = df_te.dropna(subset=[SMILES_COL, TARGET_COL]).reset_index(drop=True)

    y_tr = df_tr[TARGET_COL].astype(int).values
    y_te = df_te[TARGET_COL].astype(int).values

    print(f"Train rows: {len(df_tr)} | External rows: {len(df_te)}")
    print(f"Train class balance: actives={np.sum(y_tr==1)}, inactives={np.sum(y_tr==0)}")
    print(f"Test  class balance: actives={np.sum(y_te==1)}, inactives={np.sum(y_te==0)}")

    # 2) Build Morgan fingerprints
    print("Building Morgan fingerprints (radius=2, 2048 bits)...")
    X_tr = featurize_smiles_to_morgan_df(df_tr[SMILES_COL])
    X_te = featurize_smiles_to_morgan_df(df_te[SMILES_COL])

    print("X_train shape:", X_tr.shape)
    print("X_test  shape:", X_te.shape)

    # 3) Train LogisticRegression on full scaffold-train
    print("\nFitting LogisticRegression (balanced)...")
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(X_tr.values, y_tr)

    # Optionally save the trained model for future reuse
    joblib.dump(model, MODEL_PKL)
    print("Model saved →", MODEL_PKL)

    # 4) SHAP explainer (LinearExplainer is appropriate for logistic regression)
    print("\nBuilding SHAP LinearExplainer...")

    # To keep it fast and stable, subsample background if train set is large
    if len(X_tr) > 300:
        background = shap.sample(X_tr, 300, random_state=SEED)
    else:
        background = X_tr.copy()

    explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")

    # 5) Compute SHAP values for the external test set
    print("Computing SHAP values on external scaffold test...")
    shap_vals = explainer.shap_values(X_te)

    # Handle old/new SHAP API variants
    if isinstance(shap_vals, list):
        # SHAP might return one array per class; take the positive class
        shap_matrix = shap_vals[1]
    else:
        shap_matrix = shap_vals

    assert shap_matrix.ndim == 2, "Expected 2D SHAP matrix (n_samples, n_features)."
    print("SHAP matrix shape:", shap_matrix.shape)

    # 6) SHAP plots – summary (beeswarm) and bar

    # SHAP summary (beeswarm) for top 20 bits
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        X_te,
        max_display=20,
        show=False
    )
    out_summary = OUT_DIR / "IC50_MorganLR_SHAP_summary_external.png"
    plt.savefig(out_summary, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved SHAP beeswarm summary →", out_summary)

    # SHAP bar plot (mean |SHAP|) for top 20 bits
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        X_te,
        max_display=20,
        plot_type="bar",
        show=False
    )
    out_bar = OUT_DIR / "IC50_MorganLR_SHAP_bar_external.png"
    plt.savefig(out_bar, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved SHAP bar summary →", out_bar)

    print("\nDone. Use these two plots as the IC50 SHAP figure (external test, Morgan + LR).")


if __name__ == "__main__":
    main()
