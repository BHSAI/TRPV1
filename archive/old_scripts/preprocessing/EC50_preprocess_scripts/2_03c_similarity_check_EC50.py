# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 14:19:03 2025

@author: mabdulhameed
"""
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator  # NEW
import numpy as np
from itertools import combinations
from pathlib import Path

# -------------------------------
# Config
INPUT_CSV           = "TRPV1_EC50_cleaned_RDK.csv"
SIM_MATRIX_CSV      = "similarity_matrix.csv"
IDENTICAL_PAIRS_CSV = "identical_pairs.csv"
RADIUS              = 2
FP_SIZE             = 2048
# -------------------------------

# 1) Load data
df     = pd.read_csv(INPUT_CSV)
ids    = df["InChIKey"].tolist()
smiles = df["CANONICAL_SMILES"].tolist()

# 2) Build a single MorganGenerator (no deprecation warning) :contentReference[oaicite:0]{index=0}
fp_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=FP_SIZE,
    useBondTypes=True,
    includeChirality=False,
    countSimulation=False
)

# 3) Compute fingerprints via the generator
fps = []
for smi in smiles:
    m = Chem.MolFromSmiles(smi)
    fps.append(fp_gen.GetFingerprint(m))

# 4) Build similarity matrix
n       = len(fps)
sim_mat = np.zeros((n, n), float)
for i in range(n):
    sim_mat[i, :] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)

## 5) Save full matrix
#sim_df = pd.DataFrame(sim_mat, index=ids, columns=ids)
#sim_df.to_csv(SIM_MATRIX_CSV)
#print(f"Saved all-vs-all sim matrix → {SIM_MATRIX_CSV}")

# 6) Extract off-diagonal pairs with sim == 1.0
pairs = []
for i, j in combinations(range(n), 2):
    if sim_mat[i, j] == 1.0:
        pairs.append({
            "InChIKey_1": ids[i],
            "InChIKey_2": ids[j],
            "Similarity": 1.0,
            "SMILES_1": smiles[i],
            "SMILES_2": smiles[j]
        })

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv(IDENTICAL_PAIRS_CSV, index=False)
print(f"Saved identical-similarity pairs → {IDENTICAL_PAIRS_CSV}")
print(f"Found {len(pairs_df)} off-diagonal pairs with sim = 1.0")

