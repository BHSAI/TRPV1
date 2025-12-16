# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 11:13:52 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
Scaffold split for TRPV1 IC50 data.
Outputs: *_train_scaffold1.csv, *_exttest_scaffold1.csv
"""
import pandas as pd, random, logging
from collections import defaultdict, Counter
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ----------------------------- config ---------------------------------------
INPUT_CSV  = "./TRPV1_EC50_SMILES_cleaned_RDK.csv"
TRAIN_OUT  = "./TRPV1_EC50_scaffold_split/TRPV1_EC50_train_scaffold.csv"
TEST_OUT   = "./TRPV1_EC50_scaffold_split/TRPV1_EC50_exttest_scaffold.csv"
SPLIT_SIZES = (0.8, 0.2)
random_seed        = 42
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def bemis_murcko(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    return (MurckoScaffold.MurckoScaffoldSmiles(mol=mol,
                                                includeChirality=include_chirality)
            if mol else None)

def scaffold_index_map(smiles_list):
    scaff2idx = defaultdict(set)
    for i, smi in enumerate(smiles_list):
        scaf = bemis_murcko(smi)          # stereo‑agnostic
        if scaf:
            scaff2idx[scaf].add(i)
    return scaff2idx

def scaffold_split(smiles_list, sizes=(0.8,0.2), seed=0, balanced=True):
    assert abs(sum(sizes) - 1.0) < 1e-6
    n_total       = len(smiles_list)
    n_train_target = int(round(sizes[0] * n_total))
    train, test   = [], []

    scaff2idx = scaffold_index_map(smiles_list)

    # ――― ordering logic (DeepChem‑style) ―――
    sets = list(scaff2idx.values())
    if balanced:
        big, small = [], []
        for s in sets:
            (big if len(s) > n_total * sizes[1] / 2 else small).append(s)
        rng = random.Random(seed)
        rng.shuffle(big)
        rng.shuffle(small)
        sets = big + small
    else:
        sets = sorted(sets, key=len, reverse=True)

    # ――― greedy fill ―――
    for s in sets:
        if len(train) + len(s) <= n_train_target:
            train.extend(s)
        else:
            test.extend(s)

    # Final checks
    assert len(set(train) & set(test)) == 0
    diff = abs(len(train)/n_total - sizes[0])
    if diff > 0.05:
        logging.warning(f"Train size deviates by {diff*100:.1f}% from target.")

    return train, test

def main():
    
    df_in = pd.read_csv(INPUT_CSV)
    df_subset = df_in[["ID", "CANONICAL_SMILES", "CLASS"]].copy()
    df_subset = df_subset.rename(columns={"CANONICAL_SMILES": "SMILES"})
    #print(df_subset.head())
    df = df_subset

    # Confirm required columns are present
    required_cols = {'ID', 'SMILES', 'CLASS'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in input: {required_cols - set(df.columns)}")


    #smiles = df["SMILES" if "SMILES" in df else "CANONICAL_SMILES"].tolist()
    smiles = df["SMILES"].tolist()
    train_idx, test_idx = scaffold_split(smiles, SPLIT_SIZES, seed=random_seed, balanced=True)

    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    logging.info(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    # class presence check
    for subset, name in [(train_df, "train"), (test_df, "test")]:
        counts = Counter(subset["CLASS"])
        logging.info(f"{name} class distribution: {counts}")
        if len(counts) < 2:
            raise ValueError(f"{name} set has only one class. Resplit with different seed.")

    Path(TRAIN_OUT).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    logging.info(f"Wrote {TRAIN_OUT} and {TEST_OUT}")

if __name__ == "__main__":
    main()
