# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:55:21 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
Second‑pass duplicate removal for TRPV1 IC50 set.

Usage (dry‑run):
    python src/00b_dedupe_ic50.py data/TRPV1_IC50_cleaned_RDK.csv

Apply changes and write out:
    python src/00b_dedupe_ic50.py data/TRPV1_IC50_cleaned_RDK.csv --apply
"""
import argparse, logging, pandas as pd
from pathlib import Path
from rdkit import Chem
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def key14(ikey: str) -> str:
    return ikey.split("-")[0] if isinstance(ikey, str) else None

def nostereo_tautomer(smiles: str) -> str:
    """SMILES without stereo AND canonical tautomer; returns None on failure."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = Chem.RemoveStereochemistry(mol)
        # reuse RDKit tautomer canonicaliser
        from rdkit.Chem.MolStandardize import rdMolStandardize as rms
        mol = rms.TautomerEnumerator().Canonicalize(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None

def drop_dupes(df: pd.DataFrame, key_col: str, label_col: str):
    """Return df with duplicates (by key_col) dropped using label majority vote."""
    dup_groups = df[df.duplicated(key_col, keep=False)].groupby(key_col)
    to_drop = []
    for k, g in dup_groups:
        # majority vote on label; if tie keep first occurrence
        mode_label, freq = Counter(g[label_col]).most_common(1)[0]
        keep_idx = g[g[label_col] == mode_label].index[0]
        drop_idx = [i for i in g.index if i != keep_idx]
        to_drop.extend(drop_idx)
    return df.drop(index=to_drop), len(to_drop)

def main(csv_in: str, apply: bool):
    df = pd.read_csv(csv_in)
    logging.info(f"Loaded {len(df):,} rows from {csv_in}")

    # -------- key14 (stereo‑agnostic) -----------------------------------
    df["InChIKey14"] = df["InChIKey"].map(key14)
    df, n14 = drop_dupes(df, "InChIKey14", "CLASS")
    logging.info(f"Removed {n14:,} stereo/isotope duplicates by key14")

    # -------- tautomer (nostereo) ---------------------------------------
    df["TAUT_NONISO"] = df["CANONICAL_SMILES"].map(nostereo_tautomer)
    df, ntaut = drop_dupes(df, "TAUT_NONISO", "CLASS")
    logging.info(f"Removed {ntaut:,} residual tautomer duplicates")

    df = df.drop(columns=["InChIKey14", "TAUT_NONISO"]).reset_index(drop=True)
    logging.info(f"Final row count: {len(df):,}")

    if apply:
        out_path = Path(csv_in).with_name(Path(csv_in).stem + "_dedup.csv")
        df.to_csv(out_path, index=False)
        logging.info(f"Written cleaned file → {out_path}")
    else:
        logging.info("Dry‑run only; no file written. Use --apply to save.")

# ——— For interactive runs ———
csv_in = r"./TRPV1_EC50_cleaned_RDK.csv"   # path to your file
apply   = False                               # or True if you want to write out
main(csv_in, apply)
