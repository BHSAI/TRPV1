#!/usr/bin/env python
"""
Preprocess TRPV1 IC50 data - Step 3: Similarity Check (QC ONLY)

This script performs quality control by:
  1. Computing Morgan fingerprints for all molecules
  2. Finding pairs with identical fingerprints (Tanimoto similarity = 1.0)
  3. Saving results for manual inspection

NOTE: This is a QC/validation step only. No data is modified.
      Use this to identify potential duplicates that may have different InChIKeys.

Inputs:
  - data/intermediate/TRPV1_IC50_cleaned_RDK.csv (from step 1)

Outputs:
  - data/intermediate/TRPV1_IC50_identical_pairs.csv (pairs with sim = 1.0)

Usage:
    python 03c_similarity_check_IC50.py
"""

import sys
import logging
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Add repository root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import shared utilities
from code.utils.config import (
    get_standardized_file,
    get_similarity_pairs_file,
    COL_INCHIKEY,
    COL_CANONICAL_SMILES,
    MORGAN_RADIUS,
    MORGAN_FP_SIZE,
    MORGAN_USE_BOND_TYPES,
    MORGAN_INCLUDE_CHIRALITY,
    MORGAN_COUNT_SIMULATION,
)

# ============================================================================
# Configuration
# ============================================================================

# Input/output paths
INPUT_CSV = get_standardized_file("IC50")
OUTPUT_CSV = get_similarity_pairs_file("IC50")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ============================================================================
# Similarity Analysis
# ============================================================================

def compute_morgan_fingerprints(smiles_list):
    """
    Compute Morgan fingerprints for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of RDKit ExplicitBitVect fingerprints
    """
    # Create Morgan fingerprint generator (modern RDKit API)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE,
        useBondTypes=MORGAN_USE_BOND_TYPES,
        includeChirality=MORGAN_INCLUDE_CHIRALITY,
        countSimulation=MORGAN_COUNT_SIMULATION
    )

    # Generate fingerprints
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(fp_gen.GetFingerprint(mol))
        else:
            # Shouldn't happen if data is already validated, but handle gracefully
            logging.warning(f"Failed to parse SMILES: {smi}")
            fps.append(None)

    return fps


def find_identical_pairs(fps, ids, smiles):
    """
    Find all pairs with Tanimoto similarity = 1.0.

    Args:
        fps: List of fingerprints
        ids: List of molecule IDs (InChIKeys)
        smiles: List of SMILES strings

    Returns:
        DataFrame with columns: InChIKey_1, InChIKey_2, Similarity, SMILES_1, SMILES_2
    """
    n = len(fps)
    pairs = []

    logging.info(f"Comparing {n:,} molecules pairwise...")

    # Check all pairs (off-diagonal only)
    for i, j in combinations(range(n), 2):
        # Skip if either fingerprint is None
        if fps[i] is None or fps[j] is None:
            continue

        # Compute Tanimoto similarity
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])

        # Only keep perfect matches
        if sim == 1.0:
            pairs.append({
                "InChIKey_1": ids[i],
                "InChIKey_2": ids[j],
                "Similarity": 1.0,
                "SMILES_1": smiles[i],
                "SMILES_2": smiles[j]
            })

    return pd.DataFrame(pairs)


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Run similarity check for quality control."""
    # Load data
    logging.info(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df):,} rows")

    # Extract columns
    ids = df[COL_INCHIKEY].tolist()
    smiles = df[COL_CANONICAL_SMILES].tolist()

    # Compute Morgan fingerprints
    logging.info("Computing Morgan fingerprints...")
    fps = compute_morgan_fingerprints(smiles)

    # Find identical pairs (similarity = 1.0)
    logging.info("Finding pairs with identical fingerprints...")
    pairs_df = find_identical_pairs(fps, ids, smiles)

    # Save results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(OUTPUT_CSV, index=False)

    logging.info(f"Found {len(pairs_df):,} pairs with identical fingerprints (Tanimoto = 1.0)")
    logging.info(f"Saved results to: {OUTPUT_CSV}")

    # Summary
    if len(pairs_df) == 0:
        logging.info("No identical pairs found - dataset looks clean!")
    else:
        logging.info("Review the output file to investigate potential duplicates.")


if __name__ == "__main__":
    main()
