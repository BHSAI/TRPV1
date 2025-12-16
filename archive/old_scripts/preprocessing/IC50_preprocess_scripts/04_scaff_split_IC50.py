#!/usr/bin/env python
"""
Preprocess TRPV1 IC50 data - Step 4: Scaffold Split

This script performs scaffold-based train/test splitting:
  1. Extracts Bemis-Murcko scaffolds for each molecule
  2. Groups molecules by scaffold (prevents data leakage)
  3. Assigns scaffolds to train/test using DeepChem-style balanced splitting
  4. Validates class distribution in both splits

Inputs:
  - data/intermediate/TRPV1_IC50_cleaned_RDK.csv (from step 1)

Outputs:
  - data/pre-processed/TRPV1_IC50_train_scaffold.csv (80% train)
  - data/pre-processed/TRPV1_IC50_exttest_scaffold.csv (20% test)

Usage:
    python 04_scaff_split_IC50.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd

# Add repository root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import shared utilities
from code.utils.config import (
    get_standardized_file,
    get_train_file,
    get_test_file,
    COL_ID,
    COL_CANONICAL_SMILES,
    COL_CLASS,
    SPLIT_SIZES,
    RANDOM_SEED,
)
from code.utils.scaffold_utils import scaffold_split, validate_split

# ============================================================================
# Configuration
# ============================================================================

# Input/output paths
INPUT_CSV = get_standardized_file("IC50")
TRAIN_OUT = get_train_file("IC50")
TEST_OUT = get_test_file("IC50")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Run scaffold-based train/test splitting."""
    # Load data
    logging.info(f"Loading data from: {INPUT_CSV}")
    df_in = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df_in):,} rows")

    # Select and rename columns for splitting
    # Keep only: ID, SMILES (canonical), CLASS
    df = df_in[[COL_ID, COL_CANONICAL_SMILES, COL_CLASS]].copy()
    df = df.rename(columns={COL_CANONICAL_SMILES: "SMILES"})

    # Validate required columns
    required_cols = {COL_ID, "SMILES", COL_CLASS}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Perform scaffold split
    logging.info(f"Performing scaffold split (train/test = {SPLIT_SIZES[0]:.0%}/{SPLIT_SIZES[1]:.0%})...")
    smiles_list = df["SMILES"].tolist()
    train_idx, test_idx = scaffold_split(
        smiles_list,
        sizes=SPLIT_SIZES,
        seed=RANDOM_SEED,
        balanced=True
    )

    # Create train/test DataFrames
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    logging.info(f"Train set: {len(train_df):,} molecules ({len(train_df)/len(df):.1%})")
    logging.info(f"Test set:  {len(test_df):,} molecules ({len(test_df)/len(df):.1%})")

    # Validate split (check class distribution)
    validate_split(train_df, test_df, COL_CLASS)

    # Save outputs
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    logging.info(f"Saved train set to: {TRAIN_OUT}")
    logging.info(f"Saved test set to:  {TEST_OUT}")
    logging.info("Scaffold split completed successfully!")


if __name__ == "__main__":
    main()
