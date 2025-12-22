#!/usr/bin/env python
"""
Scaffold-based train/test split for TRPV1 data (unified IC50/EC50 processing)

This script performs scaffold-based train/test splitting:
  1. Extracts Bemis-Murcko scaffolds for each molecule
  2. Groups molecules by scaffold (prevents data leakage)
  3. Assigns scaffolds to train/test using DeepChem-style balanced splitting
  4. Validates class distribution in both splits

Inputs:
  - data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv (from step 1)

Outputs:
  - data/pre-processed/TRPV1_{ENDPOINT}_train_scaffold.csv (80% train)
  - data/pre-processed/TRPV1_{ENDPOINT}_exttest_scaffold.csv (20% test)

Usage:
    python 04_scaffold_split.py --endpoint IC50
    python 04_scaffold_split.py --endpoint EC50
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

# Add repository root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import shared utilities
from code.utils.config import (
    get_standardized_file,
    get_train_file,
    get_test_file,
    COL_ID,
    COL_CANONICAL_SMILES,
    COL_CLASS,
    COL_INCHIKEY,
    SPLIT_SIZES,
    RANDOM_SEED,
)
from code.utils.scaffold_utils import scaffold_split, validate_split

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================================================================
# Main Processing
# ============================================================================

def scaffold_splitting(endpoint):
    """
    Run scaffold-based train/test splitting for specified endpoint.

    Args:
        endpoint: 'IC50' or 'EC50'
    """
    # Get input/output paths
    input_csv = get_standardized_file(endpoint)
    train_out = get_train_file(endpoint)
    test_out = get_test_file(endpoint)

    logging.info(f"=" * 70)
    logging.info(f"STEP 4: Scaffold-based Train/Test Split - {endpoint}")
    logging.info(f"=" * 70)

    # Load data
    logging.info(f"Loading data from: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df_in = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df_in):,} rows")

    # Select and rename columns for splitting
    # Keep only: ID, SMILES (canonical), CLASS
    df = df_in[[COL_ID, COL_CANONICAL_SMILES, COL_CLASS, COL_INCHIKEY]].copy()
    df = df.rename(columns={COL_CANONICAL_SMILES: "SMILES"})

    # Validate required columns
    required_cols = {COL_ID, "SMILES", COL_CLASS, COL_INCHIKEY}
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
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logging.info(f"Saved train set to: {train_out}")
    logging.info(f"Saved test set to:  {test_out}")
    logging.info(f"Scaffold split completed successfully!")
    logging.info("")

    return len(train_df), len(test_df)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scaffold-based train/test split for TRPV1 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_scaffold_split.py --endpoint IC50
  python 04_scaffold_split.py --endpoint EC50
        """
    )
    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process (IC50 or EC50)"
    )

    args = parser.parse_args()
    scaffold_splitting(args.endpoint)


if __name__ == "__main__":
    main()
