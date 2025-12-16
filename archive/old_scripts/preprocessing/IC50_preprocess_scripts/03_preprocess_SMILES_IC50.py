#!/usr/bin/env python
"""
Preprocess TRPV1 IC50 data - Step 1: Standardization

This script performs:
  1. Filters non-organic structures (requires carbon, whitelisted elements only)
  2. Deep RDKit standardization (cleanup, charge, tautomer, stereo, isotope)
  3. Deduplicates by InChIKey

Inputs:
  - data/raw/TRPV1_chembl_IC50_cleaned_v1.csv

Outputs:
  - data/intermediate/TRPV1_IC50_cleaned_RDK.csv
    (adds columns: CANONICAL_SMILES, InChIKey)

Usage:
    python 03_preprocess_SMILES_IC50.py
"""

import sys
import logging
import time
from pathlib import Path

import pandas as pd

# Add repository root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import shared utilities
from code.utils.config import get_raw_file, get_standardized_file, COL_INCHIKEY
from code.utils.mol_processing import process_smiles

# ============================================================================
# Configuration
# ============================================================================

# Input/output paths (use centralized config)
INPUT_CSV = get_raw_file("IC50")
OUTPUT_CSV = get_standardized_file("IC50")

# Logging setup
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Run SMILES standardization pipeline."""
    t0 = time.time()

    # Load raw data
    logging.info(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df):,} rows")

    # Process all SMILES: validate + standardize + generate InChIKey
    logging.info("Processing SMILES (validation, standardization, InChIKey generation)...")
    processed = df["SMILES"].apply(process_smiles)

    # Unpack results into two columns
    df[["CANONICAL_SMILES", "InChIKey"]] = pd.DataFrame(
        processed.tolist(),
        index=df.index
    )

    # Filter out failed molecules and deduplicate by InChIKey
    n_before = len(df)
    df = (df.dropna(subset=[COL_INCHIKEY])
            .drop_duplicates(COL_INCHIKEY)
            .reset_index(drop=True))
    n_after = len(df)
    n_removed = n_before - n_after

    logging.info(f"Removed {n_removed:,} invalid/duplicate molecules")
    logging.info(f"Retained {n_after:,} unique valid molecules")

    # Save output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = time.time() - t0
    logging.info(f"Saved standardized data to: {OUTPUT_CSV}")
    logging.info(f"Processing completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
