#!/usr/bin/env python
"""
Standardize SMILES for TRPV1 data (unified IC50/EC50 processing)

This script performs:
  1. Filters non-organic structures (requires carbon, whitelisted elements only)
  2. Deep RDKit standardization (cleanup, charge, tautomer, stereo, isotope)
  3. Deduplicates by InChIKey

Inputs:
  - data/raw/TRPV1_chembl_{ENDPOINT}_cleaned_v1.csv

Outputs:
  - data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv
    (adds columns: CANONICAL_SMILES, InChIKey)

Usage:
    python 01_standardize_smiles.py --endpoint IC50
    python 01_standardize_smiles.py --endpoint EC50
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import pandas as pd

# Add repository root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import shared utilities
from code.utils.config import get_raw_file, get_standardized_file, COL_INCHIKEY
from code.utils.mol_processing import process_smiles

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

# ============================================================================
# Main Processing
# ============================================================================

def standardize_smiles(endpoint):
    """
    Run SMILES standardization pipeline for specified endpoint.

    Args:
        endpoint: 'IC50' or 'EC50'
    """
    t0 = time.time()

    # Get input/output paths
    input_csv = get_raw_file(endpoint)
    output_csv = get_standardized_file(endpoint)

    logging.info(f"=" * 70)
    logging.info(f"STEP 1: SMILES Standardization - {endpoint}")
    logging.info(f"=" * 70)

    # Load raw data
    logging.info(f"Loading data from: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
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
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - t0
    logging.info(f"Saved standardized data to: {output_csv}")
    logging.info(f"Processing completed in {elapsed:.1f}s")
    logging.info("")

    return n_after


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Standardize SMILES for TRPV1 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_standardize_smiles.py --endpoint IC50
  python 01_standardize_smiles.py --endpoint EC50
        """
    )
    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process (IC50 or EC50)"
    )

    args = parser.parse_args()
    standardize_smiles(args.endpoint)


if __name__ == "__main__":
    main()
