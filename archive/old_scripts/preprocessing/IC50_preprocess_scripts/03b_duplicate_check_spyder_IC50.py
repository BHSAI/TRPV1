#!/usr/bin/env python
"""
Preprocess TRPV1 IC50 data - Step 2: Advanced Deduplication (OPTIONAL)

This script performs additional duplicate removal beyond InChIKey:
  1. Removes stereo/isotope duplicates (InChIKey14 - connectivity layer only)
  2. Removes tautomer duplicates (stereo-agnostic tautomer comparison)
  3. Uses majority-vote for conflicting labels

NOTE: This is an OPTIONAL step. The output is NOT used by the scaffold split step.
      It's useful for creating a more strictly deduplicated dataset.

Inputs:
  - data/intermediate/TRPV1_IC50_cleaned_RDK.csv (from step 1)

Outputs:
  - data/intermediate/TRPV1_IC50_cleaned_RDK_dedup.csv (further deduplicated)

Usage:
    # Dry-run (preview only, no output file)
    python 03b_duplicate_check_spyder_IC50.py

    # Apply changes and save
    python 03b_duplicate_check_spyder_IC50.py --apply
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

# Add repository root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import shared utilities
from code.utils.config import get_standardized_file, get_deduplicated_file
from code.utils.deduplication import deduplicate_full_pipeline

# ============================================================================
# Configuration
# ============================================================================

# Input/output paths
INPUT_CSV = get_standardized_file("IC50")
OUTPUT_CSV = get_deduplicated_file("IC50")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ============================================================================
# Main Processing
# ============================================================================

def main(apply=False):
    """
    Run advanced deduplication pipeline.

    Args:
        apply: If True, save output file. If False, dry-run only.
    """
    # Load data
    logging.info(f"Loading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logging.info(f"Loaded {len(df):,} rows")

    # Run full deduplication pipeline
    logging.info("Running deduplication pipeline...")
    df_dedup, stats = deduplicate_full_pipeline(df, label_col="CLASS")

    # Log statistics
    logging.info(f"Removed {stats['stereo_isotope_duplicates']:,} stereo/isotope duplicates (InChIKey14)")
    logging.info(f"Removed {stats['tautomer_duplicates']:,} tautomer duplicates (stereo-agnostic)")
    logging.info(f"Total removed: {stats['total_removed']:,}")
    logging.info(f"Final row count: {len(df_dedup):,}")

    # Save output if apply=True
    if apply:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_dedup.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Saved deduplicated data to: {OUTPUT_CSV}")
    else:
        logging.info("DRY-RUN mode: No file written. Use --apply to save changes.")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced deduplication for TRPV1 IC50 data (optional step)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes and write output file (default: dry-run only)"
    )

    args = parser.parse_args()
    main(apply=args.apply)
