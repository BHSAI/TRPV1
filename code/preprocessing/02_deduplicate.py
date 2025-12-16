#!/usr/bin/env python
"""
Advanced deduplication for TRPV1 data - OPTIONAL STEP (unified IC50/EC50 processing)

This script performs additional duplicate removal beyond InChIKey:
  1. Removes stereo/isotope duplicates (InChIKey14 - connectivity layer only)
  2. Removes tautomer duplicates (stereo-agnostic tautomer comparison)
  3. Uses majority-vote for conflicting labels

NOTE: This is an OPTIONAL step. The output is NOT used by the scaffold split step.
      It's useful for creating a more strictly deduplicated dataset.

Inputs:
  - data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv (from step 1)

Outputs:
  - data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK_dedup.csv (further deduplicated)

Usage:
    # Dry-run (preview only, no output file)
    python 02_deduplicate.py --endpoint IC50

    # Apply changes and save
    python 02_deduplicate.py --endpoint IC50 --apply
    python 02_deduplicate.py --endpoint EC50 --apply
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
from code.utils.config import get_standardized_file, get_deduplicated_file
from code.utils.deduplication import deduplicate_full_pipeline

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

def deduplicate(endpoint, apply=False):
    """
    Run advanced deduplication pipeline for specified endpoint.

    Args:
        endpoint: 'IC50' or 'EC50'
        apply: If True, save output file. If False, dry-run only.
    """
    # Get input/output paths
    input_csv = get_standardized_file(endpoint)
    output_csv = get_deduplicated_file(endpoint)

    logging.info(f"=" * 70)
    logging.info(f"STEP 2: Advanced Deduplication (OPTIONAL) - {endpoint}")
    logging.info(f"=" * 70)

    # Load data
    logging.info(f"Loading data from: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
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
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_dedup.to_csv(output_csv, index=False)
        logging.info(f"Saved deduplicated data to: {output_csv}")
    else:
        logging.info("DRY-RUN mode: No file written. Use --apply to save changes.")

    logging.info("")
    return len(df_dedup) if apply else None


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced deduplication for TRPV1 data (optional step)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (preview only)
  python 02_deduplicate.py --endpoint IC50

  # Apply changes
  python 02_deduplicate.py --endpoint IC50 --apply
  python 02_deduplicate.py --endpoint EC50 --apply
        """
    )
    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process (IC50 or EC50)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes and write output file (default: dry-run only)"
    )

    args = parser.parse_args()
    deduplicate(args.endpoint, apply=args.apply)


if __name__ == "__main__":
    main()
