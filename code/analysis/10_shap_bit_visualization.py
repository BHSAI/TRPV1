#!/usr/bin/env python
"""
SHAP Bit Visualization - Morgan Fingerprint Substructures

Visualizes important Morgan fingerprint bits as highlighted substructures
in molecule drawings. Creates SVG images showing which atoms contribute
to each important bit.

Outputs:
- SVG files for each (bit, example molecule) combination
- Bit frequency report (.csv)

Usage:
    python 10_shap_bit_visualization.py --endpoint IC50 --bits 1665 843 1019
    python 10_shap_bit_visualization.py --endpoint EC50 --top-features-file IC50_SHAP_top_features.csv
    python 10_shap_bit_visualization.py --endpoint IC50 --dataset external --examples-per-bit 5
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import (
    get_train_file, get_test_file, get_results_dir,
    get_figure_path, COL_SMILES, MORGAN_RADIUS, MORGAN_FP_SIZE
)
from code.utils.fingerprints import smiles_to_mols
from code.utils.shap_utils import (
    find_example_molecules_for_bit,
    draw_mol_with_highlight,
    calculate_bit_frequency
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def visualize_morgan_bits(endpoint, bit_ids, dataset="external",
                         examples_per_bit=3, img_size=(400, 300)):
    """
    Visualize Morgan fingerprint bits as highlighted substructures.

    Args:
        endpoint: IC50 or EC50
        bit_ids: List of Morgan bit indices to visualize
        dataset: "train" or "external"
        examples_per_bit: Number of example molecules per bit
        img_size: Image size (width, height)
    """

    logging.info(f"=" * 70)
    logging.info(f"SHAP BIT VISUALIZATION: {endpoint} - {dataset}")
    logging.info(f"=" * 70)

    # Load data
    if dataset == "train":
        data_file = get_train_file(endpoint)
    elif dataset == "external":
        data_file = get_test_file(endpoint)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    logging.info(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    if COL_SMILES not in df.columns:
        raise ValueError(f"Column '{COL_SMILES}' not found in {data_file}")

    smiles_list = df[COL_SMILES].dropna().tolist()
    logging.info(f"Loaded {len(smiles_list)} SMILES")

    # Convert to molecules
    mols, valid_idx = smiles_to_mols(smiles_list, return_indices=True)
    logging.info(f"Valid molecules: {len(mols)}")

    # Create output directory
    out_dir = get_figure_path(endpoint, f"SHAP_bits_{dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each bit
    all_frequencies = []

    for bit in bit_ids:
        logging.info(f"\nProcessing Bit {bit}...")

        # Find example molecules
        examples = find_example_molecules_for_bit(
            mols,
            bit,
            max_examples=examples_per_bit,
            radius=MORGAN_RADIUS,
            n_bits=MORGAN_FP_SIZE
        )

        if not examples:
            logging.warning(f"  Bit {bit} not found in any molecule")
            continue

        # Save visualizations
        for i, (mol_idx, hl_atoms) in enumerate(examples, 1):
            out_file = out_dir / f"bit{bit}_example{i}_idx{mol_idx}.svg"
            legend = f"Bit {bit} | Example {i} | Mol {mol_idx}"

            draw_mol_with_highlight(
                mols[mol_idx],
                hl_atoms,
                out_file,
                legend=legend,
                img_size=img_size
            )

            logging.info(f"  Saved: {out_file.name}")

        # Calculate frequency
        count, freq = calculate_bit_frequency(
            mols,
            bit,
            radius=MORGAN_RADIUS,
            n_bits=MORGAN_FP_SIZE
        )

        logging.info(f"  Frequency: {count} molecules ({freq:.2%} of {dataset} set)")

        all_frequencies.append({
            "Bit": bit,
            "Count": count,
            "Frequency": freq,
            "Dataset": dataset
        })

    # Save frequency report
    freq_df = pd.DataFrame(all_frequencies)
    freq_file = out_dir / f"{endpoint}_bit_frequencies_{dataset}.csv"
    freq_df.to_csv(freq_file, index=False)

    logging.info(f"\nSaved frequency report to: {freq_file}")
    logging.info(f"All SVG files saved to: {out_dir}")

    return freq_df

def main(endpoint, bit_ids=None, top_features_file=None,
         dataset="external", examples_per_bit=3, img_size=(400, 300)):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - SHAP BIT VISUALIZATION")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Examples per bit: {examples_per_bit}")

    # Get bit IDs
    if bit_ids is None and top_features_file is None:
        logging.error("Must provide either --bits or --top-features-file")
        sys.exit(1)

    if top_features_file:
        # Load from SHAP top features file
        results_dir = get_results_dir(endpoint)
        feat_file = results_dir / top_features_file

        if not feat_file.exists():
            logging.error(f"Top features file not found: {feat_file}")
            logging.error("Run 09_shap_analysis.py first")
            sys.exit(1)

        logging.info(f"Loading top features from: {feat_file}")
        feat_df = pd.read_csv(feat_file)

        # Extract bit IDs
        bit_ids = [
            int(feat.replace("Bit_", ""))
            for feat in feat_df["Feature"]
            if feat.startswith("Bit_")
        ]

        logging.info(f"Loaded {len(bit_ids)} bit IDs from file")

    logging.info(f"Visualizing {len(bit_ids)} bits: {bit_ids[:10]}...")

    try:
        visualize_morgan_bits(
            endpoint,
            bit_ids,
            dataset=dataset,
            examples_per_bit=examples_per_bit,
            img_size=img_size
        )
    except Exception as e:
        logging.error(f"Bit visualization failed: {e}")
        raise

    logging.info("\n" + "=" * 70)
    logging.info("SHAP BIT VISUALIZATION COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize important Morgan bits as highlighted substructures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specify bits manually
  python 10_shap_bit_visualization.py --endpoint IC50 --bits 1665 843 1019

  # Load bits from SHAP analysis output
  python 10_shap_bit_visualization.py --endpoint IC50 --top-features-file IC50_SHAP_top_features.csv

  # Visualize on training set instead of external
  python 10_shap_bit_visualization.py --endpoint IC50 --bits 1665 843 --dataset train

  # More examples per bit
  python 10_shap_bit_visualization.py --endpoint IC50 --bits 1665 843 --examples-per-bit 5
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=None,
        help="Morgan bit IDs to visualize (e.g., --bits 1665 843 1019)"
    )

    parser.add_argument(
        "--top-features-file",
        type=str,
        default=None,
        help="Load bits from SHAP top features CSV (e.g., IC50_SHAP_top_features.csv)"
    )

    parser.add_argument(
        "--dataset",
        choices=["train", "external"],
        default="external",
        help="Dataset to use for visualization (default: external)"
    )

    parser.add_argument(
        "--examples-per-bit",
        type=int,
        default=3,
        help="Number of example molecules per bit (default: 3)"
    )

    parser.add_argument(
        "--img-width",
        type=int,
        default=400,
        help="Image width in pixels (default: 400)"
    )

    parser.add_argument(
        "--img-height",
        type=int,
        default=300,
        help="Image height in pixels (default: 300)"
    )

    args = parser.parse_args()

    main(
        args.endpoint,
        bit_ids=args.bits,
        top_features_file=args.top_features_file,
        dataset=args.dataset,
        examples_per_bit=args.examples_per_bit,
        img_size=(args.img_width, args.img_height)
    )
