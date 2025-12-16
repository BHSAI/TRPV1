#!/usr/bin/env python
"""
Generate Master Results Table

Aggregates mean metrics across all CV folds for all fingerprints and models.
Creates a comprehensive table with mean ± std for each metric.

Usage:
    python 07_generate_master_table.py --endpoint IC50
    python 07_generate_master_table.py --endpoint EC50
"""

import sys
import logging
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, MODEL_ORDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

METRICS = ["ROC_AUC", "PR_AUC", "MCC", "GMean", "F1", "Accuracy",
           "Sensitivity", "Specificity", "Precision", "NPV", "FPR", "FNR"]

def generate_master_table(endpoint, fingerprint_types=None):
    """Generate master table of mean metrics."""

    logging.info(f"Generating master table for {endpoint}")

    if fingerprint_types is None:
        fingerprint_types = ["RDKITfp", "Morgan", "MACCS"]

    all_results = []

    for fp_type in fingerprint_types:
        results_dir = get_results_dir(endpoint, fp_type)
        data_file = results_dir / f"{endpoint}_{fp_type}_per_fold_metrics_5x5.csv"

        if not data_file.exists():
            logging.warning(f"Skipping {fp_type}: file not found")
            continue

        logging.info(f"Processing {fp_type}...")

        df = pd.read_csv(data_file)

        for model in MODEL_ORDER:
            model_data = df[df['Model'] == model]

            if len(model_data) == 0:
                continue

            result = {
                'Fingerprint': fp_type,
                'Model': model,
                'N_folds': len(model_data)
            }

            for metric in METRICS:
                if metric in model_data.columns:
                    result[f"{metric}_mean"] = model_data[metric].mean()
                    result[f"{metric}_std"] = model_data[metric].std()

            all_results.append(result)

    master_df = pd.DataFrame(all_results)

    output_dir = get_results_dir(endpoint)
    output_file = output_dir / f"{endpoint}_master_table_mean_metrics.csv"
    master_df.to_csv(output_file, index=False)

    logging.info(f"\nMaster table shape: {master_df.shape}")
    logging.info(f"Saved master table to: {output_file}")

    logging.info("\nTop 5 models by ROC-AUC:")
    if 'ROC_AUC_mean' in master_df.columns:
        top_models = master_df.nlargest(5, 'ROC_AUC_mean')[['Fingerprint', 'Model', 'ROC_AUC_mean', 'MCC_mean']]
        for _, row in top_models.iterrows():
            logging.info(f"  {row['Fingerprint']:10s} {row['Model']:20s}: "
                        f"ROC-AUC={row['ROC_AUC_mean']:.3f}, MCC={row['MCC_mean']:.3f}")

    return master_df

def main(endpoint, fingerprint_types=None):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - MASTER TABLE GENERATION")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")

    generate_master_table(endpoint, fingerprint_types)

    logging.info("\n" + "=" * 70)
    logging.info("MASTER TABLE GENERATION COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate master results table with mean metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 07_generate_master_table.py --endpoint IC50
  python 07_generate_master_table.py --endpoint EC50
  python 07_generate_master_table.py --endpoint IC50 --fingerprints Morgan
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    parser.add_argument(
        "--fingerprints",
        nargs="+",
        choices=["RDKITfp", "Morgan", "MACCS", "AtomPair"],
        default=None,
        help="Fingerprint types to include (default: RDKITfp Morgan MACCS)"
    )

    args = parser.parse_args()
    main(args.endpoint, args.fingerprints)
