#!/usr/bin/env python
"""
Statistical Analysis - Repeated Measures ANOVA & Tukey HSD

Compares ML models using RM-ANOVA and Tukey post-hoc tests.
Generates statistical reports and p-value heatmaps.

Usage:
    python 03_statistical_analysis.py --endpoint IC50
    python 03_statistical_analysis.py --endpoint EC50
    python 03_statistical_analysis.py --endpoint IC50 --fingerprints Morgan RDKITfp
"""

import sys
import logging
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir
from code.utils.stats import compare_models_rm_anova
from code.utils.visualization import plot_pvalue_heatmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

METRICS = ["ROC_AUC", "MCC", "GMean"]

def run_statistical_analysis(endpoint, fingerprint_type, metric):
    """Run RM-ANOVA and Tukey HSD for specified combination."""

    logging.info(f"Analyzing {endpoint} - {fingerprint_type} - {metric}")

    results_dir = get_results_dir(endpoint, fingerprint_type)
    data_file = results_dir / f"{endpoint}_{fingerprint_type}_per_fold_metrics_5x5.csv"

    if not data_file.exists():
        logging.warning(f"File not found: {data_file}")
        return None

    df = pd.read_csv(data_file)

    try:
        stats_results = compare_models_rm_anova(
            df, metric_col=metric, model_col='Model',
            subject_col='Subject', alpha=0.05
        )

        output_dir = get_results_dir(endpoint)

        anova_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_ANOVA.txt"
        with open(anova_file, 'w') as f:
            f.write(str(stats_results['anova']))

        tukey_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_Tukey.csv"
        stats_results['tukey'].to_csv(tukey_file, index=False)

        cld_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_CLD.csv"
        stats_results['cld'].to_csv(cld_file, header=True)

        fig_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_pvalue_heatmap.png"
        plot_pvalue_heatmap(
            stats_results['pairwise_matrix'],
            title=f"{endpoint} {fingerprint_type} - {metric}",
            save_path=fig_file
        )

        logging.info(f"  ANOVA p-value: {stats_results['anova'].anova_table['Pr > F'][0]:.4f}")
        logging.info(f"  Saved statistical results to: {output_dir}")

        return stats_results

    except Exception as e:
        logging.error(f"Failed for {endpoint} {fingerprint_type} {metric}: {e}")
        return None

def main(endpoint, fingerprint_types=None):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - STATISTICAL ANALYSIS")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")

    if fingerprint_types is None:
        fingerprint_types = ["RDKITfp", "Morgan", "MACCS"]

    for fp_type in fingerprint_types:
        logging.info(f"\n{'=' * 70}")
        logging.info(f"Fingerprint: {fp_type}")
        logging.info(f"{'=' * 70}")

        for metric in METRICS:
            run_statistical_analysis(endpoint, fp_type, metric)

    logging.info("\n" + "=" * 70)
    logging.info("STATISTICAL ANALYSIS COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Statistical analysis with RM-ANOVA and Tukey HSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_statistical_analysis.py --endpoint IC50
  python 03_statistical_analysis.py --endpoint EC50
  python 03_statistical_analysis.py --endpoint IC50 --fingerprints Morgan
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
        help="Fingerprint types to analyze (default: RDKITfp Morgan MACCS)"
    )

    args = parser.parse_args()
    main(args.endpoint, args.fingerprints)
