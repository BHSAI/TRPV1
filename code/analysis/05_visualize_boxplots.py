#!/usr/bin/env python
"""
Visualize Results - Box Plots

Creates box plot visualizations showing distribution of metrics
across CV folds for each model and fingerprint.

Usage:
    python 05_visualize_boxplots.py --endpoint IC50
    python 05_visualize_boxplots.py --endpoint EC50
    python 05_visualize_boxplots.py --endpoint IC50 --fingerprint Morgan
"""

import sys
import logging
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, get_figure_path, MODEL_ORDER
from code.utils.visualization import plot_boxplots, set_publication_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

METRICS = ["ROC_AUC", "MCC", "GMean"]

def create_boxplot_visualization(endpoint, fingerprint_type, metric):
    """Create box plot for specified fingerprint and metric."""

    logging.info(f"Creating boxplot for {endpoint} - {fingerprint_type} - {metric}")

    set_publication_style()

    results_dir = get_results_dir(endpoint, fingerprint_type)
    data_file = results_dir / f"{endpoint}_{fingerprint_type}_per_fold_metrics_5x5.csv"

    if not data_file.exists():
        logging.warning(f"Data file not found: {data_file}")
        return None

    df = pd.read_csv(data_file)

    if metric not in df.columns:
        logging.warning(f"Metric {metric} not found in data")
        return None

    fig_path = get_figure_path(endpoint, f"{fingerprint_type}_{metric.lower()}_boxplot.png")

    plot_boxplots(
        data=df,
        x='Model',
        y=metric,
        title=f"{endpoint} - {fingerprint_type} - {metric} Distribution",
        xlabel="Model",
        ylabel=metric,
        figsize=(14, 6),
        palette="Set2",
        order=MODEL_ORDER,
        save_path=fig_path
    )

    logging.info(f"Saved boxplot to: {fig_path}")

    return df

def create_combined_boxplot(endpoint, metric):
    """Create combined box plot across all fingerprints."""

    logging.info(f"Creating combined boxplot for {endpoint} - {metric}")

    set_publication_style()

    fingerprint_types = ["RDKITfp", "Morgan", "MACCS"]
    all_data = []

    for fp_type in fingerprint_types:
        results_dir = get_results_dir(endpoint, fp_type)
        data_file = results_dir / f"{endpoint}_{fp_type}_per_fold_metrics_5x5.csv"

        if data_file.exists():
            df = pd.read_csv(data_file)
            if metric in df.columns:
                all_data.append(df)

    if not all_data:
        logging.warning(f"No data found for {metric}")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)

    fig_path = get_figure_path(endpoint, f"all_fingerprints_{metric.lower()}_boxplot.png")

    plot_boxplots(
        data=combined_df,
        x='Model',
        y=metric,
        hue='Fingerprint',
        title=f"{endpoint} - {metric} Across All Fingerprints",
        xlabel="Model",
        ylabel=metric,
        figsize=(16, 7),
        palette="Set2",
        order=MODEL_ORDER,
        save_path=fig_path
    )

    logging.info(f"Saved combined boxplot to: {fig_path}")

    return combined_df

def main(endpoint, fingerprint_type=None, metrics=None):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - BOXPLOT VISUALIZATION")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")

    if metrics is None:
        metrics = METRICS

    if fingerprint_type:
        for metric in metrics:
            try:
                create_boxplot_visualization(endpoint, fingerprint_type, metric)
            except Exception as e:
                logging.error(f"Failed for {fingerprint_type} {metric}: {e}")
                continue
    else:
        for metric in metrics:
            try:
                create_combined_boxplot(endpoint, metric)
            except Exception as e:
                logging.error(f"Failed for combined {metric}: {e}")
                continue

        fingerprint_types = ["RDKITfp", "Morgan", "MACCS"]
        for fp_type in fingerprint_types:
            for metric in metrics:
                try:
                    create_boxplot_visualization(endpoint, fp_type, metric)
                except Exception as e:
                    logging.error(f"Failed for {fp_type} {metric}: {e}")
                    continue

    logging.info("\n" + "=" * 70)
    logging.info("BOXPLOT VISUALIZATION COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create box plot visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_visualize_boxplots.py --endpoint IC50
  python 05_visualize_boxplots.py --endpoint EC50
  python 05_visualize_boxplots.py --endpoint IC50 --fingerprint Morgan
  python 05_visualize_boxplots.py --endpoint IC50 --metrics ROC_AUC MCC
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    parser.add_argument(
        "--fingerprint",
        choices=["RDKITfp", "Morgan", "MACCS", "AtomPair"],
        default=None,
        help="Specific fingerprint type (default: all)"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["ROC_AUC", "PR_AUC", "MCC", "GMean", "F1", "Accuracy"],
        default=None,
        help="Metrics to visualize (default: ROC_AUC MCC GMean)"
    )

    args = parser.parse_args()
    main(args.endpoint, args.fingerprint, args.metrics)
