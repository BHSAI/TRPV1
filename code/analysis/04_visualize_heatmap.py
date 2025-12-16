#!/usr/bin/env python
"""
Visualize Results - Heatmap

Creates publication-quality heatmaps showing model performance
across different fingerprints for each metric.

Usage:
    python 04_visualize_heatmap.py --endpoint IC50
    python 04_visualize_heatmap.py --endpoint EC50
    python 04_visualize_heatmap.py --endpoint IC50 --metrics ROC_AUC MCC
"""

import sys
import logging
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, get_figure_path
from code.utils.visualization import plot_heatmap, set_publication_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

METRICS = ["ROC_AUC", "MCC", "GMean", "F1", "Accuracy"]

def create_heatmap_visualization(endpoint, metric):
    """Create heatmap for specified metric."""

    logging.info(f"Creating heatmap for {endpoint} - {metric}")

    set_publication_style()

    results_dir = get_results_dir(endpoint)
    master_file = results_dir / f"{endpoint}_master_table_mean_metrics.csv"

    if not master_file.exists():
        logging.error(f"Master table not found: {master_file}")
        logging.error("Please run 07_generate_master_table.py first")
        return None

    df = pd.read_csv(master_file)

    metric_col = f"{metric}_mean"
    if metric_col not in df.columns:
        logging.warning(f"Metric {metric} not found in master table")
        return None

    pivot_data = df.pivot(
        index='Model',
        columns='Fingerprint',
        values=metric_col
    )

    fig_path = get_figure_path(endpoint, f"{metric.lower()}_heatmap.png")

    plot_heatmap(
        pivot_data,
        title=f"{endpoint} - {metric} Comparison",
        xlabel="Fingerprint",
        ylabel="Model",
        cmap="RdYlGn",
        annot=True,
        fmt=".3f",
        figsize=(10, 8),
        vmin=pivot_data.min().min(),
        vmax=pivot_data.max().max(),
        cbar_label=metric,
        save_path=fig_path
    )

    logging.info(f"Saved heatmap to: {fig_path}")

    return pivot_data

def main(endpoint, metrics=None):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - HEATMAP VISUALIZATION")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")

    if metrics is None:
        metrics = METRICS

    for metric in metrics:
        try:
            create_heatmap_visualization(endpoint, metric)
        except Exception as e:
            logging.error(f"Failed to create heatmap for {metric}: {e}")
            continue

    logging.info("\n" + "=" * 70)
    logging.info("HEATMAP VISUALIZATION COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create heatmap visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_visualize_heatmap.py --endpoint IC50
  python 04_visualize_heatmap.py --endpoint EC50
  python 04_visualize_heatmap.py --endpoint IC50 --metrics ROC_AUC MCC
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["ROC_AUC", "PR_AUC", "MCC", "GMean", "F1", "Accuracy",
                 "Sensitivity", "Specificity"],
        default=None,
        help="Metrics to visualize (default: ROC_AUC MCC GMean F1 Accuracy)"
    )

    args = parser.parse_args()
    main(args.endpoint, args.metrics)
