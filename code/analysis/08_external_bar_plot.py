#!/usr/bin/env python
"""
External Test Bar Plot

Creates publication-ready bar plot of external test performance
for the best model.

Usage:
    python 08_external_bar_plot.py --endpoint IC50
    python 08_external_bar_plot.py --endpoint EC50
    python 08_external_bar_plot.py --endpoint IC50 --model Morgan_LogisticRegression
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, get_figure_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

METRICS = ["MCC", "GMean", "ROC_AUC", "PR_AUC"]
METRIC_LABELS = ["MCC", "G-mean", "ROC-AUC", "PR-AUC"]

def create_external_bar_plot(endpoint, model_name="Morgan_LogisticRegression"):
    """Create bar plot for external test results."""

    logging.info(f"Creating external bar plot for {endpoint} - {model_name}")

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    results_dir = get_results_dir(endpoint)

    possible_files = [
        results_dir / f"{endpoint}_external_test_metrics.csv",
        results_dir / f"{endpoint}_champions_external_metrics.csv",
        results_dir / f"TRPV1_{endpoint}_external_test_results.csv"
    ]

    stats_csv = None
    for f in possible_files:
        if f.exists():
            stats_csv = f
            break

    if stats_csv is None:
        logging.error(f"No external test metrics file found in {results_dir}")
        logging.error("Please run external test evaluation first")
        return None

    logging.info(f"Loading results from: {stats_csv}")
    df = pd.read_csv(stats_csv)

    if 'Method' in df.columns:
        row = df[df["Method"] == model_name]
        if row.empty:
            logging.warning(f"Model {model_name} not found, using first row")
            row = df.iloc[0:1]
    elif 'Model' in df.columns and 'Fingerprint' in df.columns:
        fp_type, model_type = model_name.split("_", 1)
        row = df[(df["Fingerprint"] == fp_type) & (df["Model"] == model_type)]
        if row.empty:
            logging.warning(f"Model {model_name} not found, using first row")
            row = df.iloc[0:1]
    else:
        row = df.iloc[0:1]

    row = row.iloc[0]

    values = []
    for m in METRICS:
        if m in row:
            values.append(float(row[m]))
        else:
            values.append(0.0)
            logging.warning(f"Metric {m} not found in data")

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    x = range(len(METRICS))
    bars = ax.bar(
        x,
        values,
        width=0.6,
        color="#4C72B0",
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{endpoint} External Test - {model_name.replace('_', ' ')}",
                 fontsize=13, fontweight='bold')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()

    png_path = get_figure_path(endpoint, f"{endpoint}_external_{model_name}_bars.png")
    pdf_path = get_figure_path(endpoint, f"{endpoint}_external_{model_name}_bars.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved bar plots to:")
    logging.info(f"  {png_path}")
    logging.info(f"  {pdf_path}")

    return values

def main(endpoint, model_name=None):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - EXTERNAL BAR PLOT")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")

    if model_name is None:
        model_name = "Morgan_LogisticRegression"

    create_external_bar_plot(endpoint, model_name)

    logging.info("\n" + "=" * 70)
    logging.info("EXTERNAL BAR PLOT COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create external test bar plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 08_external_bar_plot.py --endpoint IC50
  python 08_external_bar_plot.py --endpoint EC50
  python 08_external_bar_plot.py --endpoint IC50 --model Morgan_LogisticRegression
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: Morgan_LogisticRegression)"
    )

    args = parser.parse_args()
    main(args.endpoint, args.model)
