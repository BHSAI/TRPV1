#!/usr/bin/env python
"""
Visualize Results - Dashboard

Creates a comprehensive multi-panel dashboard showing:
- ROC-AUC heatmap
- MCC heatmap
- Box plots for key metrics
- Model rankings

Usage:
    python 06_visualize_dashboard.py --endpoint IC50
    python 06_visualize_dashboard.py --endpoint EC50
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, get_figure_path, MODEL_ORDER
from code.utils.visualization import set_publication_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def create_dashboard(endpoint):
    """Create comprehensive dashboard visualization."""

    logging.info(f"Creating dashboard for {endpoint}")

    set_publication_style()

    results_dir = get_results_dir(endpoint)
    master_file = results_dir / f"{endpoint}_master_table_mean_metrics.csv"

    if not master_file.exists():
        logging.error(f"Master table not found: {master_file}")
        logging.error("Please run 07_generate_master_table.py first")
        return None

    df = pd.read_csv(master_file)

    fig = plt.figure(figsize=(18, 12))

    roc_pivot = df.pivot(index='Model', columns='Fingerprint', values='ROC_AUC_mean')
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(roc_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, ax=ax1, cbar_kws={'label': 'ROC-AUC'})
    ax1.set_title(f"{endpoint} - ROC-AUC", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Fingerprint")
    ax1.set_ylabel("Model")

    if 'MCC_mean' in df.columns:
        mcc_pivot = df.pivot(index='Model', columns='Fingerprint', values='MCC_mean')
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(mcc_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, ax=ax2, cbar_kws={'label': 'MCC'})
        ax2.set_title(f"{endpoint} - MCC", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Fingerprint")
        ax2.set_ylabel("Model")

    if 'GMean_mean' in df.columns:
        gmean_pivot = df.pivot(index='Model', columns='Fingerprint', values='GMean_mean')
        ax3 = plt.subplot(2, 3, 3)
        sns.heatmap(gmean_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0.5, vmax=1.0, ax=ax3, cbar_kws={'label': 'G-Mean'})
        ax3.set_title(f"{endpoint} - G-Mean", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Fingerprint")
        ax3.set_ylabel("Model")

    ax4 = plt.subplot(2, 3, 4)
    top_models = df.nlargest(10, 'ROC_AUC_mean')[['Model', 'Fingerprint', 'ROC_AUC_mean']]
    top_models['Label'] = top_models['Model'] + '\n(' + top_models['Fingerprint'] + ')'
    ax4.barh(range(len(top_models)), top_models['ROC_AUC_mean'], color='steelblue')
    ax4.set_yticks(range(len(top_models)))
    ax4.set_yticklabels(top_models['Label'], fontsize=9)
    ax4.set_xlabel('ROC-AUC', fontsize=11)
    ax4.set_title('Top 10 Models by ROC-AUC', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()

    if 'MCC_mean' in df.columns:
        ax5 = plt.subplot(2, 3, 5)
        top_mcc = df.nlargest(10, 'MCC_mean')[['Model', 'Fingerprint', 'MCC_mean']]
        top_mcc['Label'] = top_mcc['Model'] + '\n(' + top_mcc['Fingerprint'] + ')'
        ax5.barh(range(len(top_mcc)), top_mcc['MCC_mean'], color='coral')
        ax5.set_yticks(range(len(top_mcc)))
        ax5.set_yticklabels(top_mcc['Label'], fontsize=9)
        ax5.set_xlabel('MCC', fontsize=11)
        ax5.set_title('Top 10 Models by MCC', fontsize=14, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        ax5.invert_yaxis()

    ax6 = plt.subplot(2, 3, 6)
    metric_means = {
        'ROC-AUC': df['ROC_AUC_mean'].mean(),
    }
    if 'MCC_mean' in df.columns:
        metric_means['MCC'] = df['MCC_mean'].mean()
    if 'GMean_mean' in df.columns:
        metric_means['G-Mean'] = df['GMean_mean'].mean()
    if 'F1_mean' in df.columns:
        metric_means['F1'] = df['F1_mean'].mean()

    ax6.bar(metric_means.keys(), metric_means.values(), color=['steelblue', 'coral', 'seagreen', 'mediumpurple'])
    ax6.set_ylabel('Mean Value', fontsize=11)
    ax6.set_title('Overall Metric Averages', fontsize=14, fontweight='bold')
    ax6.set_ylim(0, 1.0)
    ax6.grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(metric_means.items()):
        ax6.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle(f'TRPV1 {endpoint} - ML Benchmark Dashboard',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout()

    fig_path = get_figure_path(endpoint, f"{endpoint}_dashboard.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved dashboard to: {fig_path}")

    return fig

def main(endpoint):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - DASHBOARD VISUALIZATION")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")

    create_dashboard(endpoint)

    logging.info("\n" + "=" * 70)
    logging.info("DASHBOARD VISUALIZATION COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create comprehensive dashboard visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 06_visualize_dashboard.py --endpoint IC50
  python 06_visualize_dashboard.py --endpoint EC50
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    args = parser.parse_args()
    main(args.endpoint)
