# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:42:00 2025

@author: mabdulhameed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reload data
ic50 = pd.read_csv('TRPV1_IC50_5x5cv_stats/IC50_master_mean_scores.csv')
#ec50 = pd.read_csv('/mnt/data/EC50_master_mean_scores.csv')

# Metrics in the order you want them on the x-axis
metrics = ["MCC", "GMean", "Specificity", "ROC_AUC", "PR_AUC", "F1", "Sensitivity"]

# Fixed descriptor and model orders
descriptor_order = ["Morgan", "RDKit", "MACCS", "Mordred"]
model_order = [
    "LogisticRegression",
    "XGBoost",
    "LightGBM",
    "SVM",
    "RandomForest",
    "KNN",
    "Bayesian",
]

def grouped_minmax_heatmap(
    df,
    metrics,
    descriptor_order=None,
    model_order=None,
    add_separators=True,
    cmap_name="viridis",
):
    """
    df: DataFrame with columns ['Descriptor', 'Model'] + metrics
    metrics: list of metric column names to include as columns in the heatmap
    descriptor_order: list of descriptor names in desired block order
    model_order: list of model names in desired row order (same for each block)
    add_separators: if True, insert blank rows between descriptor blocks
    """

    # Subset to what we need
    subset = df[["Descriptor", "Model"] + metrics].copy()

    # If no explicit descriptor order is given, use sorted unique
    if descriptor_order is None:
        descriptor_order = sorted(subset["Descriptor"].unique())

    all_rows = []
    row_labels = []
    block_bounds = []  # (desc, start_idx, end_idx) *before* separator row

    for i, desc in enumerate(descriptor_order):
        block = subset[subset["Descriptor"] == desc].copy()
        if block.empty:
            continue

        # Enforce same model order in each block
        if model_order is not None:
            block["Model"] = pd.Categorical(block["Model"], categories=model_order, ordered=True)
            block = block.sort_values("Model")
        # Drop models not present (if any)
        block = block.dropna(subset=["Model"])

        start_idx = len(all_rows)
        for _, row in block.iterrows():
            all_rows.append(row[metrics].values)
            row_labels.append(row["Model"])
        end_idx = len(all_rows)
        block_bounds.append((desc, start_idx, end_idx))

        # Insert a separator row (NaNs) between descriptor blocks
        if add_separators and i < len(descriptor_order) - 1:
            all_rows.append([np.nan] * len(metrics))
            row_labels.append("")  # blank label for separator

    # Convert to DataFrame
    mm = pd.DataFrame(all_rows, columns=metrics)

    # Min–max scaling per metric, safely handling constant metrics
    mins = mm.min(skipna=True)
    ranges = mm.max(skipna=True) - mins
    ranges[ranges == 0] = 1.0
    mm_scaled = (mm - mins) / ranges

    # Mask NaNs (separator rows) so they render as white
    data = np.ma.masked_invalid(mm_scaled.values)

    # Set up colormap
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="white")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.grid(False)          # turn off grid if it was enabled
    
    # y-axis labels: models (blank for separator rows)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    # x-axis labels: metrics
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")

    # Descriptor labels on the left, centered in each block
    for desc, start, end in block_bounds:
        # compute midpoint row index (ignores separator row, since we stored bounds pre-separator)
        mid = (start + end - 1) / 2.0
        ax.text(
            -2.6,
            mid,
            desc,
            va="center",
            ha="right",
            fontsize=12,
            rotation=90,
        )
        # increase left margin so labels are not cut off
    fig.subplots_adjust(left=0.22)   # tweak 0.2–0.28 as needed
    # Optional: thicker horizontal lines at block boundaries (just above each block)
    # (Use start index; separator row gives additional white space)
    for desc, start, end in block_bounds[1:]:
        ax.axhline(start - 0.5, color="white", linewidth=2)

    # Colorbar with a simple label
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Min–max scaled metric")

    plt.tight_layout()
    plt.show()


# Example calls (assuming ic50 and ec50 master mean-score tables are loaded):
# ic50 = pd.read_csv("IC50_master_mean_scores.csv")
# ec50 = pd.read_csv("EC50_master_mean_scores.csv")

grouped_minmax_heatmap(ic50, metrics, descriptor_order, model_order)
# grouped_minmax_heatmap(ec50, metrics, descriptor_order, model_order)
