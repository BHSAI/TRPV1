# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 21:40:26 2025

@author: mabdulhameed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12


# ----------------- CONFIG -----------------
STATS_DIR = Path("TRPV1_EC50_5x5cv_stats")

# Per-fold 5×5 CV metrics for each fingerprint
# (update paths/filenames here if yours differ)
PER_FOLD_FILES = {
    "Morgan":  Path("TRPV1_EC50_5x5cv_stats/EC50_Morgan_rand_per_fold_metrics_5x5.csv"),
    "RDKit":   Path("TRPV1_EC50_5x5cv_stats/EC50_RDKITfp_rand_per_fold_metrics_5x5.csv"),
    "MACCS":   Path("TRPV1_EC50_5x5cv_stats/EC50_MACCS_rand_per_fold_metrics_5x5.csv"),
    "Mordred": Path("TRPV1_EC50_5x5cv_stats/EC50_Mordred_scaffCV_per_fold_metrics_5x5.csv"),
}

METRICS = ["MCC", "GMean"]

METRIC_LABELS = {
    "MCC": "MCC",
    "GMean": "G-Mean",
}

MODEL_ORDER = [
    "LogisticRegression",
    "XGBoost",
    "LightGBM",
    "SVM",
    "RandomForest",
    "KNN",
    "Bayesian",
]

OUT_PNG = STATS_DIR / "EC50_S2_MCC_GMean_boxplots_5x5CV.png"
OUT_PDF = STATS_DIR / "EC50_S2_MCC_GMean_boxplots_5x5CV.pdf"

# Optional styling to approximate journal look
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

sns.set_style("white")
sns.set_context("paper")

# ----------------- LOAD & CONCAT -----------------

all_rows = []
for desc, csv_path in PER_FOLD_FILES.items():
    if not csv_path.exists():
        print(f"[WARN] Missing file for {desc}: {csv_path} – skipping.")
        continue

    df = pd.read_csv(csv_path)
    df["Descriptor"] = desc

    # keep only models we care about, in consistent order
    df = df[df["Model"].isin(MODEL_ORDER)].copy()
    df["Model"] = pd.Categorical(df["Model"],
                                 categories=MODEL_ORDER,
                                 ordered=True)
    df = df.sort_values("Model")

    all_rows.append(df)

if not all_rows:
    raise RuntimeError("No per-fold EC50 CV files loaded – check PER_FOLD_FILES paths.")

df_all = pd.concat(all_rows, axis=0, ignore_index=True)

# Boxplot styling
boxprops    = dict(facecolor="#9BBFE0", edgecolor="black")
medianprops = dict(color="black", linewidth=1.2)
whiskerprops = dict(color="black", linewidth=1.0)
capprops     = dict(color="black", linewidth=1.0)
     

# ----------------- PLOT FIGURE S2 -----------------

n_desc = len(PER_FOLD_FILES)
fig, axes = plt.subplots(
    nrows=n_desc,
    ncols=len(METRICS),
    figsize=(10, 10),
    sharey="col",
)

if n_desc == 1:
    axes = np.array([axes])

descriptor_list = list(PER_FOLD_FILES.keys())

for i, desc in enumerate(descriptor_list):
    sub = df_all[df_all["Descriptor"] == desc]

    for j, metric in enumerate(METRICS):
        ax = axes[i, j]

        sns.boxplot(
            data=sub,
            x="Model",
            y=metric,
            order=MODEL_ORDER,
            patch_artist=True,          # important so facecolor is used
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            fliersize=2,
            linewidth=1,
            ax=ax,
        )


        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Column titles on top row only
        label = METRIC_LABELS.get(metric, metric)
        if i == 0:
            ax.set_title(label, fontsize=12, pad=6)
        else:
            ax.set_title("")

        # X-labels only on bottom row
        if i < n_desc - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("") # removed 'model' 
            ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=9)

        # Y-labels: descriptor on left column, blank on right
        if j == 0:
            ax.set_ylabel(desc, fontsize=11)
        else:
            ax.set_ylabel("")

# Add some space on left so descriptor labels are clear
fig.subplots_adjust(left=0.12, bottom=0.15, hspace=0.25, wspace=0.25)

# Save at print quality
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print(f"Figure S2 saved to:\n  {OUT_PNG}\n  {OUT_PDF}")
