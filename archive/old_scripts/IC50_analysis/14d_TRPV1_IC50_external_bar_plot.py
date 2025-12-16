# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 23:11:51 2025

@author: mabdulhameed
"""

#!/usr/bin/env python
"""
07_TRPV1_IC50_external_bars_MorganLR.py

Create a publication-ready bar plot of external scaffold-test
performance for the Morgan + LogisticRegression model (IC50).

Metrics shown:
  - MCC
  - G-mean
  - ROC-AUC
  - PR-AUC
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ---------- CONFIG ----------
STATS_CSV = Path("TRPV1_IC50_champions_external_metrics.csv")

OUT_PNG = Path("TRPV1_IC50_external_MorganLR_bars.png")
OUT_PDF = Path("TRPV1_IC50_external_MorganLR_bars.pdf")

# Metrics to plot (column names in the CSV)
METRICS = ["MCC", "GMean", "ROC_AUC", "PR_AUC"]
METRIC_LABELS = ["MCC", "G-mean", "ROC-AUC", "PR-AUC"]

# Fonts
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 12

# ---------- LOAD DATA ----------
df = pd.read_csv(STATS_CSV)

row = df[df["Method"] == "Morgan_LR"]
if row.empty:
    raise ValueError("No row with Method == 'Morgan_LR' found in the CSV.")

row = row.iloc[0]

values = [float(row[m]) for m in METRICS]

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(4.0, 3.5))

x = range(len(METRICS))
bars = ax.bar(
    x,
    values,
    width=0.6,
    color="#4C72B0",   # muted blue
    edgecolor="black",
    linewidth=1.0,
)

ax.set_xticks(x)
ax.set_xticklabels(METRIC_LABELS, rotation=45, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate bars with numeric values
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.02,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

fig.tight_layout()

# ---------- SAVE ----------
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print(f"Saved external bar plots to:\n  {OUT_PNG}\n  {OUT_PDF}")
