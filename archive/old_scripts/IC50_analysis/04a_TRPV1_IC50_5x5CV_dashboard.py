# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:52:26 2025

@author: mabdulhameed
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────── GLOBAL STYLE ──────────────
# Use Arial, 12 pt (falls back silently if Arial is not installed)
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
})

# Clean white background (no grey gridlines)
sns.set_style("white")

# ────────────── CONFIG ──────────────

TASK = "IC50"

# Folder where you put all 5×5 CV statistics & Tukey outputs
DATA_DIR = Path("TRPV1_IC50_5x5cv_stats")

# Per-fold 5×5 CV metric files for each fingerprint
FP_FILES = {
    "RDKITfp": "IC50_RDKITfp_rand_per_fold_metrics_5x5.csv",
    "MACCS":   "IC50_MACCS_rand_per_fold_metrics_5x5.csv",
    "Morgan":  "IC50_Morgan_rand_per_fold_metrics_5x5.csv",
    "Mordred": "IC50_Mordred_scaffCV_per_fold_metrics_5x5.csv",
}

# Metrics to visualize (from per-fold CSVs)
METRICS = ["MCC", "GMean"]
DISPLAY = {
    "MCC": "MCC",
    "GMean": "G-Mean",
}

MODEL_ORDER = [
    "KNN", "SVM", "Bayesian", "LogisticRegression",
    "RandomForest", "LightGBM", "XGBoost",
]

# Colour map for adjusted p-values (same as RM-Tukey script)
_BINS = [0, 0.001, 0.01, 0.05, 1]
_CMAP = mpl.colors.ListedColormap(["#00441b", "#238b45", "#99d8c9", "#fee0d2"])
_NORM = mpl.colors.BoundaryNorm(_BINS, ncolors=_CMAP.N)

PALETTE = sns.color_palette("Set2", n_colors=len(MODEL_ORDER))


# ────────────── UTILS ──────────────

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def make_subject_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'Subject' column exists; build from Repeat+Fold if needed.
    """
    if "Subject" in df.columns:
        return df
    if {"Repeat", "Fold"}.issubset(df.columns):
        df = df.copy()
        df["Subject"] = df["Repeat"].astype(str) + "_F" + df["Fold"].astype(str)
        return df
    warnings.warn("No 'Subject', 'Repeat'+'Fold' not found; skipping Subject.")
    return df


# ────────────── MAIN ──────────────

def main() -> None:
    ensure_dir(DATA_DIR)

    for fp_name, per_fold_file in FP_FILES.items():
        csv_path = DATA_DIR / per_fold_file
        if not csv_path.exists():
            warnings.warn(f"[{fp_name}] per-fold file not found: {csv_path} — skipping.")
            continue

        df = pd.read_csv(csv_path)
        df = make_subject_col(df)

        if "Model" not in df.columns:
            warnings.warn(f"[{fp_name}] No 'Model' column in {csv_path}; skipping.")
            continue

        present_models = [m for m in MODEL_ORDER if m in df["Model"].unique()]
        if len(present_models) < 2:
            warnings.warn(f"[{fp_name}] <2 models present — skipping dashboard.")
            continue

        print(f"\n[{fp_name}] Building 5×5 CV dashboard from {csv_path}")

        # Figure scaffold: 2 metrics × 2 columns (box/strip + Tukey heatmap)
        n_metrics = len(METRICS)
        fig, axes = plt.subplots(
            nrows=n_metrics,
            ncols=2,
            figsize=(12, 8.5),   # slightly shorter → squeezed vertically
            gridspec_kw={"width_ratios": [2, 2]},
            constrained_layout=False,  # we'll control layout manually
        )

        # Adjust layout: reserve space on right for colorbar and tighten hspace
        fig.subplots_adjust(right=0.88, hspace=0.55)

        # If n_metrics == 1, axes is 1D; normalize to 2D [r, c]
        if n_metrics == 1:
            axes = np.array([axes])

        for r, metric in enumerate(METRICS):
            if metric not in df.columns:
                warnings.warn(f"[{fp_name}] Metric '{metric}' not found; skipping.")
                continue

            disp = DISPLAY.get(metric, metric)

            # Long-format sub-DF for plotting
            sub = (
                df[["Model", metric]]
                .rename(columns={metric: "score"})
                .dropna()
            )
            sub = sub[sub["Model"].isin(present_models)]

            # Left panel: box + strip plot
            ax_box = axes[r, 0]
            sns.boxplot(
                data=sub, x="Model", y="score",
                order=present_models, palette=PALETTE,
                width=0.6, linewidth=1, fliersize=2,
                ax=ax_box,
            )
            sns.stripplot(
                data=sub, x="Model", y="score",
                order=present_models, color="k",
                size=2.5, alpha=0.35, jitter=0.25,
                ax=ax_box,
            )

            # No grid, clean spines, consistent y-limits
            ax_box.grid(False)
            sns.despine(ax=ax_box)

            ax_box.set_xlabel("")
            ax_box.set_ylabel(disp)
            ax_box.set_xticklabels(present_models, rotation=45, ha="right")

            # Uniform y-axis range for both metrics (avoid cutoff)
            ax_box.set_ylim(0.4, 1.0)

            # Right panel: Tukey HSD p-value heatmap
            ax_hm = axes[r, 1]

            pmat_path = DATA_DIR / f"{TASK}_{fp_name}_CV_RM_TukeyHSD_pmatrix_{metric}.csv"
            if not pmat_path.exists():
                warnings.warn(
                    f"[{fp_name}] Tukey p-matrix for {metric} not found "
                    f"({pmat_path}); heatmap will be blank."
                )
                empty = pd.DataFrame(
                    np.nan,
                    index=present_models,
                    columns=present_models,
                )
                sns.heatmap(
                    empty,
                    cmap=_CMAP,
                    norm=_NORM,
                    cbar=False,
                    square=True,
                    linewidths=0.5,
                    linecolor="white",
                    xticklabels=present_models,
                    yticklabels=present_models,
                    ax=ax_hm,
                )
            else:
                pmat = pd.read_csv(pmat_path, index_col=0)
                pmat = pmat.reindex(index=present_models, columns=present_models)
                sns.heatmap(
                    pmat,
                    cmap=_CMAP,
                    norm=_NORM,
                    cbar=False,
                    square=True,
                    linewidths=0.5,
                    linecolor="white",
                    xticklabels=present_models,
                    yticklabels=present_models,
                    ax=ax_hm,
                )

            ax_hm.set_title(f"{disp} | pairwise comparisons", pad=6)
            ax_hm.set_xlabel("")
            ax_hm.set_ylabel("")
            ax_hm.set_xticklabels(present_models, rotation=45, ha="right")
            ax_hm.set_yticklabels(present_models, rotation=0)

        # Panel labels: A for MCC row, B for G-Mean row
        panel_labels = ["A", "B"]
        for r, label in enumerate(panel_labels):
            # Left boxplot panel label
            axes[r, 0].text(
                -0.15, 1.05, label + ")",
                transform=axes[r, 0].transAxes,
                fontsize=13,
                #fontweight="bold",
                ha="right",
                va="bottom",
                )

        # Unified colorbar for p-values, placed fully to the right
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # x=0.90 > right=0.88 → no overlap
        cb = mpl.colorbar.ColorbarBase(
            cax,
            cmap=_CMAP,
            norm=_NORM,
            ticks=[0.001, 0.01, 0.05],
        )
        cb.set_ticklabels(["0.001", "0.01", "0.05"])
        cb.set_label("Tukey-adjusted p-value")

        # Save
        pdf_file = DATA_DIR / f"{TASK}_{fp_name}_2mtericsCV_dashboard_5x5.pdf"
        png_file = DATA_DIR / f"{TASK}_{fp_name}_2mtericsCV_dashboard_5x5.png"
        fig.savefig(pdf_file, bbox_inches="tight")
        fig.savefig(png_file, dpi=600, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved dashboard → {pdf_file.name}, {png_file.name}")

    print("\nAll dashboards done.")


if __name__ == "__main__":
    main()
