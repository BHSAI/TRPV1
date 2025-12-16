# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:20:54 2025

@author: mabdulhameed
"""

"""
Champion-only method comparison for TRPV1 IC50 (MCC + G-Mean only)
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
import math

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import qsturng, psturng

# ────────────── GLOBAL STYLE ──────────────
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
})
sns.set_style("white")

# ────────────── CONFIG ──────────────

TASK = "IC50"

STATS_DIR = Path("TRPV1_IC50_5x5cv_stats")

CHAMPIONS = [
    ("Morgan_LR",    "Morgan",   "LogisticRegression",
     "IC50_Morgan_rand_per_fold_metrics_5x5.csv"),
    ("RDKITfp_LR",   "RDKITfp",  "LogisticRegression",
     "IC50_RDKITfp_rand_per_fold_metrics_5x5.csv"),
    ("MACCS_XGB",    "MACCS",    "XGBoost",
     "IC50_MACCS_rand_per_fold_metrics_5x5.csv"),
    ("Mordred_LGBM", "Mordred",  "LightGBM",
     "IC50_Mordred_scaffCV_per_fold_metrics_5x5.csv"),
]

# Only MCC and G-Mean
METRICS = ["MCC", "GMean"]
DISPLAY = {
    "MCC": "MCC",
    "GMean": "G-Mean",
}

ALPHA = 0.05

_BINS = [0, 0.001, 0.01, 0.05, 1]
_CMAP = mpl.colors.ListedColormap(["#00441b", "#238b45", "#99d8c9", "#fee0d2"])
_NORM = mpl.colors.BoundaryNorm(_BINS, ncolors=_CMAP.N)

PALETTE = sns.color_palette("Set2", n_colors=len(CHAMPIONS))


# ────────────── UTILS ──────────────

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def make_subject_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Subject" in df.columns:
        return df
    if {"Repeat", "Fold"}.issubset(df.columns):
        df = df.copy()
        df["Subject"] = df["Repeat"].astype(str) + "_F" + df["Fold"].astype(str)
        return df
    raise ValueError("No 'Subject' and cannot build from 'Repeat' + 'Fold'.")


def rm_anova_balanced(df_long: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    wide = df_long.pivot(index="Subject", columns="Model", values="score")[models]
    complete_subjects = wide.dropna().index
    if len(complete_subjects) < 2:
        raise ValueError("Not enough complete subjects after dropping NaNs.")
    sub_bal = df_long[df_long["Subject"].isin(complete_subjects)].copy()
    return AnovaRM(
        data=sub_bal,
        depvar="score",
        subject="Subject",
        within=["Model"],
    ).fit().anova_table


def rm_tukey_hsd_exact(
    df_long: pd.DataFrame,
    models: list[str],
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    wide = df_long.pivot_table(
        index="Subject",
        columns="Model",
        values="score",
    )[models].dropna()

    n, k = wide.shape
    Y = wide.values
    grand = Y.mean()
    subj_means = Y.mean(axis=1, keepdims=True)
    grp_means = Y.mean(axis=0, keepdims=True)

    SS_total = ((Y - grand) ** 2).sum()
    SS_subj = (k * ((subj_means - grand) ** 2)).sum()
    SS_treat = (n * ((grp_means - grand) ** 2)).sum()
    SS_error = SS_total - SS_subj - SS_treat
    df_error = (n - 1) * (k - 1)
    MS_error = SS_error / df_error

    means = wide.mean(axis=0)
    se = float(np.sqrt(2 * MS_error / n))
    qcrit = float(qsturng(1 - alpha, k, df_error))

    pmat = pd.DataFrame(1.0, index=models, columns=models, dtype=float)
    rows = []

    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i < j:
                diff = float(means[a] - means[b])
                qstat = abs(diff) / se
                p_adj = float(psturng(qstat * math.sqrt(2), k, df_error))
                lo = diff - (qcrit / math.sqrt(2) * se)
                hi = diff + (qcrit / math.sqrt(2) * se)
                pmat.loc[a, b] = pmat.loc[b, a] = p_adj
                rows.append({
                    "group1": a,
                    "group2": b,
                    "meandiff": diff,
                    "lower": lo,
                    "upper": hi,
                    "p-adj": p_adj,
                    "group1_mean": float(means[a]),
                    "group2_mean": float(means[b]),
                    "MS_error": float(MS_error),
                    "df_error": int(df_error),
                    "n_subjects": int(n),
                })

    pairwise = pd.DataFrame(
        rows,
        index=[f"{r['group1']} - {r['group2']}" for r in rows],
    )
    return pmat, pairwise


# ────────────── MAIN ──────────────

def build_champion_long_df() -> pd.DataFrame:
    all_rows = []

    for method_label, fp_name, model_name, csv_name in CHAMPIONS:
        csv_path = STATS_DIR / csv_name
        if not csv_path.exists():
            print(f"[WARN] Missing file for {method_label}: {csv_path} — skipping.")
            continue

        df = pd.read_csv(csv_path)
        df = make_subject_col(df)

        if "Model" not in df.columns:
            print(f"[WARN] No 'Model' column in {csv_path}; skipping {method_label}.")
            continue

        df = df[df["Model"] == model_name].copy()
        if df.empty:
            print(f"[WARN] No rows for model '{model_name}' in {csv_path}; skipping.")
            continue

        df["Method"] = method_label

        keep_cols = ["Subject", "Repeat", "Fold", "Method"] + METRICS
        present = [c for c in keep_cols if c in df.columns]
        all_rows.append(df[present])

        print(f"[INFO] Loaded {len(df)} rows for {method_label} from {csv_path}")

    if not all_rows:
        raise RuntimeError("No champion data loaded; check file paths and names.")

    long_df = pd.concat(all_rows, axis=0, ignore_index=True)
    return long_df


def run_stats_and_plots(df_long: pd.DataFrame) -> None:
    methods = [m[0] for m in CHAMPIONS if m[0] in df_long["Method"].unique()]
    print(f"\nChampion methods included: {methods}")

    out_dir = STATS_DIR
    ensure_dir(out_dir)

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        nrows=n_metrics,
        ncols=2,
        figsize=(12, 8.5),
        gridspec_kw={"width_ratios": [2, 2]},
        constrained_layout=False,
    )
    # Make room for colorbar and space between rows
    fig.subplots_adjust(right=0.88, hspace=0.55)

    if n_metrics == 1:
        axes = np.array([axes])

    for r, metric in enumerate(METRICS):
        if metric not in df_long.columns:
            print(f"[WARN] Metric '{metric}' not found in champion DF; skipping.")
            continue

        disp = DISPLAY.get(metric, metric)

        sub = (
            df_long[["Subject", "Method", metric]]
            .rename(columns={"Method": "Model", metric: "score"})
            .dropna()
        )

        # RM-ANOVA + Tukey
        try:
            aov_tbl = rm_anova_balanced(sub, methods)
        except Exception as e:
            print(f"[WARN] RM-ANOVA failed for {metric}: {e}; skipping metric.")
            continue

        pmat, tukey_pairs = rm_tukey_hsd_exact(sub, methods, alpha=ALPHA)

        wide = sub.pivot_table(
            index="Subject",
            columns="Model",
            values="score",
        )[methods]
        group_means = wide.mean(axis=0).to_frame(metric)

        # Save stats
        aov_csv = out_dir / f"{TASK}_Champions_CV_RM_ANOVA_{metric}.csv"
        pmat_csv = out_dir / f"{TASK}_Champions_CV_RM_TukeyHSD_pmatrix_{metric}.csv"
        pairs_csv = out_dir / f"{TASK}_Champions_CV_RM_TukeyHSD_pairs_{metric}.csv"
        means_csv = out_dir / f"{TASK}_Champions_CV_group_means_{metric}.csv"

        aov_tbl.to_csv(aov_csv)
        pmat.to_csv(pmat_csv)
        tukey_pairs.to_csv(pairs_csv)
        group_means.to_csv(means_csv)

        print(f"[{metric}] Saved: {aov_csv.name}, {pmat_csv.name}, {pairs_csv.name}, {means_csv.name}")

        # Box + strip plot
        ax_box = axes[r, 0]
        sns.boxplot(
            data=sub,
            x="Model",
            y="score",
            order=methods,
            palette=PALETTE,
            width=0.6,
            linewidth=1,
            fliersize=2,
            ax=ax_box,
        )
        sns.stripplot(
            data=sub,
            x="Model",
            y="score",
            order=methods,
            color="k",
            size=3,
            alpha=0.4,
            jitter=0.25,
            ax=ax_box,
        )

        ax_box.grid(False)
        sns.despine(ax=ax_box)

        ax_box.set_xlabel("")
        ax_box.set_ylabel(disp)
        ax_box.set_xticklabels(methods, rotation=45, ha="right")

        # Consistent y-axis for both metrics
        ax_box.set_ylim(0.4, 1.0)

        # Tukey p-value heatmap
        ax_hm = axes[r, 1]
        pmat_ordered = pmat.reindex(index=methods, columns=methods)
        sns.heatmap(
            pmat_ordered,
            cmap=_CMAP,
            norm=_NORM,
            cbar=False,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=methods,
            yticklabels=methods,
            ax=ax_hm,
        )
        ax_hm.set_title(f"{disp} | pairwise comparisons", pad=6)
        ax_hm.set_xlabel("")
        ax_hm.set_ylabel("")
        ax_hm.set_xticklabels(methods, rotation=45, ha="right")
        ax_hm.set_yticklabels(methods, rotation=0)

    # Panel labels A), B) on boxplot side
    panel_labels = ["A", "B"]
    for r, label in enumerate(panel_labels[:n_metrics]):
        axes[r, 0].text(
            -0.15, 1.05, label + ")",
            transform=axes[r, 0].transAxes,
            fontsize=13,
            #fontweight="bold",
            ha="right",
            va="bottom",
        )

    # Unified colorbar
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cb = mpl.colorbar.ColorbarBase(
        cax,
        cmap=_CMAP,
        norm=_NORM,
        ticks=[0.001, 0.01, 0.05],
    )
    cb.set_ticklabels(["0.001", "0.01", "0.05"])
    cb.set_label("Tukey-adjusted p-value")

    dash_pdf = out_dir / f"{TASK}_Champions_CV_2metricsdashboard_5x5.pdf"
    dash_png = out_dir / f"{TASK}_Champions_CV_2metricsdashboard_5x5.png"
    fig.savefig(dash_pdf, bbox_inches="tight")
    fig.savefig(dash_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDashboard saved: {dash_pdf.name}, {dash_png.name}")


def main() -> None:
    ensure_dir(STATS_DIR)
    df_long = build_champion_long_df()
    print(f"\nTotal champion rows: {len(df_long)}")
    run_stats_and_plots(df_long)
    print("\nChampion RM-ANOVA + Tukey analysis complete.")


if __name__ == "__main__":
    main()
