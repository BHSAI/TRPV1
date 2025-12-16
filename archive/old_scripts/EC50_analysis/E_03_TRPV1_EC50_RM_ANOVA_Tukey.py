# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:09:55 2025

@author: mabdulhameed
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

# ────────────── CONFIG ──────────────

# Folder where you placed all per-fold CSVs
DATA_DIR = Path("TRPV1_EC50_5x5cv_stats")

# Task label (used in filenames)
TASK = "EC50"

# Map fingerprint name → per-fold CSV filename
# Adjust names here if your files differ
FP_FILES = {
    "RDKITfp": "EC50_RDKITfp_rand_per_fold_metrics_5x5.csv",
    "MACCS":   "EC50_MACCS_rand_per_fold_metrics_5x5.csv",
    "Morgan":  "EC50_Morgan_rand_per_fold_metrics_5x5.csv",
    "Mordred": "EC50_Mordred_scaffCV_per_fold_metrics_5x5.csv",
}

# Metrics to analyze (column names in your per-fold CSVs)
METRICS = ["ROC_AUC", "MCC", "GMean"]  # add "PR_AUC" here if you like

# Pretty display names for plots
DISPLAY = {
    "ROC_AUC": "ROC-AUC",
    "MCC":     "MCC",
    "GMean":   "G-Mean",
    "PR_AUC":  "PR-AUC",
}

# Your preferred model order
MODEL_ORDER = [
    "KNN", "SVM", "Bayesian", "LogisticRegression",
    "RandomForest", "LightGBM", "XGBoost",
]

ALPHA = 0.05

# Heatmap bins (for adjusted p-values)
_BINS = [0, 0.001, 0.01, 0.05, 1]
_CMAP = mpl.colors.ListedColormap(["#00441b", "#238b45", "#99d8c9", "#fee0d2"])
_NORM = mpl.colors.BoundaryNorm(_BINS, ncolors=_CMAP.N)

# ────────────── UTILS ──────────────

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def make_subject_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'Subject' column exists; build from Repeat+Fold if needed.
    Subject = each CV split (Repeat_Fold) so that each algorithm is
    evaluated on the same subjects in RM-ANOVA.
    """
    if "Subject" in df.columns:
        return df
    if {"Repeat", "Fold"}.issubset(df.columns):
        df = df.copy()
        df["Subject"] = df["Repeat"].astype(str) + "_F" + df["Fold"].astype(str)
        return df
    raise ValueError("No 'Subject' found and cannot build from 'Repeat'+'Fold'.")


def compact_letter_display(pmat: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """
    Create a Compact Letter Display (CLD) from a symmetric matrix of
    adjusted p-values. Models that are NOT significantly different
    (p >= alpha) share letters.

    Greedy grouping is fine for small M (<= 10).
    """
    models = list(pmat.index)
    groups: list[list[str]] = []

    for m in models:
        placed = False
        for g in groups:
            if all(pmat.loc[m, x] >= alpha for x in g):
                g.append(m)
                placed = True
                break
        if not placed:
            groups.append([m])

    letters: dict[str, str] = {}
    for i, g in enumerate(groups):
        letter = chr(ord("A") + i)
        for m in g:
            letters.setdefault(m, "")
            letters[m] += letter

    return pd.Series(letters)


def rm_anova_balanced(df_long: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """
    Run one-factor RM-ANOVA using statsmodels.AnovaRM after dropping
    incomplete subjects.

    df_long columns: Subject, Model, score
    Returns the anova table (DataFrame).
    """
    wide = df_long.pivot(index="Subject", columns="Model", values="score")[models]
    complete_subjects = wide.dropna().index
    if len(complete_subjects) < 2:
        raise ValueError("Not enough complete subjects (need >= 2) after dropping NaNs.")
    sub_bal = df_long[df_long["Subject"].isin(complete_subjects)].copy()
    return AnovaRM(
        data=sub_bal, depvar="score", subject="Subject", within=["Model"]
    ).fit().anova_table


def rm_tukey_hsd_exact(
    df_long: pd.DataFrame, models: list[str], alpha: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exact 1-factor RM-Tukey(HSD)-style:

      - Build wide balanced matrix (subjects × models)
      - Compute RM ANOVA components; extract MS_error, df_error, n_subjects
      - Compute Tukey p-values (studentized-range) + CIs for pairwise diffs

    Returns:
      pmat (adjusted p-value matrix), pairwise (long table with diffs, CI, p-adj)
    """
    wide = df_long.pivot_table(index="Subject", columns="Model", values="score")[models].dropna()
    n, k = wide.shape
    Y = wide.values
    grand = Y.mean()
    subj_means = Y.mean(axis=1, keepdims=True)   # n×1
    grp_means  = Y.mean(axis=0, keepdims=True)   # 1×k

    # Sums of squares (RM-ANOVA decomposition)
    SS_total = ((Y - grand) ** 2).sum()
    SS_subj  = (k * ((subj_means - grand) ** 2)).sum()
    SS_treat = (n * ((grp_means  - grand) ** 2)).sum()
    SS_error = SS_total - SS_subj - SS_treat
    df_error = (n - 1) * (k - 1)
    MS_error = SS_error / df_error

    means = wide.mean(axis=0)  # EMMs in 1-factor balanced RM
    se = float(np.sqrt(2 * MS_error / n))
    qcrit = float(qsturng(1 - alpha, k, df_error))

    pmat = pd.DataFrame(1.0, index=models, columns=models, dtype=float)
    rows = []

    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i < j:
                diff = float(means[a] - means[b])
                qstat = abs(diff) / se
                # Studentized-range p-value
                p_adj = float(psturng(qstat * math.sqrt(2), k, df_error))
                lo = diff - (qcrit / math.sqrt(2) * se)
                hi = diff + (qcrit / math.sqrt(2) * se)

                pmat.loc[a, b] = pmat.loc[b, a] = p_adj
                rows.append(
                    {
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
                    }
                )

    pairwise = pd.DataFrame(
        rows, index=[f"{r['group1']} - {r['group2']}" for r in rows]
    )
    return pmat, pairwise


def plot_box(wide: pd.DataFrame, title: str, out_png: str) -> None:
    fig = plt.figure(figsize=(8, 4))
    sns.boxplot(data=[wide[c].dropna().values for c in wide.columns], width=0.6)
    # overwrite default tick labels with model names
    plt.xticks(
        ticks=range(len(wide.columns)),
        labels=list(wide.columns),
        rotation=30,
        ha="right",
    )
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(pmat: pd.DataFrame, title: str, out_png: str) -> None:
    fig = plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        pmat,
        cmap=_CAMP,
        norm=_NORM,
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=pmat.columns,
        yticklabels=pmat.index,
    )
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# Fix typo in cmap variable name
_CAMP = _CMAP  # alias to avoid confusion if used above


# ────────────── MAIN ──────────────

def main() -> None:
    ensure_dir(DATA_DIR)

    for fp_name, filename in FP_FILES.items():
        csv_path = DATA_DIR / filename
        if not csv_path.exists():
            print(f"[WARN] Missing per-fold file for {fp_name}: {csv_path} — skipping.")
            continue

        df = pd.read_csv(csv_path)
        if "Model" not in df.columns:
            print(f"[WARN] No 'Model' column in {csv_path}; skipping.")
            continue

        # Build Subject if needed
        try:
            df = make_subject_col(df)
        except ValueError as e:
            print(f"[WARN] {e} in {csv_path}; skipping.")
            continue

        present_models = [m for m in MODEL_ORDER if m in df["Model"].unique()]
        if len(present_models) < 2:
            print(f"[WARN] <2 models present for {fp_name}; skipping.")
            continue

        print(f"\n[{fp_name}] Using file: {csv_path}")
        out_dir = DATA_DIR  # write all outputs into same stats folder
        ensure_dir(out_dir)

        for metric in METRICS:
            if metric not in df.columns:
                print(f"  - Metric '{metric}' not found in file; skipping.")
                continue

            disp = DISPLAY.get(metric, metric)

            # Long format: Subject, Model, score
            sub = (
                df[["Subject", "Model", metric]]
                .rename(columns={metric: "score"})
                .dropna()
            )
            sub = sub[sub["Model"].isin(present_models)].copy()

            # Balanced wide matrix (subjects × models)
            wide = (
                sub.pivot_table(
                    index="Subject", columns="Model", values="score"
                )[present_models]
                .dropna()
            )
            if wide.shape[0] < 2:
                print(f"  - Not enough complete subjects for {metric}; skipping.")
                continue

            # RM-ANOVA
            try:
                aov_tbl = rm_anova_balanced(sub, present_models)
            except Exception as e:
                print(f"  - RM-ANOVA failed for {metric} ({fp_name}): {e}; skipping.")
                continue

            # Exact RM-Tukey(HSD)
            pmat, tukey_pairs = rm_tukey_hsd_exact(sub, present_models, alpha=ALPHA)

            # CLD letters from Tukey p-matrix
            cld = (
                compact_letter_display(pmat.loc[present_models, present_models], alpha=ALPHA)
                .reindex(present_models)
            )

            # Group means (across subjects) for convenience
            group_means = wide.mean(axis=0).reindex(present_models)

            # Filenames
            aov_csv   = out_dir / f"{TASK}_{fp_name}_CV_RMANOVA_{metric}.csv"
            pair_csv  = out_dir / f"{TASK}_{fp_name}_CV_RM_TukeyHSD_pairs_{metric}.csv"
            pmat_csv  = out_dir / f"{TASK}_{fp_name}_CV_RM_TukeyHSD_pmatrix_{metric}.csv"
            cld_csv   = out_dir / f"{TASK}_{fp_name}_CV_CLD_{metric}.csv"
            means_csv = out_dir / f"{TASK}_{fp_name}_CV_group_means_{metric}.csv"

            # Save tables
            aov_tbl.to_csv(aov_csv)
            tukey_pairs.to_csv(pair_csv)
            pmat.to_csv(pmat_csv)
            cld.to_frame("CLD").to_csv(cld_csv)
            group_means.to_frame(metric).to_csv(means_csv)

            # Plots
            box_png  = out_dir / f"{TASK}_{fp_name}_CV_boxplot_{metric}.png"
            heat_png = out_dir / f"{TASK}_{fp_name}_CV_RM_TukeyHSD_heatmap_{metric}.png"

            plot_box(wide, title=f"{disp} ({fp_name}, CV per-fold)", out_png=str(box_png))
            plot_heatmap(
                pmat.loc[present_models, present_models],
                title=f"{disp} — RM-Tukey(HSD) adj p ({fp_name})",
                out_png=str(heat_png),
            )

            print(
                f"  - {metric}: saved "
                f"{aov_csv.name}, {pair_csv.name}, {pmat_csv.name}, "
                f"{cld_csv.name}, {means_csv.name}, {box_png.name}, {heat_png.name}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
