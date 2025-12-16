"""
Statistical analysis utilities for model comparison.

This module provides functions for:
- Repeated Measures ANOVA
- Tukey HSD post-hoc tests
- Compact Letter Display (CLD) generation
- Statistical comparison of ML models across CV folds
"""

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import qsturng, psturng

# ============================================================================
# Data Preparation
# ============================================================================

def prepare_subject_column(df):
    """
    Ensure DataFrame has 'Subject' column for RM-ANOVA.

    Subject represents each CV split (Repeat_Fold combination).

    Args:
        df: DataFrame with 'Repeat' and 'Fold' columns

    Returns:
        DataFrame with 'Subject' column added
    """
    if "Subject" in df.columns:
        return df

    if {"Repeat", "Fold"}.issubset(df.columns):
        df = df.copy()
        df["Subject"] = df["Repeat"].astype(str) + "_F" + df["Fold"].astype(str)
        return df

    raise ValueError("No 'Subject' column found and cannot build from 'Repeat'+'Fold'.")


# ============================================================================
# Repeated Measures ANOVA
# ============================================================================

def perform_repeated_measures_anova(df, depvar, subject, within):
    """
    Perform Repeated Measures ANOVA.

    Args:
        df: DataFrame with repeated measurements
        depvar: Dependent variable column name (e.g., metric)
        subject: Subject ID column name
        within: Within-subject factor column name (e.g., model name)

    Returns:
        ANOVA results object from statsmodels
    """
    rm = AnovaRM(
        data=df,
        depvar=depvar,
        subject=subject,
        within=[within]
    )
    return rm.fit()


# ============================================================================
# Tukey HSD Post-hoc Test
# ============================================================================

def tukey_hsd_pairwise(df, metric_col, group_col, subject_col="Subject"):
    """
    Perform Tukey HSD post-hoc pairwise comparisons.

    Uses Studentized Range distribution for pairwise comparisons
    after RM-ANOVA.

    Args:
        df: DataFrame with measurements
        metric_col: Column containing metric values
        group_col: Column containing group labels (e.g., model names)
        subject_col: Column containing subject IDs

    Returns:
        DataFrame with pairwise comparison results (Tukey HSD)
    """
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    n_subjects = df[subject_col].nunique()

    means = df.groupby(group_col)[metric_col].mean()
    group_data = {g: df[df[group_col] == g][metric_col].values for g in groups}

    # Compute MSE (Mean Squared Error) within subjects
    residuals = []
    for subj in df[subject_col].unique():
        subj_data = df[df[subject_col] == subj]
        grand_mean = subj_data[metric_col].mean()
        for grp in groups:
            grp_vals = subj_data[subj_data[group_col] == grp][metric_col].values
            if len(grp_vals) > 0:
                residuals.append(grp_vals[0] - grand_mean)

    mse = np.var(residuals, ddof=1) if len(residuals) > 1 else 0
    se = np.sqrt(mse / n_subjects)

    results = []
    for g1, g2 in combinations(groups, 2):
        diff = abs(means[g1] - means[g2])
        q_stat = diff / se if se > 0 else 0

        # Studentized range distribution
        # df = (n_subjects - 1) * (n_groups - 1)
        df_error = (n_subjects - 1) * (n_groups - 1)
        p_value = psturng(q_stat, n_groups, df_error)

        results.append({
            "Group1": g1,
            "Group2": g2,
            "MeanDiff": diff,
            "Q-stat": q_stat,
            "p-value": p_value,
            "Significant": p_value < 0.05
        })

    return pd.DataFrame(results)


def create_pairwise_matrix(tukey_results, alpha=0.05):
    """
    Create symmetric matrix of adjusted p-values from Tukey results.

    Args:
        tukey_results: DataFrame from tukey_hsd_pairwise
        alpha: Significance level (default: 0.05)

    Returns:
        DataFrame: Symmetric matrix of p-values
    """
    groups = sorted(set(tukey_results["Group1"].tolist() + tukey_results["Group2"].tolist()))
    pmat = pd.DataFrame(1.0, index=groups, columns=groups)

    for _, row in tukey_results.iterrows():
        g1, g2, pval = row["Group1"], row["Group2"], row["p-value"]
        pmat.loc[g1, g2] = pval
        pmat.loc[g2, g1] = pval

    # Diagonal is 1.0 (same group)
    for g in groups:
        pmat.loc[g, g] = 1.0

    return pmat


# ============================================================================
# Compact Letter Display (CLD)
# ============================================================================

def compact_letter_display(pmat, alpha=0.05):
    """
    Create Compact Letter Display (CLD) from pairwise p-value matrix.

    Models that are NOT significantly different (p >= alpha) share letters.
    Uses greedy grouping algorithm.

    Args:
        pmat: Symmetric matrix of p-values (from create_pairwise_matrix)
        alpha: Significance level (default: 0.05)

    Returns:
        pd.Series: Letter assignments for each model
    """
    models = list(pmat.index)
    groups = []

    for m in models:
        placed = False
        for g in groups:
            # Check if model m is not significantly different from all members of group g
            if all(pmat.loc[m, x] >= alpha for x in g):
                g.append(m)
                placed = True
                break

        if not placed:
            groups.append([m])

    # Assign letters to groups
    letters = {}
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for i, grp in enumerate(groups):
        letter = alphabet[i % len(alphabet)]
        for model in grp:
            if model in letters:
                letters[model] += letter
            else:
                letters[model] = letter

    return pd.Series(letters, name="CLD")


# ============================================================================
# Complete Statistical Analysis Pipeline
# ============================================================================

def compare_models_rm_anova(df, metric_col, model_col, subject_col="Subject", alpha=0.05):
    """
    Complete statistical comparison pipeline:
    1. RM-ANOVA to test for overall differences
    2. Tukey HSD pairwise comparisons
    3. Compact Letter Display generation

    Args:
        df: DataFrame with CV results
        metric_col: Metric to compare (e.g., 'ROC_AUC')
        model_col: Column with model names
        subject_col: Column with subject IDs
        alpha: Significance level

    Returns:
        dict with:
            - 'anova': ANOVA results
            - 'tukey': Tukey HSD results DataFrame
            - 'pairwise_matrix': Symmetric p-value matrix
            - 'cld': Compact Letter Display Series
    """
    # Ensure subject column exists
    df = prepare_subject_column(df)

    # Perform RM-ANOVA
    anova_result = perform_repeated_measures_anova(
        df, depvar=metric_col, subject=subject_col, within=model_col
    )

    # Tukey HSD pairwise comparisons
    tukey_result = tukey_hsd_pairwise(df, metric_col, model_col, subject_col)

    # Create pairwise matrix
    pairwise_matrix = create_pairwise_matrix(tukey_result, alpha)

    # Generate CLD
    cld = compact_letter_display(pairwise_matrix, alpha)

    return {
        "anova": anova_result,
        "tukey": tukey_result,
        "pairwise_matrix": pairwise_matrix,
        "cld": cld
    }
