# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:29:24 2025

@author: mabdulhameed
"""

import pandas as pd
from pathlib import Path

# Point this to your folder with the per-fold CSVs
data_dir = Path("./TRPV1_EC50_5x5cv_stats")

ec50_files = {
    "Morgan":  data_dir / "EC50_Morgan_rand_per_fold_metrics_5x5.csv",
    "RDKit":   data_dir / "EC50_RDKITfp_rand_per_fold_metrics_5x5.csv",
    "MACCS":   data_dir / "EC50_MACCS_rand_per_fold_metrics_5x5.csv",
    "Mordred": data_dir / "EC50_Mordred_scaffCV_per_fold_metrics_5x5.csv",
}

mean_tables = []

for desc, filepath in ec50_files.items():
    df = pd.read_csv(filepath)

    # Sanity check: you should have 25 rows per model
    print(desc, df["Model"].value_counts())

    # Group by Model and take the mean of metrics
    metrics_cols = [
        "ROC_AUC", "PR_AUC", "Accuracy",
        "Sensitivity", "Specificity", "GMean",
        "Precision", "F1", "MCC", "Kappa",
        "TP", "TN", "FN"
    ]

    grouped = (
        df.groupby("Model", as_index=False)[metrics_cols]
          .mean()
    )

    grouped["Descriptor"] = desc
    mean_tables.append(grouped)

# Concatenate into one 28-row master table
ec50_master = pd.concat(mean_tables, ignore_index=True)

# Optional: reorder columns for nicer display
col_order = (
    ["Descriptor", "Model"] +
    metrics_cols
)
ec50_master = ec50_master[col_order]

# Save to CSV for Table 1
out_path = data_dir / "EC50_master_mean_scores.csv"
ec50_master.to_csv(out_path, index=False)

print(f"Saved IE50 master table to: {out_path}")
print(ec50_master.head())
