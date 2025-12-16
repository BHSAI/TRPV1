# Quick Refactoring Templates - Copy & Adapt

This document provides ready-to-use templates for quickly refactoring the remaining analysis scripts. Each template shows the complete structure - just copy, adapt the specifics, and you're done.

---

## ✅ Utilities Completed

All utilities are ready to use:
- ✅ `code/utils/config.py` - Paths and constants
- ✅ `code/utils/fingerprints.py` - Molecular fingerprints
- ✅ `code/utils/models.py` - ML model factories
- ✅ `code/utils/evaluation.py` - Metrics calculation
- ✅ `code/utils/stats.py` - **NEW** Statistical analysis
- ✅ `code/utils/visualization.py` - **NEW** Plotting functions

---

## Template 1: Cross-Validation Script

**Use for**: `01_cross_validation_fingerprints.py`, `02_cross_validation_mordred.py`

```python
#!/usr/bin/env python
"""
5x5 Repeated Cross-Validation with [Fingerprints/Mordred]

Performs 5-fold cross-validation repeated 5 times for model comparison.

Usage:
    python [script_name].py --endpoint IC50
    python [script_name].py --endpoint EC50
"""

import sys
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import utilities
from code.utils.config import (
    get_train_file, get_results_dir,
    BASE_RANDOM_STATE, N_REPEATS, N_FOLDS,
    MODEL_ORDER, COL_SMILES, COL_CLASS
)
from code.utils.fingerprints import generate_fingerprints, smiles_to_mols
from code.utils.models import create_model
from code.utils.evaluation import calculate_metrics, get_positive_class_probabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set random seeds
random.seed(BASE_RANDOM_STATE)
np.random.seed(BASE_RANDOM_STATE)

def run_cv(endpoint, fingerprint_type):
    """Run cross-validation for specified endpoint and fingerprint."""

    logging.info(f"Running CV for {endpoint} - {fingerprint_type}")

    # Load data
    train_file = get_train_file(endpoint)
    df = pd.read_csv(train_file)
    df = df.dropna(subset=[COL_SMILES, COL_CLASS]).reset_index(drop=True)

    # Convert SMILES to molecules
    mols, valid_idx = smiles_to_mols(df[COL_SMILES].tolist(), return_indices=True)
    df = df.iloc[valid_idx].reset_index(drop=True)

    # Generate fingerprints
    logging.info(f"Generating {fingerprint_type} fingerprints...")
    X = generate_fingerprints(mols, fingerprint_type)
    y = df[COL_CLASS].values

    # Cross-validation
    all_results = []

    for repeat in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=BASE_RANDOM_STATE + repeat)

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for model_name in MODEL_ORDER:
                # Create and train model
                model = create_model(model_name, random_state=BASE_RANDOM_STATE)
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_val)
                y_prob = get_positive_class_probabilities(model, X_val)

                # Metrics
                metrics = calculate_metrics(y_val, y_pred, y_prob)

                # Store results
                result = {
                    'Repeat': repeat + 1,
                    'Fold': fold + 1,
                    'Model': model_name,
                    'Fingerprint': fingerprint_type,
                    **metrics
                }
                all_results.append(result)

                logging.info(f"R{repeat+1}F{fold+1} {model_name}: ROC-AUC={metrics['ROC_AUC']:.3f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    output_dir = get_results_dir(endpoint, fingerprint_type)
    output_file = output_dir / f"{endpoint}_{fingerprint_type}_per_fold_metrics_5x5.csv"
    results_df.to_csv(output_file, index=False)

    logging.info(f"Saved results to: {output_file}")
    return results_df

def main(endpoint):
    """Main execution."""
    # Run for all fingerprint types
    for fp_type in ["RDKITfp", "Morgan", "MACCS"]:
        run_cv(endpoint, fp_type)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-validation with fingerprints")
    parser.add_argument("--endpoint", choices=["IC50", "EC50"], required=True)
    args = parser.parse_args()
    main(args.endpoint)
```

**Adaptations needed**:
- For Mordred: Replace `generate_fingerprints` with Mordred descriptor calculation
- Adjust fingerprint types list as needed

---

## Template 2: Statistical Analysis Script

**Use for**: `03_statistical_analysis.py`

```python
#!/usr/bin/env python
"""
Statistical Analysis - Repeated Measures ANOVA & Tukey HSD

Compares ML models using RM-ANOVA and Tukey post-hoc tests.

Usage:
    python 03_statistical_analysis.py --endpoint IC50
    python 03_statistical_analysis.py --endpoint EC50
"""

import sys
import logging
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, FINGERPRINT_TYPES
from code.utils.stats import compare_models_rm_anova
from code.utils.visualization import plot_pvalue_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METRICS = ["ROC_AUC", "MCC", "GMean"]

def run_statistical_analysis(endpoint, fingerprint_type, metric):
    """Run RM-ANOVA and Tukey HSD for specified combination."""

    logging.info(f"Analyzing {endpoint} - {fingerprint_type} - {metric}")

    # Load CV results
    results_dir = get_results_dir(endpoint, fingerprint_type)
    data_file = results_dir / f"{endpoint}_{fingerprint_type}_per_fold_metrics_5x5.csv"

    if not data_file.exists():
        logging.warning(f"File not found: {data_file}")
        return None

    df = pd.read_csv(data_file)

    # Run statistical analysis
    stats_results = compare_models_rm_anova(
        df, metric_col=metric, model_col='Model',
        subject_col='Subject', alpha=0.05
    )

    # Save results
    output_dir = get_results_dir(endpoint)

    # Save ANOVA results
    anova_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_ANOVA.txt"
    with open(anova_file, 'w') as f:
        f.write(str(stats_results['anova']))

    # Save Tukey results
    tukey_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_Tukey.csv"
    stats_results['tukey'].to_csv(tukey_file, index=False)

    # Save CLD
    cld_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_CLD.csv"
    stats_results['cld'].to_csv(cld_file, header=True)

    # Plot p-value heatmap
    fig_file = output_dir / f"{endpoint}_{fingerprint_type}_{metric}_pvalue_heatmap.png"
    plot_pvalue_heatmap(
        stats_results['pairwise_matrix'],
        title=f"{endpoint} {fingerprint_type} - {metric}",
        save_path=fig_file
    )

    logging.info(f"Saved statistical results to: {output_dir}")
    return stats_results

def main(endpoint):
    """Main execution."""
    for fp_type in FINGERPRINT_TYPES:
        for metric in METRICS:
            run_statistical_analysis(endpoint, fp_type, metric)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Statistical analysis")
    parser.add_argument("--endpoint", choices=["IC50", "EC50"], required=True)
    args = parser.parse_args()
    main(args.endpoint)
```

---

## Template 3: Master Table Generation

**Use for**: `07_generate_master_table.py`

```python
#!/usr/bin/env python
"""
Generate Master Results Table

Aggregates mean metrics across all CV folds.

Usage:
    python 07_generate_master_table.py --endpoint IC50
"""

import sys
import logging
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, FINGERPRINT_TYPES, MODEL_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METRICS = ["ROC_AUC", "PR_AUC", "MCC", "GMean", "F1", "Accuracy"]

def generate_master_table(endpoint):
    """Generate master table of mean metrics."""

    all_results = []

    for fp_type in FINGERPRINT_TYPES:
        results_dir = get_results_dir(endpoint, fp_type)
        data_file = results_dir / f"{endpoint}_{fp_type}_per_fold_metrics_5x5.csv"

        if not data_file.exists():
            logging.warning(f"Skipping {fp_type}: file not found")
            continue

        df = pd.read_csv(data_file)

        # Calculate mean metrics per model
        for model in MODEL_ORDER:
            model_data = df[df['Model'] == model]

            if len(model_data) == 0:
                continue

            result = {'Fingerprint': fp_type, 'Model': model}

            for metric in METRICS:
                if metric in model_data.columns:
                    result[f"{metric}_mean"] = model_data[metric].mean()
                    result[f"{metric}_std"] = model_data[metric].std()

            all_results.append(result)

    # Create master table
    master_df = pd.DataFrame(all_results)

    # Save
    output_dir = get_results_dir(endpoint)
    output_file = output_dir / f"{endpoint}_master_table_mean_metrics.csv"
    master_df.to_csv(output_file, index=False)

    logging.info(f"Saved master table to: {output_file}")
    return master_df

def main(endpoint):
    """Main execution."""
    generate_master_table(endpoint)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate master results table")
    parser.add_argument("--endpoint", choices=["IC50", "EC50"], required=True)
    args = parser.parse_args()
    main(args.endpoint)
```

---

## Template 4: Visualization Script

**Use for**: `04_visualize_heatmap.py`, `05_visualize_boxplots.py`, etc.

```python
#!/usr/bin/env python
"""
Visualize Results - [Heatmap/Boxplot/Dashboard]

Creates publication-quality figures from CV results.

Usage:
    python [script_name].py --endpoint IC50
"""

import sys
import logging
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_results_dir, get_figure_path, FINGERPRINT_TYPES
from code.utils.visualization import plot_heatmap, plot_boxplots, set_publication_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def create_visualizations(endpoint):
    """Create visualizations for endpoint."""

    set_publication_style()

    # Load master table
    results_dir = get_results_dir(endpoint)
    master_file = results_dir / f"{endpoint}_master_table_mean_metrics.csv"

    if not master_file.exists():
        logging.error(f"Master table not found: {master_file}")
        return

    df = pd.read_csv(master_file)

    # Example: Heatmap of ROC-AUC
    pivot_data = df.pivot(index='Model', columns='Fingerprint', values='ROC_AUC_mean')

    fig_path = get_figure_path(endpoint, "roc_auc_heatmap.png")
    plot_heatmap(
        pivot_data,
        title=f"{endpoint} - ROC-AUC Comparison",
        xlabel="Fingerprint",
        ylabel="Model",
        cmap="RdYlGn",
        save_path=fig_path
    )

    logging.info(f"Saved heatmap to: {fig_path}")

def main(endpoint):
    """Main execution."""
    create_visualizations(endpoint)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create visualizations")
    parser.add_argument("--endpoint", choices=["IC50", "EC50"], required=True)
    args = parser.parse_args()
    main(args.endpoint)
```

---

## Template 5: External Test Evaluation

**Use for**: `08_external_test_evaluation.py`

```python
#!/usr/bin/env python
"""
External Test Set Evaluation

Evaluates best model on external test set.

Usage:
    python 08_external_test_evaluation.py --endpoint IC50
"""

import sys
import logging
import pickle
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import get_test_file, get_model_path, get_results_dir, COL_SMILES, COL_CLASS
from code.utils.fingerprints import generate_fingerprints, smiles_to_mols
from code.utils.evaluation import calculate_metrics, get_positive_class_probabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def evaluate_external_test(endpoint, fingerprint_type="Morgan", model_name="LogisticRegression"):
    """Evaluate model on external test set."""

    logging.info(f"Evaluating {endpoint} - {fingerprint_type} {model_name}")

    # Load test data
    test_file = get_test_file(endpoint)
    df = pd.read_csv(test_file)
    df = df.dropna(subset=[COL_SMILES, COL_CLASS]).reset_index(drop=True)

    # Generate fingerprints
    mols, valid_idx = smiles_to_mols(df[COL_SMILES].tolist(), return_indices=True)
    df = df.iloc[valid_idx].reset_index(drop=True)
    X_test = generate_fingerprints(mols, fingerprint_type)
    y_test = df[COL_CLASS].values

    # Load trained model
    model_file = get_model_path(endpoint, fingerprint_type, model_name)

    if not model_file.exists():
        logging.error(f"Model not found: {model_file}")
        return None

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = get_positive_class_probabilities(model, X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    # Save results
    results_dir = get_results_dir(endpoint)
    results_file = results_dir / f"{endpoint}_{fingerprint_type}_{model_name}_external_test_metrics.csv"

    pd.DataFrame([metrics]).to_csv(results_file, index=False)

    logging.info(f"External test results: ROC-AUC={metrics['ROC_AUC']:.3f}, MCC={metrics['MCC']:.3f}")
    logging.info(f"Saved to: {results_file}")

    return metrics

def main(endpoint):
    """Main execution."""
    evaluate_external_test(endpoint)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate on external test set")
    parser.add_argument("--endpoint", choices=["IC50", "EC50"], required=True)
    args = parser.parse_args()
    main(args.endpoint)
```

---

## Quick Reference: Common Replacements

### **Path Replacements**:
```python
# OLD:
TRAIN_CSV = "TRPV1_IC50_scaffold_split/TRPV1_IC50_train_scaffold.csv"
OUT_FOLDER = "TRPV1_IC50_Morgan"

# NEW:
from code.utils.config import get_train_file, get_results_dir
train_csv = get_train_file(endpoint)  # endpoint = "IC50" or "EC50"
out_folder = get_results_dir(endpoint, "Morgan")
```

### **Fingerprint Generation**:
```python
# OLD:
def fp_morgan(mol):
    arr = np.zeros(2048, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(morgan_gen.GetFingerprint(mol), arr)
    return arr

X = np.array([fp_morgan(m) for m in mols])

# NEW:
from code.utils.fingerprints import generate_fingerprints
X = generate_fingerprints(mols, "Morgan")
```

### **Model Creation**:
```python
# OLD:
model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=rs)

# NEW:
from code.utils.models import create_model
model = create_model("LogisticRegression", random_state=rs)
```

### **Metrics Calculation**:
```python
# OLD:
def calc_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # ... 20 lines of metric calculations ...
    return {...}

metrics = calc_metrics(y_true, y_pred, y_prob)

# NEW:
from code.utils.evaluation import calculate_metrics
metrics = calculate_metrics(y_true, y_pred, y_prob)
```

### **Statistical Analysis**:
```python
# OLD:
# ... 50+ lines of ANOVA/Tukey code ...

# NEW:
from code.utils.stats import compare_models_rm_anova
results = compare_models_rm_anova(df, metric_col='ROC_AUC', model_col='Model')
```

---

## Checklist for Each Script

When refactoring:

- [ ] Add proper shebang (`#!/usr/bin/env python`)
- [ ] Add comprehensive docstring with usage
- [ ] Remove author/date headers
- [ ] Add path setup (SCRIPT_DIR, REPO_ROOT, sys.path)
- [ ] Import from `code.utils.*`
- [ ] Replace all hardcoded paths with config functions
- [ ] Remove duplicate utility functions
- [ ] Add `--endpoint` argument
- [ ] Add logging (not print statements)
- [ ] Clean up comments (remove unnecessary ones)
- [ ] Remove commented-out code
- [ ] Add proper error handling
- [ ] Test with both IC50 and EC50

---

## Priority Order

1. ✅ **Cross-validation** (Templates 1) - Most duplicated
2. ✅ **Statistical analysis** (Template 2) - Needed by many scripts
3. ✅ **Master table** (Template 3) - Quick win
4. **Visualizations** (Template 4) - Multiple scripts
5. **External test** (Template 5) - Important results

---

**All templates are ready to use! Just copy, adapt endpoint-specific logic, and test.**
