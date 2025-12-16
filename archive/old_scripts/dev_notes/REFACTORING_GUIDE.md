# TRPV1 ML Benchmark - Refactoring Guide

This document provides a comprehensive guide for completing the codebase refactoring to eliminate all duplication between IC50 and EC50 analysis scripts.

---

## ✅ Completed Refactoring

### **1. Preprocessing Pipeline (DONE)**

✅ **Utilities Created**:
- `code/utils/config.py` - Centralized paths and constants
- `code/utils/mol_processing.py` - SMILES validation and standardization
- `code/utils/deduplication.py` - Duplicate removal functions
- `code/utils/scaffold_utils.py` - Scaffold splitting

✅ **Unified Scripts Created**:
- `code/preprocessing/01_standardize_smiles.py` - Works for IC50/EC50
- `code/preprocessing/02_deduplicate.py` - Works for IC50/EC50
- `code/preprocessing/03_similarity_check.py` - Works for IC50/EC50
- `code/preprocessing/04_scaffold_split.py` - Works for IC50/EC50
- `code/preprocessing/run_preprocessing.py` - Master runner

✅ **Legacy Scripts Refactored**:
- `code/preprocessing/IC50_preprocess_scripts/*` - Updated to use utils
- Removed hardcoded paths
- Fixed code quality issues

---

### **2. Analysis Utilities (DONE)**

✅ **New Utilities Created**:
- `code/utils/fingerprints.py` - Molecular fingerprint generation
  - RDKit FP, Morgan FP, MACCS keys, Atom Pair
  - Unified interface via `FINGERPRINT_GENERATORS` dict
  - SMILES to Mol conversion utilities

- `code/utils/models.py` - ML model factories
  - KNN, SVM, Naive Bayes, Logistic Regression
  - Random Forest, XGBoost, LightGBM
  - Consistent model creation via `create_model(name, random_state)`
  - Availability checking for optional packages

- `code/utils/evaluation.py` - Evaluation metrics
  - `calculate_metrics()` - All classification metrics
  - `get_positive_class_probabilities()` - Handle edge cases
  - `aggregate_cv_metrics()` - Cross-validation aggregation
  - `summarize_model_performance()` - Comprehensive summary

✅ **Config Updated**:
- Added analysis constants: `N_REPEATS=5`, `N_FOLDS=5`, `BASE_RANDOM_STATE=42`
- Added `FINGERPRINT_TYPES`, `MODEL_ORDER`
- Added path functions: `get_results_dir()`, `get_model_path()`, `get_figure_path()`

---

## 📋 Remaining Refactoring Tasks

### **3. Analysis Scripts (TODO)**

The following scripts need to be refactored into unified versions:

#### **Cross-Validation Scripts**

**Current State**:
- `code/IC50_analysis/01_TRPV1_IC50_5x5CV_fingerprints.py` (480 lines)
- `code/EC50_analysis/E_01_TRPV1_EC50_5x5CV_fingerprints.py` (480 lines)
- **~95% duplicate code!**

**Needed**:
- Create `code/analysis/01_cross_validation_fingerprints.py --endpoint {IC50|EC50}`

**Key Changes**:
1. Replace hardcoded paths:
   ```python
   # OLD:
   TRAIN_CSV = "TRPV1_IC50_scaffold_split/TRPV1_IC50_train_scaffold.csv"
   OUT_FOLDERS = {"Morgan": "TRPV1_IC50_Morgan", ...}

   # NEW:
   from code.utils.config import get_train_file, get_results_dir
   train_csv = get_train_file(endpoint)
   out_folder = get_results_dir(endpoint, fingerprint_type)
   ```

2. Use shared utilities:
   ```python
   # OLD: Local fp_morgan(), fp_rdk(), fp_maccs() functions
   # NEW:
   from code.utils.fingerprints import generate_fingerprints
   X = generate_fingerprints(mols, fingerprint_type)
   ```

3. Use shared model factories:
   ```python
   # OLD: get_model_factories() returning lambda functions
   # NEW:
   from code.utils.models import create_model
   model = create_model(model_name, random_state=rs)
   ```

4. Use shared metrics:
   ```python
   # OLD: calc_metrics() function
   # NEW:
   from code.utils.evaluation import calculate_metrics
   metrics = calculate_metrics(y_true, y_pred, y_prob)
   ```

---

#### **Mordred Descriptor Scripts**

**Current State**:
- `code/IC50_analysis/02_TRPV1_IC50_5x5CV_Mordred.py`
- `code/EC50_analysis/E_02_TRPV1_EC50_5x5CV_Mordred.py`

**Needed**:
- Create `code/analysis/02_cross_validation_mordred.py --endpoint {IC50|EC50}`

**Key Changes**:
- Same as fingerprint script but uses Mordred descriptors instead
- Consider creating `code/utils/descriptors.py` for Mordred calculation

---

#### **Statistical Analysis Scripts**

**Current State**:
- `code/IC50_analysis/03_TRPV1_IC50_RM_ANOVA_Tukey.py`
- `code/EC50_analysis/E_03_TRPV1_EC50_RM_ANOVA_Tukey.py`

**Needed**:
- Create `code/analysis/03_statistical_analysis.py --endpoint {IC50|EC50}`

**Key Changes**:
1. Extract shared ANOVA/Tukey functions to `code/utils/stats.py`:
   ```python
   def perform_rm_anova(data, subjects, conditions, values):
       """Repeated measures ANOVA using statsmodels."""
       ...

   def perform_tukey_hsd(data, groups, values):
       """Tukey HSD post-hoc test."""
       ...
   ```

2. Use config for paths:
   ```python
   from code.utils.config import get_results_dir
   results_dir = get_results_dir(endpoint)
   ```

---

#### **Visualization Scripts**

**Current State** (12 visualization scripts total):
- `04a_TRPV1_IC50_5x5CV_dashboard.py` / `E_04_TRPV1_EC50_5x5CV_dashboard.py`
- `04b_multimetric_heatmap.py` / `E_04b_multimetric_heatmapEC50.py`
- `04c_allboxplots_4fps_7ML_supp.py` / `E_04c_allboxplots_4fps_7ML_supp.py`
- `14d_TRPV1_IC50_external_bar_plot.py` / `E_14d_TRPV1_EC50_external_bar_plot.py`
- And more...

**Needed**:
1. Create `code/utils/visualization.py` with shared plotting functions:
   ```python
   def plot_cv_dashboard(results_df, endpoint, save_path):
       """Create cross-validation results dashboard."""
       ...

   def plot_heatmap(data, title, save_path):
       """Create multi-metric heatmap."""
       ...

   def plot_boxplots(data, x, y, hue, save_path):
       """Create boxplot comparisons."""
       ...

   def plot_bar_chart(data, x, y, save_path):
       """Create bar chart for external test results."""
       ...
   ```

2. Create unified scripts:
   - `code/analysis/04_visualize_cv_results.py --endpoint {IC50|EC50}`
   - `code/analysis/05_visualize_heatmaps.py --endpoint {IC50|EC50}`
   - `code/analysis/06_visualize_boxplots.py --endpoint {IC50|EC50}`

3. Use config for figure paths:
   ```python
   from code/utils.config import get_figure_path
   fig_path = get_figure_path(endpoint, "cv_dashboard.png")
   ```

---

#### **Model Training & External Test Scripts**

**Current State**:
- `12_TRPV1_IC50_master_table_mean.py` / `E_12_TRPV1_EC50_master_table_mean.py`
- `13_TRPV1_IC50_MorganLR_SDC_AD.py` / `E_14_TRPV1_EC50_MorganLR_external_AD_metrics.py`
- `15_TRPV1_IC50_MorganLR_SHAP_external.py` / `E_15_TRPV1_EC50_MorganLR_SHAP_external.py`
- `16_TRPV1_scaffold_IC50_SHAP_bit_visuals.py` / `E-16_TRPV1_scaffold_EC50_SHAP_bit_visuals.py`

**Needed**:
- `code/analysis/07_generate_master_table.py --endpoint {IC50|EC50}`
- `code/analysis/08_external_test_evaluation.py --endpoint {IC50|EC50}`
- `code/analysis/09_shap_analysis.py --endpoint {IC50|EC50}`
- `code/analysis/10_shap_visualization.py --endpoint {IC50|EC50}`

---

## 🔧 Refactoring Pattern Template

For each analysis script pair, follow this pattern:

### **Step 1: Identify Differences**

Read both IC50 and EC50 versions side-by-side. Differences usually are:
1. Input file paths (IC50 vs EC50)
2. Output directory names
3. Figure titles/labels
4. Hardcoded strings with endpoint name

### **Step 2: Extract Shared Functions**

Move reusable functions to appropriate utility module:
- Fingerprint/descriptor generation → `code/utils/fingerprints.py` or `code/utils/descriptors.py`
- Model creation → `code/utils/models.py` (already done)
- Metrics calculation → `code/utils/evaluation.py` (already done)
- Statistical tests → `code/utils/stats.py` (create new)
- Plotting → `code/utils/visualization.py` (create new)

### **Step 3: Create Unified Script**

Template for unified script:

```python
#!/usr/bin/env python
"""
[Description of what this script does]

Usage:
    python [script_name].py --endpoint IC50
    python [script_name].py --endpoint EC50
"""

import sys
import argparse
import logging
from pathlib import Path

# Add repository root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import from shared utilities
from code.utils.config import (
    get_train_file,
    get_results_dir,
    BASE_RANDOM_STATE,
    N_REPEATS,
    N_FOLDS,
)
from code.utils.fingerprints import generate_fingerprints
from code.utils.models import create_model, MODEL_ORDER
from code.utils.evaluation import calculate_metrics

# Logging setup
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

def main(endpoint):
    """
    Main function for [script purpose].

    Args:
        endpoint: 'IC50' or 'EC50'
    """
    logging.info(f"Running analysis for {endpoint}")

    # Use config functions for paths
    train_file = get_train_file(endpoint)
    results_dir = get_results_dir(endpoint)

    # Load data
    df = pd.read_csv(train_file)

    # ... rest of analysis logic ...
    # Use shared utilities instead of local functions

    logging.info(f"Completed analysis for {endpoint}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[Script description]")
    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to analyze"
    )
    args = parser.parse_args()
    main(args.endpoint)
```

### **Step 4: Test**

```bash
# Test with IC50
python code/analysis/[script_name].py --endpoint IC50

# Test with EC50
python code/analysis/[script_name].py --endpoint EC50

# Compare outputs with original scripts
```

---

## 📊 Duplication Analysis Summary

### **Total Duplication Found**:

| Category | IC50 Scripts | EC50 Scripts | Duplicate Lines | % Duplicate |
|----------|-------------|-------------|-----------------|-------------|
| Preprocessing | 4 scripts | 4 scripts | ~400 | 95% |
| Cross-validation | 2 scripts | 2 scripts | ~960 | 95% |
| Statistical analysis | 1 script | 1 script | ~200 | 90% |
| Visualization | 6 scripts | 5 scripts | ~1100 | 85% |
| Model evaluation | 4 scripts | 3 scripts | ~700 | 90% |
| **TOTAL** | **17 scripts** | **15 scripts** | **~3360 lines** | **~90%** |

### **After Refactoring**:

| Category | Unified Scripts | Utility Modules | Total Lines | Reduction |
|----------|----------------|-----------------|-------------|-----------|
| Preprocessing | 4 + 1 runner | 4 utils | ~800 | 60% |
| Analysis (est.) | 10-12 + 1 runner | 3-4 utils | ~1500 | 70% |
| **TOTAL** | **~16 scripts** | **~8 utils** | **~2300** | **~65%** |

---

## 🎯 Prioritization

### **High Priority** (Do First):
1. ✅ Preprocessing utils (DONE)
2. ✅ Analysis utils (DONE)
3. Cross-validation scripts (01, 02) - Most duplicated
4. Statistical analysis (03) - Used by many scripts
5. Master table generation (12) - Aggregates results

### **Medium Priority**:
6. Heatmap/dashboard visualizations (04a, 04b, 04c)
7. External test evaluation (13/E_14)
8. Bar plots (14d)

### **Low Priority** (Less duplication):
9. SHAP analysis (15, 16) - More complex, fewer duplicates
10. Champions analysis (05a) - IC50 only

---

## 🚀 Quick Win Scripts

These scripts can be refactored quickly (< 30 min each):

1. **`12_master_table_mean.py`** - Just aggregates CSV files
   - Replace paths with `get_results_dir(endpoint)`
   - Add `--endpoint` flag
   - Done!

2. **`14d_external_bar_plot.py`** - Simple bar plot
   - Move plotting code to `visualization.py`
   - Replace paths
   - Add `--endpoint` flag

3. **`03_RM_ANOVA_Tukey.py`** - Statistical tests
   - Move ANOVA/Tukey to `stats.py`
   - Use config for I/O paths
   - Add `--endpoint` flag

---

## 📝 Code Quality Checklist

When refactoring each script:

### **Paths**:
- [ ] No hardcoded paths (use `get_*_file()` or `get_*_dir()`)
- [ ] All paths use `pathlib.Path`
- [ ] Paths are repository-relative

### **Imports**:
- [ ] No duplicate utility functions
- [ ] Import from `code.utils.*`
- [ ] Organized import groups (stdlib, 3rd party, local)

### **Style**:
- [ ] Remove unnecessary comments (keep only complex logic)
- [ ] Remove commented-out code
- [ ] Remove author/date headers
- [ ] Consistent docstrings
- [ ] Type hints on function signatures (optional but nice)

### **Functionality**:
- [ ] Add `--endpoint` CLI argument
- [ ] Logging instead of print statements
- [ ] Error handling for missing files
- [ ] Validate inputs

### **Testing**:
- [ ] Runs without errors for IC50
- [ ] Runs without errors for EC50
- [ ] Output files created in correct locations
- [ ] Results match original scripts

---

## 🛠️ Helper Scripts

### **Find Hardcoded Paths**:

```bash
# Find all hardcoded IC50/EC50 paths
grep -r "IC50" code/IC50_analysis/*.py | grep -E "(csv|pkl|png)"
grep -r "EC50" code/EC50_analysis/*.py | grep -E "(csv|pkl|png)"
```

### **Check for Duplicate Functions**:

```bash
# Find function definitions
grep -r "^def " code/IC50_analysis/ code/EC50_analysis/ | sort | uniq -d
```

---

## 📚 Resources

- **Refactored Examples**: See `code/preprocessing/` for completed refactoring
- **Utility Modules**: See `code/utils/` for reusable components
- **Config Reference**: See `code/utils/config.py` for all path functions
- **Installation**: See `INSTALLATION.md` for dependency setup

---

## ✅ Completion Checklist

### Phase 1: Utilities (DONE)
- [x] Create `fingerprints.py`
- [x] Create `models.py`
- [x] Create `evaluation.py`
- [x] Update `config.py`
- [x] Update `__init__.py`

### Phase 2: Core Analysis (TODO)
- [ ] Create `stats.py` utility
- [ ] Create `visualization.py` utility
- [ ] Create `descriptors.py` utility (for Mordred)
- [ ] Refactor `01_cross_validation_fingerprints.py`
- [ ] Refactor `02_cross_validation_mordred.py`
- [ ] Refactor `03_statistical_analysis.py`

### Phase 3: Visualization (TODO)
- [ ] Refactor `04_cv_dashboard.py`
- [ ] Refactor `05_heatmap.py`
- [ ] Refactor `06_boxplots.py`
- [ ] Refactor other visualization scripts

### Phase 4: Model Evaluation (TODO)
- [ ] Refactor `07_master_table.py`
- [ ] Refactor `08_external_test.py`
- [ ] Refactor `09_shap_analysis.py`
- [ ] Refactor `10_shap_visualization.py`

### Phase 5: Cleanup (TODO)
- [ ] Remove or archive old IC50/EC50 scripts
- [ ] Update README files
- [ ] Create master analysis runner
- [ ] Add tests (optional)

---

**Last Updated**: 2025-12-16
**Status**: Preprocessing complete, analysis utilities ready, core analysis scripts pending
