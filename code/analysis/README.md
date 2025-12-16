# TRPV1 ML Benchmark - Analysis Scripts

**Status**: Fully refactored and unified (December 2025)

This directory contains unified analysis scripts for the TRPV1 ML benchmark. All scripts work for both IC50 and EC50 endpoints using a single codebase.

---

## 📋 Quick Start

### Run Complete Analysis Pipeline

```bash
# Single endpoint
python run_analysis.py --endpoint IC50

# Both endpoints
python run_analysis.py --endpoints IC50 EC50

# Skip optional Mordred step
python run_analysis.py --endpoint IC50 --skip-mordred

# Run specific steps only
python run_analysis.py --endpoint IC50 --steps 1 3 4 7
```

### Run Individual Scripts

```bash
# Cross-validation with fingerprints
python 01_cross_validation_fingerprints.py --endpoint IC50

# Statistical analysis
python 03_statistical_analysis.py --endpoint IC50

# Generate master table
python 07_generate_master_table.py --endpoint IC50

# Create visualizations
python 04_visualize_heatmap.py --endpoint IC50
python 05_visualize_boxplots.py --endpoint IC50
python 06_visualize_dashboard.py --endpoint IC50
```

---

## 📂 Script Overview

### **1. Cross-Validation Scripts**

#### `01_cross_validation_fingerprints.py`
- **Purpose**: 5×5 repeated stratified cross-validation with molecular fingerprints
- **Fingerprints**: RDKITfp, Morgan, MACCS, AtomPair (default: first 3)
- **Models**: KNN, SVM, Bayesian, LogisticRegression, RandomForest, XGBoost, LightGBM
- **Output**: `{endpoint}_{fingerprint}_per_fold_metrics_5x5.csv`

```bash
# Run with all default fingerprints (RDKit, Morgan, MACCS)
python 01_cross_validation_fingerprints.py --endpoint IC50

# Specify fingerprints
python 01_cross_validation_fingerprints.py --endpoint IC50 --fingerprints Morgan RDKITfp
```

**Output Location**: `results/{endpoint}_results/TRPV1_{endpoint}_{fingerprint}/`

---

#### `02_cross_validation_mordred.py`
- **Purpose**: 5×5 CV with Mordred molecular descriptors
- **Features**: Automatic descriptor cleaning (variance, correlation filtering)
- **Scaling**: MinMax scaling, saved for external test use
- **Output**: `{endpoint}_Mordred_per_fold_metrics_5x5.csv`
- **Note**: Requires `mordred` package (`pip install mordred`)

```bash
python 02_cross_validation_mordred.py --endpoint IC50
```

**Output Location**: `results/{endpoint}_results/TRPV1_{endpoint}_Mordred/`

**Saved Files**:
- `{endpoint}_Mordred_features.pkl` - Feature list
- `{endpoint}_Mordred_scaler.pkl` - Fitted scaler

---

### **2. Statistical Analysis**

#### `03_statistical_analysis.py`
- **Purpose**: Compare models using Repeated Measures ANOVA and Tukey HSD
- **Metrics**: ROC-AUC, MCC, G-Mean
- **Tests**: RM-ANOVA for overall differences, Tukey HSD for pairwise comparisons
- **Output**: ANOVA results, Tukey tables, CLD (Compact Letter Display), p-value heatmaps

```bash
# Run for all fingerprints and metrics
python 03_statistical_analysis.py --endpoint IC50

# Specific fingerprints
python 03_statistical_analysis.py --endpoint IC50 --fingerprints Morgan
```

**Output Files** (per fingerprint × metric):
- `{endpoint}_{fingerprint}_{metric}_ANOVA.txt`
- `{endpoint}_{fingerprint}_{metric}_Tukey.csv`
- `{endpoint}_{fingerprint}_{metric}_CLD.csv`
- `{endpoint}_{fingerprint}_{metric}_pvalue_heatmap.png`

**Output Location**: `results/{endpoint}_results/`

---

### **3. Results Aggregation**

#### `07_generate_master_table.py`
- **Purpose**: Aggregate mean ± std metrics across all CV folds
- **Input**: Per-fold CV results from scripts 01 and 02
- **Output**: Single CSV with mean and std for each model × fingerprint

```bash
python 07_generate_master_table.py --endpoint IC50

# Specific fingerprints only
python 07_generate_master_table.py --endpoint IC50 --fingerprints Morgan RDKITfp
```

**Output File**: `results/{endpoint}_results/{endpoint}_master_table_mean_metrics.csv`

**Columns**:
- Fingerprint, Model, N_folds
- For each metric: `{metric}_mean`, `{metric}_std`

---

### **4. Visualization Scripts**

#### `04_visualize_heatmap.py`
- **Purpose**: Create heatmaps of model performance across fingerprints
- **Metrics**: ROC-AUC, MCC, G-Mean, F1, Accuracy

```bash
# All default metrics
python 04_visualize_heatmap.py --endpoint IC50

# Specific metrics
python 04_visualize_heatmap.py --endpoint IC50 --metrics ROC_AUC MCC
```

**Output**: `figures/{endpoint}_figures/{metric}_heatmap.png`

---

#### `05_visualize_boxplots.py`
- **Purpose**: Box plots showing metric distributions across CV folds
- **Options**: Single fingerprint or combined across all fingerprints

```bash
# All fingerprints combined
python 05_visualize_boxplots.py --endpoint IC50

# Single fingerprint
python 05_visualize_boxplots.py --endpoint IC50 --fingerprint Morgan

# Specific metrics
python 05_visualize_boxplots.py --endpoint IC50 --metrics ROC_AUC MCC
```

**Output**: `figures/{endpoint}_figures/{fingerprint}_{metric}_boxplot.png`

---

#### `06_visualize_dashboard.py`
- **Purpose**: Comprehensive 6-panel dashboard
- **Panels**:
  1. ROC-AUC heatmap
  2. MCC heatmap
  3. G-Mean heatmap
  4. Top 10 models by ROC-AUC (bar chart)
  5. Top 10 models by MCC (bar chart)
  6. Overall metric averages

```bash
python 06_visualize_dashboard.py --endpoint IC50
```

**Output**: `figures/{endpoint}_figures/{endpoint}_dashboard.png`

---

#### `08_external_bar_plot.py`
- **Purpose**: Bar plot of external test performance for best model
- **Metrics**: MCC, G-Mean, ROC-AUC, PR-AUC

```bash
# Default model (Morgan + LogisticRegression)
python 08_external_bar_plot.py --endpoint IC50

# Specify model
python 08_external_bar_plot.py --endpoint IC50 --model Morgan_LogisticRegression
```

**Output**:
- `figures/{endpoint}_figures/{endpoint}_external_{model}_bars.png`
- `figures/{endpoint}_figures/{endpoint}_external_{model}_bars.pdf`

---

#### `09_shap_analysis.py`
- **Purpose**: SHAP analysis for model interpretation
- **Model**: Morgan fingerprints + Logistic Regression
- **Output**: SHAP values, beeswarm plots, bar plots, top features list
- **Note**: Requires `shap` package (`pip install shap`)

```bash
# Run SHAP analysis
python 09_shap_analysis.py --endpoint IC50

# Custom number of top features
python 09_shap_analysis.py --endpoint IC50 --top-features 30

# Adjust background size for speed
python 09_shap_analysis.py --endpoint IC50 --background-size 200
```

**Output Files**:
- `figures/{endpoint}_figures/{endpoint}_SHAP_summary_beeswarm.png`
- `figures/{endpoint}_figures/{endpoint}_SHAP_bar.png`
- `results/{endpoint}_results/{endpoint}_SHAP_top_features.csv`
- `results/{endpoint}_results/{endpoint}_Morgan_LogisticRegression_shap.pkl`

**Output Location**: `results/{endpoint}_results/` and `figures/{endpoint}_figures/`

---

#### `10_shap_bit_visualization.py`
- **Purpose**: Visualize important Morgan bits as highlighted substructures
- **Input**: Top SHAP features from script 09
- **Output**: SVG images with highlighted atoms, bit frequency report

```bash
# Load bits from SHAP analysis output (recommended)
python 10_shap_bit_visualization.py --endpoint IC50 --top-features-file IC50_SHAP_top_features.csv

# Specify bits manually
python 10_shap_bit_visualization.py --endpoint IC50 --bits 1665 843 1019

# Use training set instead of external
python 10_shap_bit_visualization.py --endpoint IC50 --top-features-file IC50_SHAP_top_features.csv --dataset train

# More examples per bit
python 10_shap_bit_visualization.py --endpoint IC50 --bits 1665 843 --examples-per-bit 5
```

**Output Files**:
- `figures/{endpoint}_figures/SHAP_bits_{dataset}/bit{N}_example{M}_idx{I}.svg`
- `figures/{endpoint}_figures/SHAP_bits_{dataset}/{endpoint}_bit_frequencies_{dataset}.csv`

**Note**: Requires running script 09 first to generate top features list.

---

### **5. Master Pipeline Runner**

#### `run_analysis.py`
- **Purpose**: Execute entire analysis pipeline with one command
- **Steps**:
  1. Cross-validation with fingerprints
  2. Cross-validation with Mordred
  3. Statistical analysis
  4. Heatmap visualization
  5. Boxplot visualization
  6. Dashboard visualization
  7. Master table generation
  8. Bar plot visualization
  9. SHAP analysis (optional)
  10. SHAP bit visualization (optional)

```bash
# Full pipeline
python run_analysis.py --endpoint IC50

# Both endpoints
python run_analysis.py --endpoints IC50 EC50

# Skip Mordred (optional step)
python run_analysis.py --endpoint IC50 --skip-mordred

# Run specific steps
python run_analysis.py --endpoint IC50 --steps 1 3 7

# Continue on errors
python run_analysis.py --endpoint IC50 --skip-on-error
```

---

## 🔧 Dependencies

### Required
- pandas, numpy
- rdkit
- scikit-learn
- matplotlib, seaborn
- statsmodels
- xgboost (optional but recommended)
- lightgbm (optional but recommended)

### Optional
- mordred (for script 02)
- shap (for scripts 09-10)

Install all:
```bash
conda env create -f ../../environment.yml
conda activate trpv1_ml_benchmark
```

Or with pip:
```bash
pip install -r ../../requirements.txt
pip install shap  # For SHAP analysis
```

---

## 📊 Output Structure

```
results/
├── IC50_results/
│   ├── TRPV1_IC50_Morgan/
│   │   └── IC50_Morgan_per_fold_metrics_5x5.csv
│   ├── TRPV1_IC50_RDKITfp/
│   │   └── IC50_RDKITfp_per_fold_metrics_5x5.csv
│   ├── TRPV1_IC50_MACCS/
│   │   └── IC50_MACCS_per_fold_metrics_5x5.csv
│   ├── TRPV1_IC50_Mordred/
│   │   ├── IC50_Mordred_per_fold_metrics_5x5.csv
│   │   ├── IC50_Mordred_features.pkl
│   │   └── IC50_Mordred_scaler.pkl
│   ├── IC50_master_table_mean_metrics.csv
│   ├── IC50_Morgan_ROC_AUC_ANOVA.txt
│   ├── IC50_Morgan_ROC_AUC_Tukey.csv
│   ├── IC50_Morgan_ROC_AUC_CLD.csv
│   └── IC50_Morgan_ROC_AUC_pvalue_heatmap.png
│
├── EC50_results/
│   └── ... (same structure)
│
figures/
├── IC50_figures/
│   ├── roc_auc_heatmap.png
│   ├── mcc_heatmap.png
│   ├── Morgan_roc_auc_boxplot.png
│   ├── all_fingerprints_roc_auc_boxplot.png
│   ├── IC50_dashboard.png
│   └── IC50_external_Morgan_LogisticRegression_bars.png
│
└── EC50_figures/
    └── ... (same structure)
```

---

## 🚀 Typical Workflow

### Complete Analysis from Scratch

```bash
# 1. Make sure preprocessing is done
cd ../preprocessing
python run_preprocessing.py --endpoints IC50 EC50

# 2. Run full analysis
cd ../analysis
python run_analysis.py --endpoints IC50 EC50 --skip-mordred

# 3. Results are in:
#    - results/IC50_results/
#    - results/EC50_results/
#    - figures/IC50_figures/
#    - figures/EC50_figures/
```

### Run Individual Components

```bash
# Just cross-validation
python 01_cross_validation_fingerprints.py --endpoint IC50

# Generate summary table
python 07_generate_master_table.py --endpoint IC50

# Create visualizations
python 04_visualize_heatmap.py --endpoint IC50
python 06_visualize_dashboard.py --endpoint IC50
```

---

## 📈 Performance Metrics

All scripts calculate the following metrics:
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **MCC**: Matthews correlation coefficient
- **G-Mean**: Geometric mean of sensitivity and specificity
- **F1**: F1 score
- **Accuracy**: Overall accuracy
- **Sensitivity**: True positive rate (recall)
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
- **NPV**: Negative predictive value
- **FPR**: False positive rate
- **FNR**: False negative rate

---

## 🔄 Comparison with Legacy Scripts

### Before Refactoring
- **32 scripts** (16 IC50 + 16 EC50)
- ~90% code duplication between endpoints
- Hardcoded local paths
- No centralized configuration

### After Refactoring
- **10 unified scripts** + 1 master runner
- 0% duplication (single codebase for both endpoints)
- Repository-relative paths (GitHub-friendly)
- Centralized configuration in `code/utils/config.py`

### Code Reduction
- **Before**: ~4000 lines of analysis code
- **After**: ~1200 lines of unified scripts + ~800 lines of utilities
- **Total Reduction**: ~50% less code to maintain

---

## 🛠️ Utilities Used

All scripts leverage shared utilities from `code/utils/`:

- **config.py** - Paths and constants
- **fingerprints.py** - Molecular fingerprint generation
- **descriptors.py** - Mordred descriptor calculation
- **models.py** - ML model factories
- **evaluation.py** - Metrics calculation
- **stats.py** - Statistical analysis (ANOVA, Tukey HSD)
- **visualization.py** - Plotting functions

See `code/utils/README.md` for utility documentation.

---

## ❓ Troubleshooting

### "Master table not found"
Run `07_generate_master_table.py` before visualization scripts:
```bash
python 07_generate_master_table.py --endpoint IC50
```

### "Mordred package not installed"
Install mordred or skip step 2:
```bash
pip install mordred
# or
python run_analysis.py --endpoint IC50 --skip-mordred
```

### "XGBoost/LightGBM not available"
Install optional packages or they'll be skipped automatically:
```bash
pip install xgboost lightgbm
```

### "Training file not found"
Run preprocessing first:
```bash
cd ../preprocessing
python run_preprocessing.py --endpoint IC50
```

### "SHAP package not installed"
Install shap:
```bash
pip install shap
```

### "Top features file not found" (script 10)
Run SHAP analysis first:
```bash
python 09_shap_analysis.py --endpoint IC50
```

---

## 📝 Adding New Analysis Scripts

To add a new analysis script:

1. **Copy template** from `QUICK_REFACTORING_TEMPLATES.md`
2. **Add path setup**:
   ```python
   SCRIPT_DIR = Path(__file__).resolve().parent
   REPO_ROOT = SCRIPT_DIR.parent.parent
   sys.path.insert(0, str(REPO_ROOT))
   ```
3. **Import utilities**:
   ```python
   from code.utils.config import get_results_dir, get_train_file
   from code.utils.fingerprints import generate_fingerprints
   ```
4. **Add `--endpoint` argument**
5. **Use config functions for all paths**
6. **Add to `run_analysis.py` if part of pipeline**

---

## 📖 Citation

If you use these scripts, please cite:
```
[Your citation here]
```

---

**Last Updated**: December 2025
**Maintainer**: Mohamed Abdulhameed
**Status**: ✅ Production Ready
