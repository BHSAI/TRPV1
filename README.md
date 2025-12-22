# TRPV1 Classification Models: Systematic Evaluation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Supporting information for the manuscript:

**Title:** TRPV1 classification models: systematic evaluation across algorithms and molecular representations  
**Authors:** Mohamed Diwan M. AbdulHameed and Anders Wallqvist  
**Manuscript Status:** Under review

---

## Overview

This repository provides a comprehensive benchmark of machine learning models for TRPV1 modulation classification using IC50 and EC50 endpoints.

**Dec 2025 update:** The codebase has been fully updated to eliminate duplication, use repository-relative paths, and provide a unified analysis pipeline.

---

## Key Features

- **7 ML algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, KNN, Gaussian Naive Bayes
- **4 molecular representations**: Morgan fingerprints, RDKit fingerprints, MACCS keys, Mordred descriptors
- **Validation**: repeated stratified cross-validation with scaffold-based external test sets
- **Statistical analysis**: repeated-measures ANOVA and Tukey HSD for model comparison
- **Interpretability**: SHAP-based model interpretation
- **Endpoint support**: IC50, EC50, or both via command-line arguments

---

## Quick Start

### Installation

```bash
git clone https://github.com/BHSAI/TRPV1.git
cd TRPV1

conda env create -f environment.yml
conda activate trpv1_ml_benchmark

python check_dependencies.py
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions and troubleshooting.

---

### Run Complete Pipeline

```bash
# Preprocessing for both endpoints
python code/preprocessing/run_preprocessing.py --endpoints IC50 EC50

# Analysis for both endpoints
python code/analysis/run_analysis.py --endpoints IC50 EC50
```

Outputs are written to endpoint-specific subfolders under:
- `results/`
- `figures/`

---

### Run Individual Steps (examples)

```bash
# Preprocessing
python code/preprocessing/01_standardize_smiles.py --endpoint IC50
python code/preprocessing/04_scaffold_split.py --endpoint IC50

# Analysis
python code/analysis/01_cross_validation_fingerprints.py --endpoint IC50
python code/analysis/07_generate_master_table.py --endpoint IC50
python code/analysis/06_visualize_dashboard.py --endpoint IC50
```

For full script documentation, see:
- `code/preprocessing/README_UNIFIED_PIPELINE.md`
- `code/analysis/README.md`

---

## Repository Structure (High Level)

```
TRPV1_ML_benchmark/
├── code/            # Preprocessing, analysis, utilities
├── data/            # Raw and processed datasets
├── results/         # Tables and evaluation outputs
├── figures/         # Plots for manuscript and SI
├── models/          # Final models
```

---

## Project Scope

This benchmark evaluates TRPV1 classification performance across algorithms, molecular representations, and validation regimes.

### Machine Learning Algorithms (7)
- **Linear**: Logistic Regression  
- **Tree-based**: Random Forest, XGBoost, LightGBM  
- **Instance-based**: K-Nearest Neighbors  
- **Probabilistic**: Gaussian Naive Bayes  
- **Kernel-based**: Support Vector Machine (RBF)

### Molecular Representations (4)
- **Morgan fingerprints**: radius = 2, 2048 bits  
- **RDKit fingerprints**: 2048 bits  
- **MACCS keys**: 166-bit keys  
- **Mordred descriptors**: 1600+ descriptors

### Validation Strategy
- **Internal**: 5×5 repeated stratified cross-validation  
- **External**: scaffold-based test split (20%)  
- **Statistics**: repeated-measures ANOVA + Tukey HSD  
- **Applicability domain**: SDC-based analysis

### Model Interpretation
- **SHAP analysis**: feature importance and dependence plots  
- **Fingerprint interpretation**: bit-level / substructure analysis

---

## Performance Metrics

All models are evaluated using:

- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve
- **MCC**: Matthews correlation coefficient
- **G-Mean**: Geometric mean of sensitivity and specificity
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Sensitivity** (Recall): True positive rate
- **Specificity**: True negative rate
- **Precision**: Positive predictive value
---

## Citation

If you use this code or data, please cite:

```bibtex
@article{abdulhameed2025trpv1,
  title={TRPV1 classification models: systematic evaluation across algorithms and molecular representations},
  author={AbdulHameed, Mohamed Diwan M. and Wallqvist, Anders},
  journal={[Journal Name]},
  year={2025},
  note={Under review}
}
```

---

## Contributing

This repository supports a manuscript under review. For questions or bugs, please open a GitHub issue and include relevant details (command run, endpoint, and error messages).

---

## Acknowledgments

- RDKit community for molecular processing tools  
- Scikit-learn contributors for ML infrastructure  
- The TRPV1 research community for bioactivity data  

---

**Last Updated**: December 2025  
**Manuscript Status**: Under review
