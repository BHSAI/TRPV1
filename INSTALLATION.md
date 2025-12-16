# Installation Guide - TRPV1 ML Benchmark

This guide will help you set up the Python environment needed to run the TRPV1 ML benchmark preprocessing and analysis pipeline.

---

## Prerequisites

- Python 3.10 or 3.11 (recommended)
- Git (for cloning the repository)
- Conda (recommended) or pip

---

## Quick Start (Recommended - Using Conda)

### 1. Install Miniconda or Anaconda

If you don't have conda installed:

- Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Or Anaconda: https://www.anaconda.com/download

### 2. Create Environment from YAML

```bash
# Navigate to repository
cd TRPV1_ML_benchmark

# Create conda environment with all dependencies
conda env create -f environment.yml

# Activate the environment
conda activate trpv1_ml_benchmark
```

### 3. Verify Installation

```bash
# Test that all packages are installed
python -c "import rdkit; import pandas; import sklearn; print('✓ All dependencies installed successfully!')"
```

### 4. Run Preprocessing

```bash
cd code/preprocessing
python run_preprocessing.py --endpoints IC50 --steps 1
```

---

## Alternative Installation (Using pip + conda for RDKit)

### Option A: Install RDKit via conda, everything else via pip

```bash
# Create a conda environment with just Python and RDKit
conda create -n trpv1 python=3.10
conda activate trpv1
conda install -c conda-forge rdkit

# Install remaining dependencies via pip
pip install -r requirements.txt
```

### Option B: Pure pip (may have issues with RDKit on some systems)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all dependencies including rdkit-pypi
pip install rdkit-pypi
pip install -r requirements.txt
```

**Note**: `rdkit-pypi` may not work on all systems. If you encounter issues, use conda to install RDKit instead.

---

## Dependency List

### Core Dependencies (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical operations |
| rdkit | ≥2023.3.1 | Molecular processing |
| scikit-learn | ≥1.3.0 | Machine learning |
| xgboost | ≥2.0.0 | Gradient boosting |
| lightgbm | ≥4.0.0 | Gradient boosting |
| statsmodels | ≥0.14.0 | Statistical analysis |
| matplotlib | ≥3.7.0 | Visualization |
| seaborn | ≥0.12.0 | Statistical plots |
| shap | ≥0.42.0 | Model interpretation |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| mordred | Molecular descriptor calculation (if needed) |

---

## Troubleshooting

### Issue: "No module named 'rdkit'"

**Solution**: RDKit is best installed via conda:

```bash
conda install -c conda-forge rdkit
```

If you must use pip:

```bash
pip install rdkit-pypi
```

---

### Issue: "No module named 'pandas'" or other packages

**Solution**: Make sure you've activated your environment:

```bash
# Conda
conda activate trpv1_ml_benchmark

# Or venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

### Issue: Import errors on Windows

**Solution**: Some packages may require Microsoft Visual C++ Build Tools:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Retry package installation

---

### Issue: Conda environment creation fails

**Solution**: Try creating environment manually:

```bash
# Create base environment
conda create -n trpv1 python=3.10

# Activate
conda activate trpv1

# Install packages one by one
conda install -c conda-forge rdkit pandas numpy scikit-learn
conda install -c conda-forge xgboost lightgbm statsmodels
conda install -c conda-forge matplotlib seaborn scipy

# Install pip packages
pip install shap
```

---

## Verifying Your Installation

Run this Python script to check all dependencies:

```python
import sys

def check_imports():
    """Check if all required packages can be imported."""
    packages = {
        'pandas': 'Data processing',
        'numpy': 'Numerical operations',
        'rdkit': 'Molecular chemistry',
        'sklearn': 'Machine learning',
        'xgboost': 'Gradient boosting',
        'lightgbm': 'Gradient boosting',
        'statsmodels': 'Statistical analysis',
        'matplotlib': 'Visualization',
        'seaborn': 'Statistical plots',
        'shap': 'Model interpretation',
    }

    print("Checking dependencies...\n")
    missing = []

    for package, purpose in packages.items():
        try:
            __import__(package)
            print(f"✓ {package:15s} - {purpose}")
        except ImportError:
            print(f"✗ {package:15s} - {purpose} (MISSING)")
            missing.append(package)

    print("\n" + "="*50)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("✓ All dependencies installed successfully!")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
```

Save as `check_dependencies.py` and run:

```bash
python check_dependencies.py
```

---

## Next Steps

Once installation is complete:

1. **Verify data files exist** in `data/raw/`:
   - `TRPV1_chembl_IC50_cleaned_v1.csv`
   - `TRPV1_chembl_EC50_cleaned_v1.csv`

2. **Run preprocessing pipeline**:
   ```bash
   cd code/preprocessing
   python run_preprocessing.py
   ```

3. **See documentation**:
   - Preprocessing: `code/preprocessing/README_UNIFIED_PIPELINE.md`
   - Main README: `README.md`

---

## Environment Management

### Deactivate environment

```bash
# Conda
conda deactivate

# Venv
deactivate
```

### Remove environment

```bash
# Conda
conda env remove -n trpv1_ml_benchmark

# Venv
rm -rf venv  # or rmdir /s venv on Windows
```

### Update environment

```bash
# Conda - update all packages
conda update --all

# Pip - upgrade all packages
pip install --upgrade -r requirements.txt
```

---

## Support

If you encounter issues not covered here:

1. Check Python version: `python --version` (should be 3.10 or 3.11)
2. Check conda version: `conda --version`
3. Try creating a fresh environment
4. Consult package-specific documentation (especially RDKit)

---

**Last Updated**: 2025-12-16
