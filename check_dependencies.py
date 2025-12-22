#!/usr/bin/env python
"""
Check if all required dependencies for TRPV1 ML Benchmark are installed.

Usage:
    python check_dependencies.py
"""

import sys

def check_imports():
    """Check if all required packages can be imported."""
    packages = {
        "Core Data Processing": {
            "pandas": "Data manipulation and analysis",
            "numpy": "Numerical operations and arrays",
        },
        "Chemistry/Molecular Processing": {
            "rdkit": "Molecular structure handling and fingerprints",
        },
        "Molecular Descriptors": {
            # Provided by mordredcommunity[full], but imported as `mordred`
            "mordred": "Mordred descriptors (provided by mordredcommunity)",
        },
        "Machine Learning": {
            "sklearn": "Scikit-learn ML algorithms",
            "xgboost": "XGBoost gradient boosting",
            "lightgbm": "LightGBM gradient boosting",
        },
        "Statistical Analysis": {
            "statsmodels": "Statistical models and tests",
            "scipy": "Scientific computing",
        },
        "Visualization": {
            "matplotlib": "Plotting and visualization",
            "seaborn": "Statistical data visualization",
        },
        "Model Interpretation": {
            "shap": "SHAP model explanations",
        },
    }

    print("=" * 70)
    print("TRPV1 ML BENCHMARK - DEPENDENCY CHECK")
    print("=" * 70)
    print()

    all_missing = []
    all_found = []

    for category, category_packages in packages.items():
        print(f"{category}:")
        print("-" * 70)

        for package, purpose in category_packages.items():
            try:
                # Try to import the package
                mod = __import__(package)

                # Get version if available
                version = getattr(mod, '__version__', 'unknown')

                print(f"  ✓ {package:15s} v{version:12s} - {purpose}")
                all_found.append(package)

            except ImportError:
                print(f"  ✗ {package:15s} {'MISSING':13s} - {purpose}")
                all_missing.append(package)

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Found: {len(all_found)}/{len(all_found) + len(all_missing)} packages")

    if all_missing:
        print(f"\n❌ Missing packages: {', '.join(all_missing)}")
        print("\nTo install missing packages:")
        print("-" * 70)
        
        step = 1
        
        if 'rdkit' in all_missing:
            print(f"\n{step}. Install RDKit (required for molecular processing):")
            print("   conda install -c conda-forge rdkit")
            print("   OR")
            print("   pip install rdkit")
            all_missing.remove('rdkit')
            step += 1

        if 'mordred' in all_missing:
            print(f"\n{step}. Install Mordred descriptors (recommended):")
            print('   pip install "mordredcommunity[full]"')
            all_missing.remove('mordred')
            step += 1
            
        if all_missing:
            print(f"\n{step}. Install remaining packages:")
            print(f"   pip install {' '.join(all_missing)}")

        print("\nOR install everything at once:")
        print("   conda env create -f environment.yml")
        print("   conda activate trpv1_ml_benchmark")

        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        print("\nYou can now run the preprocessing pipeline:")
        print("   cd code/preprocessing")
        print("   python run_preprocessing.py")
        return True


def check_python_version():
    """Check if Python version is compatible."""
    major, minor = sys.version_info[:2]
    print(f"Python version: {major}.{minor}")

    if major < 3 or (major == 3 and minor < 10):
        print("⚠️  Warning: Python 3.10 or higher is recommended")
        print(f"   You are using Python {major}.{minor}")
        return False
    elif major == 3 and minor > 11:
        print("⚠️  Warning: Python 3.10-3.11 is recommended")
        print(f"   You are using Python {major}.{minor} (may have compatibility issues)")
        return False
    else:
        print("✓ Python version is compatible")
        return True


if __name__ == "__main__":
    print()
    version_ok = check_python_version()
    print()

    imports_ok = check_imports()

    print()
    print("=" * 70)

    sys.exit(0 if (version_ok and imports_ok) else 1)
