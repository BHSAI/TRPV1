# TRPV1 ML Benchmark - Unified Preprocessing Pipeline

This directory contains the **refactored, unified preprocessing pipeline** for both IC50 and EC50 endpoints. All scripts have been updated to:

- ✅ Use **repository-relative paths** (works from GitHub)
- ✅ Accept **`--endpoint` flag** (IC50/EC50 in single codebase)
- ✅ Import from **shared utilities** (eliminates code duplication)
- ✅ Follow **consistent structure** and documentation

---

## 📁 Directory Structure

```
code/preprocessing/
├── run_preprocessing.py          # ⭐ Master pipeline runner
├── 01_standardize_smiles.py      # Step 1: SMILES standardization (REQUIRED)
├── 02_deduplicate.py             # Step 2: Advanced deduplication (OPTIONAL)
├── 03_similarity_check.py        # Step 3: Similarity check QC (VALIDATION)
├── 04_scaffold_split.py          # Step 4: Scaffold split (REQUIRED)
├── IC50_preprocess_scripts/      # Legacy IC50 scripts (refactored)
└── EC50_preprocess_scripts/      # Legacy EC50 scripts (original)
```

---

## 🚀 Quick Start

### Run Full Pipeline (Both Endpoints)

```bash
cd code/preprocessing
python run_preprocessing.py
```

This runs **required steps only** (steps 1 and 4) for **both IC50 and EC50**.

### Run for Single Endpoint

```bash
# IC50 only
python run_preprocessing.py --endpoints IC50

# EC50 only
python run_preprocessing.py --endpoints EC50
```

### Include Optional Steps

```bash
# Include advanced deduplication (step 2)
python run_preprocessing.py --include-dedup

# Include QC similarity check (step 3)
python run_preprocessing.py --include-qc

# Include all steps
python run_preprocessing.py --include-dedup --include-qc
```

---

## 📋 Pipeline Steps

### **Step 1: SMILES Standardization** (REQUIRED)

**Script**: `01_standardize_smiles.py`

**What it does**:
- Filters non-organic molecules (requires carbon, whitelisted elements only)
- Deep RDKit standardization:
  - InChI round-trip
  - Cleanup (valence, aromaticity)
  - Fragment parent (largest fragment)
  - Neutralization
  - Strip isotopes
  - Normalize stereochemistry
  - Canonical tautomer
- Generates InChIKeys
- Deduplicates by InChIKey

**Input**: `data/raw/TRPV1_chembl_{ENDPOINT}_cleaned_v1.csv`
**Output**: `data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv`

**Usage**:
```bash
python 01_standardize_smiles.py --endpoint IC50
python 01_standardize_smiles.py --endpoint EC50
```

---

### **Step 2: Advanced Deduplication** (OPTIONAL)

**Script**: `02_deduplicate.py`

**What it does**:
- Removes stereo/isotope duplicates (InChIKey14 - connectivity layer)
- Removes tautomer duplicates (stereo-agnostic)
- Uses majority-vote for conflicting labels

**⚠️ NOTE**: This is an **optional** step. The output is **NOT** used by step 4 (scaffold split). Use this if you want a more strictly deduplicated dataset for analysis.

**Input**: `data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv`
**Output**: `data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK_dedup.csv`

**Usage**:
```bash
# Dry-run (preview only, no file written)
python 02_deduplicate.py --endpoint IC50

# Apply changes
python 02_deduplicate.py --endpoint IC50 --apply
```

---

### **Step 3: Similarity Check** (QC/VALIDATION ONLY)

**Script**: `03_similarity_check.py`

**What it does**:
- Computes Morgan fingerprints (radius=2, size=2048)
- Finds pairs with identical fingerprints (Tanimoto = 1.0)
- Saves results for manual inspection

**⚠️ NOTE**: This is a **QC/validation** step only. No data is modified. Use this to identify potential duplicates that may have different InChIKeys.

**Input**: `data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv`
**Output**: `data/intermediate/TRPV1_{ENDPOINT}_identical_pairs.csv`

**Usage**:
```bash
python 03_similarity_check.py --endpoint IC50
python 03_similarity_check.py --endpoint EC50
```

---

### **Step 4: Scaffold Split** (REQUIRED)

**Script**: `04_scaffold_split.py`

**What it does**:
- Extracts Bemis-Murcko scaffolds
- Groups molecules by scaffold (prevents data leakage)
- Balanced scaffold splitting (DeepChem-style, 80/20 train/test)
- Validates class distribution

**Input**: `data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv`
**Output**:
- `data/pre-processed/TRPV1_{ENDPOINT}_train_scaffold.csv` (80% train)
- `data/pre-processed/TRPV1_{ENDPOINT}_exttest_scaffold.csv` (20% test)

**Usage**:
```bash
python 04_scaffold_split.py --endpoint IC50
python 04_scaffold_split.py --endpoint EC50
```

---

## 🔧 Advanced Usage

### Run Specific Steps

```bash
# Run only steps 1 and 4 (required steps)
python run_preprocessing.py --steps 1 4

# Run only standardization
python run_preprocessing.py --steps 1

# Run standardization and deduplication for IC50
python run_preprocessing.py --endpoints IC50 --steps 1 2
```

### Help Information

```bash
# Master script help
python run_preprocessing.py --help

# Individual step help
python 01_standardize_smiles.py --help
python 02_deduplicate.py --help
python 03_similarity_check.py --help
python 04_scaffold_split.py --help
```

---

## 📊 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ data/raw/TRPV1_chembl_{ENDPOINT}_cleaned_v1.csv            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Step 1: Standardize │  (REQUIRED)
              └──────────┬───────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ data/intermediate/TRPV1_{ENDPOINT}_cleaned_RDK.csv         │
└──┬──────────────────────┬───────────────────────────────┬───┘
   │                      │                               │
   │ (optional)           │ (QC only)                     │
   ▼                      ▼                               │
┌─────────────┐   ┌──────────────┐                       │
│ Step 2:     │   │ Step 3:      │                       │
│ Deduplicate │   │ Similarity   │                       │
│             │   │ Check        │                       │
└──────┬──────┘   └──────┬───────┘                       │
       │                 │                               │
       │                 ▼                               │
       │   TRPV1_{ENDPOINT}_identical_pairs.csv          │
       │   (validation output)                           │
       │                                                 │
       ▼                                                 │
TRPV1_{ENDPOINT}_cleaned_RDK_dedup.csv                   │
(not used by step 4)                                     │
                                                         │
                                                         │
                         ┌───────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Step 4: Scaffold     │  (REQUIRED)
              │         Split        │
              └──────────┬───────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
┌────────────────────┐      ┌──────────────────────┐
│ Train Set (80%)    │      │ External Test (20%)  │
└────────────────────┘      └──────────────────────┘
TRPV1_{ENDPOINT}_train_     TRPV1_{ENDPOINT}_exttest_
scaffold.csv                scaffold.csv
```

---

## 🛠️ Shared Utilities

All scripts import from the **`code/utils/`** module:

- **`config.py`**: Paths, parameters, constants
- **`mol_processing.py`**: SMILES validation, standardization, InChIKey generation
- **`deduplication.py`**: Duplicate detection and removal
- **`scaffold_utils.py`**: Scaffold extraction and splitting

This eliminates **~200 lines** of duplicated code and ensures consistency.

---

## 📝 Key Improvements vs. Legacy Scripts

| Feature | Legacy Scripts | New Unified Scripts |
|---------|---------------|---------------------|
| **Paths** | Hardcoded `./file.csv` | Repository-relative via `pathlib` |
| **IC50/EC50** | Separate scripts (8 total) | Unified scripts (4 total) + `--endpoint` flag |
| **Code Duplication** | ~200 lines duplicated | Shared utilities in `code/utils/` |
| **Documentation** | Minimal | Comprehensive docstrings + this README |
| **CLI** | Some missing argparse | All scripts have proper CLI |
| **Error Handling** | Minimal | Validates inputs, handles missing files |
| **Master Runner** | None | `run_preprocessing.py` orchestrates pipeline |
| **Logging** | Inconsistent | Unified format across all scripts |

---

## 🎯 Recommended Workflow

### For Development/Testing
```bash
# Run single endpoint with all steps
python run_preprocessing.py --endpoints IC50 --include-dedup --include-qc
```

### For Production (Paper Results)
```bash
# Run both endpoints, required steps only
python run_preprocessing.py
```

### For QC/Validation
```bash
# Check for potential issues
python run_preprocessing.py --include-qc
```

---

## 📌 Notes

1. **Step 2 (deduplication) is NOT connected to step 4**: The scaffold split uses the output from step 1, not step 2. Step 2 creates an alternative, more strictly deduplicated dataset.

2. **Step 3 (similarity check) is validation only**: It doesn't modify data, just identifies potential issues for manual review.

3. **Required steps**: Only steps 1 and 4 are required for the main ML pipeline.

4. **Input files**: Ensure raw data files exist in `data/raw/`:
   - `TRPV1_chembl_IC50_cleaned_v1.csv`
   - `TRPV1_chembl_EC50_cleaned_v1.csv`

5. **Output organization**: All outputs follow the naming convention in `code/utils/config.py`.

---

## 🐛 Troubleshooting

**Problem**: `FileNotFoundError: Input file not found`
**Solution**: Ensure you're running from repository root or `code/preprocessing/` directory. Check that raw data files exist.

**Problem**: `ModuleNotFoundError: No module named 'code.utils'`
**Solution**: The script adds repo root to `sys.path`. Make sure you're running from the repository.

**Problem**: Script runs but outputs go to wrong directory
**Solution**: All paths are repository-relative. Check that `code/utils/config.py` paths are correct.

---

## 📚 Further Reading

- See `code/utils/README.md` for utilities documentation (if created)
- See legacy `IC50_preprocess_scripts/` for original implementation
- See `code/README_IC50_analysis.md` for downstream analysis pipeline

---

**Last Updated**: 2025-12-16
**Author**: Refactored for unified IC50/EC50 processing
