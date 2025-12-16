#!/usr/bin/env python
"""
5x5 Repeated Cross-Validation with Molecular Fingerprints

Performs 5-fold cross-validation repeated 5 times for model comparison
using molecular fingerprints (RDKit, Morgan, MACCS, AtomPair).

Usage:
    python 01_cross_validation_fingerprints.py --endpoint IC50
    python 01_cross_validation_fingerprints.py --endpoint EC50
    python 01_cross_validation_fingerprints.py --endpoint IC50 --fingerprints Morgan RDKITfp
"""

import sys
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import (
    get_train_file, get_results_dir,
    BASE_RANDOM_STATE, N_REPEATS, N_FOLDS,
    MODEL_ORDER, COL_SMILES, COL_CLASS
)
from code.utils.fingerprints import generate_fingerprints, smiles_to_mols
from code.utils.models import create_model, check_model_availability
from code.utils.evaluation import calculate_metrics, get_positive_class_probabilities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

random.seed(BASE_RANDOM_STATE)
np.random.seed(BASE_RANDOM_STATE)

def load_and_prepare_data(endpoint):
    """Load training data and prepare molecules."""
    train_file = get_train_file(endpoint)

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    logging.info(f"Loading training data from: {train_file}")
    df = pd.read_csv(train_file)

    df = df.dropna(subset=[COL_SMILES, COL_CLASS]).reset_index(drop=True)
    logging.info(f"Loaded {len(df)} samples")

    mols, valid_idx = smiles_to_mols(df[COL_SMILES].tolist(), return_indices=True)
    df = df.iloc[valid_idx].reset_index(drop=True)
    logging.info(f"Valid molecules: {len(mols)}")

    y = df[COL_CLASS].values
    class_counts = pd.Series(y).value_counts()
    logging.info(f"Class distribution: {dict(class_counts)}")

    return df, mols, y

def run_cross_validation(endpoint, fingerprint_type, mols, y):
    """Run 5x5 cross-validation for given fingerprint type."""

    logging.info(f"=" * 70)
    logging.info(f"CV: {endpoint} - {fingerprint_type}")
    logging.info(f"=" * 70)

    logging.info(f"Generating {fingerprint_type} fingerprints...")
    X = generate_fingerprints(mols, fingerprint_type)
    logging.info(f"Fingerprint shape: {X.shape}")

    all_results = []

    for repeat in range(N_REPEATS):
        logging.info(f"\nRepeat {repeat + 1}/{N_REPEATS}")

        cv = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=BASE_RANDOM_STATE + repeat
        )

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for model_name in MODEL_ORDER:
                try:
                    model = create_model(model_name, random_state=BASE_RANDOM_STATE)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_val)
                    y_prob = get_positive_class_probabilities(model, X_val)

                    metrics = calculate_metrics(y_val, y_pred, y_prob)

                    result = {
                        'Repeat': repeat + 1,
                        'Fold': fold + 1,
                        'Model': model_name,
                        'Fingerprint': fingerprint_type,
                        **metrics
                    }
                    all_results.append(result)

                    logging.info(
                        f"  R{repeat+1}F{fold+1} {model_name:20s}: "
                        f"ROC={metrics['ROC_AUC']:.3f} MCC={metrics['MCC']:.3f}"
                    )

                except Exception as e:
                    logging.error(f"Failed for {model_name}: {e}")
                    continue

    results_df = pd.DataFrame(all_results)

    output_dir = get_results_dir(endpoint, fingerprint_type)
    output_file = output_dir / f"{endpoint}_{fingerprint_type}_per_fold_metrics_5x5.csv"
    results_df.to_csv(output_file, index=False)

    mean_roc = results_df.groupby('Model')['ROC_AUC'].mean()
    logging.info(f"\nMean ROC-AUC by Model:")
    for model, roc in mean_roc.sort_values(ascending=False).items():
        logging.info(f"  {model:20s}: {roc:.3f}")

    logging.info(f"\nSaved results to: {output_file}")
    return results_df

def main(endpoint, fingerprint_types=None):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - CROSS-VALIDATION WITH FINGERPRINTS")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")
    logging.info(f"Repeats: {N_REPEATS}, Folds: {N_FOLDS}")

    check_model_availability()

    df, mols, y = load_and_prepare_data(endpoint)

    if fingerprint_types is None:
        fingerprint_types = ["RDKITfp", "Morgan", "MACCS"]

    for fp_type in fingerprint_types:
        try:
            run_cross_validation(endpoint, fp_type, mols, y)
        except Exception as e:
            logging.error(f"Failed for {fp_type}: {e}")
            continue

    logging.info("\n" + "=" * 70)
    logging.info("CROSS-VALIDATION COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="5x5 Cross-Validation with Molecular Fingerprints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_cross_validation_fingerprints.py --endpoint IC50
  python 01_cross_validation_fingerprints.py --endpoint EC50
  python 01_cross_validation_fingerprints.py --endpoint IC50 --fingerprints Morgan
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    parser.add_argument(
        "--fingerprints",
        nargs="+",
        choices=["RDKITfp", "Morgan", "MACCS", "AtomPair"],
        default=None,
        help="Fingerprint types to use (default: RDKITfp Morgan MACCS)"
    )

    args = parser.parse_args()
    main(args.endpoint, args.fingerprints)
