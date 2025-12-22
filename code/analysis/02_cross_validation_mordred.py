#!/usr/bin/env python
"""
5x5 Repeated Cross-Validation with Mordred Descriptors

Performs 5-fold cross-validation repeated 5 times for model comparison
using Mordred molecular descriptors with feature cleaning and scaling.

Requires: pip install "mordredcommunity[full]"

Usage:
    python 02_cross_validation_mordred.py --endpoint IC50
    python 02_cross_validation_mordred.py --endpoint EC50
"""

import sys
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import (
    get_train_file, get_results_dir,
    BASE_RANDOM_STATE, N_REPEATS, N_FOLDS,
    MODEL_ORDER, COL_SMILES, COL_CLASS
)
from code.utils.fingerprints import smiles_to_mols
from code.utils.descriptors import (
    check_mordred_available,
    compute_mordred_descriptors,
    clean_descriptors,
    save_descriptor_info
)
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

def run_cross_validation(endpoint, mols, y):
    """Run 5x5 cross-validation with Mordred descriptors."""

    logging.info(f"=" * 70)
    logging.info(f"CV: {endpoint} - Mordred Descriptors")
    logging.info(f"=" * 70)

    if not check_mordred_available():
        logging.error("Mordred package not installed!")
        logging.error("Install with: pip install mordredcommunity[full]")
        return None

    logging.info(f"Computing Mordred descriptors...")
    desc_df = compute_mordred_descriptors(mols)

    if desc_df.empty:
        logging.error("Failed to compute descriptors")
        return None

    logging.info(f"Raw descriptors: {desc_df.shape}")

    logging.info("Cleaning descriptors...")
    desc_df = clean_descriptors(
        desc_df,
        variance_threshold=0.01,
        correlation_threshold=0.95,
        handle_missing='drop',
        verbose=True
    )

    if desc_df.shape[1] == 0:
        logging.error("No descriptors remaining after cleaning")
        return None

    X = desc_df.values

    logging.info(f"Fitting scaler on full training set...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    output_dir = get_results_dir(endpoint, "Mordred")
    save_descriptor_info(desc_df, scaler, output_dir, endpoint)

    all_results = []

    for repeat in range(N_REPEATS):
        logging.info(f"\nRepeat {repeat + 1}/{N_REPEATS}")

        cv = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=BASE_RANDOM_STATE + repeat
        )

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
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
                        'Descriptor': 'Mordred',
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

    output_file = output_dir / f"{endpoint}_Mordred_per_fold_metrics_5x5.csv"
    results_df.to_csv(output_file, index=False)

    mean_roc = results_df.groupby('Model')['ROC_AUC'].mean()
    logging.info(f"\nMean ROC-AUC by Model:")
    for model, roc in mean_roc.sort_values(ascending=False).items():
        logging.info(f"  {model:20s}: {roc:.3f}")

    logging.info(f"\nSaved results to: {output_file}")
    return results_df

def main(endpoint):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - CROSS-VALIDATION WITH MORDRED")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")
    logging.info(f"Repeats: {N_REPEATS}, Folds: {N_FOLDS}")

    check_model_availability()

    df, mols, y = load_and_prepare_data(endpoint)

    try:
        run_cross_validation(endpoint, mols, y)
    except Exception as e:
        logging.error(f"Failed: {e}")
        raise

    logging.info("\n" + "=" * 70)
    logging.info("CROSS-VALIDATION COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="5x5 Cross-Validation with Mordred Descriptors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_cross_validation_mordred.py --endpoint IC50
  python 02_cross_validation_mordred.py --endpoint EC50

Note: Requires mordred package (pip install mordred)
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    args = parser.parse_args()
    main(args.endpoint)
