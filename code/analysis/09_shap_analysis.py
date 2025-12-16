#!/usr/bin/env python
"""
SHAP Analysis for External Test Set

Computes SHAP values for the best model (Morgan + Logistic Regression)
on the external test set and creates visualization plots.

Requires: pip install shap

Outputs:
- SHAP beeswarm summary plot (top 20 features)
- SHAP bar plot (mean |SHAP| values)
- Trained model file (.pkl)
- Top SHAP features list (.csv)

Usage:
    python 09_shap_analysis.py --endpoint IC50
    python 09_shap_analysis.py --endpoint EC50
    python 09_shap_analysis.py --endpoint IC50 --top-features 30
"""

import sys
import logging
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from code.utils.config import (
    get_train_file, get_test_file, get_results_dir,
    get_figure_path, BASE_RANDOM_STATE,
    COL_SMILES, COL_CLASS, MORGAN_RADIUS, MORGAN_FP_SIZE
)
from code.utils.models import create_model
from code.utils.shap_utils import (
    check_shap_available,
    featurize_smiles_to_morgan_df,
    compute_shap_values,
    create_shap_summary_plot,
    create_shap_bar_plot,
    get_top_shap_features
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_shap_analysis(endpoint, top_features=20, background_size=300):
    """Run SHAP analysis on external test set."""

    logging.info(f"=" * 70)
    logging.info(f"SHAP ANALYSIS: {endpoint}")
    logging.info(f"=" * 70)

    if not check_shap_available():
        logging.error("SHAP package not installed!")
        logging.error("Install with: pip install shap")
        return None

    # Load data
    train_file = get_train_file(endpoint)
    test_file = get_test_file(endpoint)

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    logging.info(f"Loading training data from: {train_file}")
    df_train = pd.read_csv(train_file)

    logging.info(f"Loading test data from: {test_file}")
    df_test = pd.read_csv(test_file)

    # Clean data
    df_train = df_train.dropna(subset=[COL_SMILES, COL_CLASS]).reset_index(drop=True)
    df_test = df_test.dropna(subset=[COL_SMILES, COL_CLASS]).reset_index(drop=True)

    y_train = df_train[COL_CLASS].astype(int).values
    y_test = df_test[COL_CLASS].astype(int).values

    logging.info(f"Train: {len(df_train)} samples")
    logging.info(f"Test: {len(df_test)} samples")
    logging.info(f"Train class balance: actives={np.sum(y_train==1)}, "
                f"inactives={np.sum(y_train==0)}")
    logging.info(f"Test class balance: actives={np.sum(y_test==1)}, "
                f"inactives={np.sum(y_test==0)}")

    # Generate Morgan fingerprints
    logging.info(f"Generating Morgan fingerprints (radius={MORGAN_RADIUS}, "
                f"{MORGAN_FP_SIZE} bits)...")

    X_train = featurize_smiles_to_morgan_df(
        df_train[COL_SMILES],
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_FP_SIZE
    )

    X_test = featurize_smiles_to_morgan_df(
        df_test[COL_SMILES],
        radius=MORGAN_RADIUS,
        n_bits=MORGAN_FP_SIZE
    )

    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}")

    # Train model
    logging.info("Training LogisticRegression (balanced)...")
    model = create_model("LogisticRegression", random_state=BASE_RANDOM_STATE)
    model.fit(X_train.values, y_train)

    # Save trained model
    results_dir = get_results_dir(endpoint)
    model_path = results_dir / f"{endpoint}_Morgan_LogisticRegression_shap.pkl"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to: {model_path}")

    # Compute SHAP values
    logging.info("Computing SHAP values...")
    shap_values = compute_shap_values(
        model,
        X_train,
        X_test,
        background_sample_size=background_size,
        random_state=BASE_RANDOM_STATE
    )

    logging.info(f"SHAP values shape: {shap_values.shape}")

    # Create SHAP plots
    logging.info("Creating SHAP visualizations...")

    # Beeswarm plot
    summary_path = get_figure_path(endpoint, f"{endpoint}_SHAP_summary_beeswarm.png")
    create_shap_summary_plot(
        shap_values,
        X_test,
        max_display=top_features,
        save_path=summary_path
    )
    logging.info(f"Saved SHAP beeswarm plot to: {summary_path}")

    # Bar plot
    bar_path = get_figure_path(endpoint, f"{endpoint}_SHAP_bar.png")
    create_shap_bar_plot(
        shap_values,
        X_test,
        max_display=top_features,
        save_path=bar_path
    )
    logging.info(f"Saved SHAP bar plot to: {bar_path}")

    # Get top features
    feature_names = X_test.columns.tolist()
    top_feat = get_top_shap_features(shap_values, feature_names, n_top=top_features)

    # Save top features
    top_feat_df = pd.DataFrame(top_feat, columns=["Feature", "Mean_Abs_SHAP"])
    top_feat_file = results_dir / f"{endpoint}_SHAP_top_features.csv"
    top_feat_df.to_csv(top_feat_file, index=False)

    logging.info(f"\nTop {min(10, len(top_feat))} SHAP features:")
    for i, (feat, val) in enumerate(top_feat[:10], 1):
        bit_id = feat.replace("Bit_", "")
        logging.info(f"  {i}. {feat} (bit {bit_id}): {val:.4f}")

    logging.info(f"\nSaved top features to: {top_feat_file}")

    return {
        "shap_values": shap_values,
        "model": model,
        "top_features": top_feat,
        "X_test": X_test,
        "y_test": y_test
    }

def main(endpoint, top_features=20, background_size=300):
    """Main execution function."""

    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - SHAP ANALYSIS")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")
    logging.info(f"Top features: {top_features}")
    logging.info(f"Background size: {background_size}")

    try:
        run_shap_analysis(endpoint, top_features, background_size)
    except Exception as e:
        logging.error(f"SHAP analysis failed: {e}")
        raise

    logging.info("\n" + "=" * 70)
    logging.info("SHAP ANALYSIS COMPLETE")
    logging.info("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SHAP analysis for external test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 09_shap_analysis.py --endpoint IC50
  python 09_shap_analysis.py --endpoint EC50
  python 09_shap_analysis.py --endpoint IC50 --top-features 30

Note: Requires shap package (pip install shap)
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        required=True,
        help="Endpoint to process"
    )

    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Number of top features to display (default: 20)"
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=300,
        help="Background sample size for SHAP (default: 300)"
    )

    args = parser.parse_args()
    main(args.endpoint, args.top_features, args.background_size)
