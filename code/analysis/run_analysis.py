#!/usr/bin/env python
"""
Master Analysis Runner for TRPV1 ML Benchmark

Runs the complete analysis pipeline:
1. Cross-validation with fingerprints (RDKit, Morgan, MACCS)
2. Cross-validation with Mordred descriptors (optional)
3. Statistical analysis (RM-ANOVA, Tukey HSD)
4. Master table generation
5. Visualizations (heatmaps, boxplots, dashboard, bar plots)

Usage:
    python run_analysis.py --endpoint IC50
    python run_analysis.py --endpoint EC50
    python run_analysis.py --endpoints IC50 EC50
    python run_analysis.py --endpoint IC50 --steps 1 2 3
    python run_analysis.py --endpoint IC50 --skip-mordred
"""

import sys
import subprocess
import logging
from pathlib import Path
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

ANALYSIS_STEPS = {
    1: {
        'name': 'Cross-validation with fingerprints',
        'script': '01_cross_validation_fingerprints.py',
        'description': 'Run 5x5 CV with RDKit, Morgan, MACCS fingerprints'
    },
    2: {
        'name': 'Cross-validation with Mordred',
        'script': '02_cross_validation_mordred.py',
        'description': 'Run 5x5 CV with Mordred descriptors'
    },
    3: {
        'name': 'Statistical analysis',
        'script': '03_statistical_analysis.py',
        'description': 'RM-ANOVA and Tukey HSD post-hoc tests'
    },
    4: {
        'name': 'Heatmap visualization',
        'script': '04_visualize_heatmap.py',
        'description': 'Create metric heatmaps'
    },
    5: {
        'name': 'Boxplot visualization',
        'script': '05_visualize_boxplots.py',
        'description': 'Create boxplot distributions'
    },
    6: {
        'name': 'Dashboard visualization',
        'script': '06_visualize_dashboard.py',
        'description': 'Create comprehensive dashboard'
    },
    7: {
        'name': 'Master table generation',
        'script': '07_generate_master_table.py',
        'description': 'Aggregate mean metrics across CV'
    },
    8: {
        'name': 'Bar plot visualization',
        'script': '08_external_bar_plot.py',
        'description': 'External test bar chart'
    },
    9: {
        'name': 'SHAP analysis',
        'script': '09_shap_analysis.py',
        'description': 'SHAP values and plots for best model',
        'optional': True
    },
    10: {
        'name': 'SHAP bit visualization',
        'script': '10_shap_bit_visualization.py',
        'description': 'Visualize important Morgan bits',
        'optional': True
    },
}

def run_step(step_num, endpoint, skip_on_error=False):
    """Run a single analysis step."""

    if step_num not in ANALYSIS_STEPS:
        logging.error(f"Invalid step number: {step_num}")
        return False

    step_info = ANALYSIS_STEPS[step_num]
    script_path = SCRIPT_DIR / step_info['script']

    logging.info("")
    logging.info("=" * 70)
    logging.info(f"STEP {step_num}: {step_info['name']}")
    logging.info(f"Description: {step_info['description']}")
    logging.info("=" * 70)

    if not script_path.exists():
        logging.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path), "--endpoint", endpoint]

    # Special handling for SHAP bit visualization
    if step_num == 10:
        # Load top features from SHAP analysis output
        cmd.extend(["--top-features-file", f"{endpoint}_SHAP_top_features.csv"])

    try:
        logging.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            check=True,
            capture_output=False,
            text=True
        )

        logging.info(f"✓ Step {step_num} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Step {step_num} failed with exit code {e.returncode}")
        if skip_on_error and step_info.get('optional', False):
            logging.warning(f"Skipping optional step {step_num}")
            return True
        return False

    except Exception as e:
        logging.error(f"✗ Step {step_num} failed: {e}")
        if skip_on_error and step_info.get('optional', False):
            logging.warning(f"Skipping optional step {step_num}")
            return True
        return False

def run_full_pipeline(endpoint, steps=None, skip_mordred=False, skip_on_error=False):
    """Run the complete analysis pipeline."""

    logging.info("")
    logging.info("=" * 70)
    logging.info("TRPV1 ML BENCHMARK - ANALYSIS PIPELINE")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")

    if steps is None:
        steps = list(ANALYSIS_STEPS.keys())
        if skip_mordred and 2 in steps:
            steps.remove(2)

    logging.info(f"Steps to run: {steps}")
    logging.info("")

    completed_steps = []
    failed_steps = []

    for step_num in steps:
        success = run_step(step_num, endpoint, skip_on_error=skip_on_error)

        if success:
            completed_steps.append(step_num)
        else:
            failed_steps.append(step_num)
            if not skip_on_error:
                logging.error(f"Pipeline stopped at step {step_num}")
                break

    logging.info("")
    logging.info("=" * 70)
    logging.info("PIPELINE SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Endpoint: {endpoint}")
    logging.info(f"Completed: {len(completed_steps)}/{len(steps)} steps")

    if completed_steps:
        logging.info(f"✓ Successful steps: {completed_steps}")

    if failed_steps:
        logging.error(f"✗ Failed steps: {failed_steps}")

    logging.info("=" * 70)

    return len(failed_steps) == 0

def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description="Master analysis runner for TRPV1 ML Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Steps:
  1. Cross-validation with fingerprints (RDKit, Morgan, MACCS)
  2. Cross-validation with Mordred descriptors
  3. Statistical analysis (RM-ANOVA, Tukey HSD)
  4. Heatmap visualization
  5. Boxplot visualization
  6. Dashboard visualization
  7. Master table generation
  8. Bar plot visualization
  9. SHAP analysis (optional, requires shap package)
  10. SHAP bit visualization (optional, requires step 9)

Examples:
  # Run full pipeline for IC50
  python run_analysis.py --endpoint IC50

  # Run for both endpoints
  python run_analysis.py --endpoints IC50 EC50

  # Run specific steps only
  python run_analysis.py --endpoint IC50 --steps 1 3 7

  # Skip Mordred (requires optional package)
  python run_analysis.py --endpoint IC50 --skip-mordred

  # Continue on errors
  python run_analysis.py --endpoint IC50 --skip-on-error
        """
    )

    parser.add_argument(
        "--endpoint",
        choices=["IC50", "EC50"],
        help="Single endpoint to process"
    )

    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=["IC50", "EC50"],
        help="Multiple endpoints to process"
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=list(ANALYSIS_STEPS.keys()),
        default=None,
        help="Specific steps to run (default: all)"
    )

    parser.add_argument(
        "--skip-mordred",
        action="store_true",
        help="Skip Mordred cross-validation (step 2)"
    )

    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails"
    )

    args = parser.parse_args()

    if args.endpoint:
        endpoints = [args.endpoint]
    elif args.endpoints:
        endpoints = args.endpoints
    else:
        parser.error("Must specify either --endpoint or --endpoints")

    all_success = True

    for endpoint in endpoints:
        logging.info("")
        logging.info("#" * 70)
        logging.info(f"# PROCESSING ENDPOINT: {endpoint}")
        logging.info("#" * 70)

        success = run_full_pipeline(
            endpoint,
            steps=args.steps,
            skip_mordred=args.skip_mordred,
            skip_on_error=args.skip_on_error
        )

        if not success:
            all_success = False

    logging.info("")
    logging.info("#" * 70)
    logging.info("# ANALYSIS PIPELINE COMPLETE")
    logging.info("#" * 70)

    if all_success:
        logging.info("✓ All endpoints completed successfully")
        sys.exit(0)
    else:
        logging.error("✗ Some endpoints failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
