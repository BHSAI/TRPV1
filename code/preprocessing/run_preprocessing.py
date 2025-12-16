#!/usr/bin/env python
"""
Master preprocessing pipeline runner for TRPV1 ML benchmark.

This script runs the complete preprocessing pipeline for IC50 and/or EC50 endpoints:
  1. SMILES standardization (required)
  2. Advanced deduplication (optional)
  3. Similarity check (QC only)
  4. Scaffold-based train/test split (required)

Usage:
    # Run full pipeline for both endpoints (required steps only)
    python run_preprocessing.py

    # Run full pipeline for IC50 only
    python run_preprocessing.py --endpoints IC50

    # Run full pipeline for EC50 only
    python run_preprocessing.py --endpoints EC50

    # Run specific steps only
    python run_preprocessing.py --steps 1 4

    # Run all steps including optional deduplication
    python run_preprocessing.py --include-dedup

    # Run all steps including QC similarity check
    python run_preprocessing.py --include-qc
"""

import sys
import argparse
import logging
import time
import subprocess
from pathlib import Path

# Add repository root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

# ============================================================================
# Pipeline Steps Configuration
# ============================================================================

STEPS = {
    1: {
        "name": "SMILES Standardization",
        "script": "01_standardize_smiles.py",
        "required": True,
        "description": "Validate, standardize SMILES, generate InChIKeys, deduplicate by InChIKey"
    },
    2: {
        "name": "Advanced Deduplication",
        "script": "02_deduplicate.py",
        "required": False,
        "extra_args": ["--apply"],
        "description": "Remove stereo/isotope and tautomer duplicates (optional, not used by step 4)"
    },
    3: {
        "name": "Similarity Check",
        "script": "03_similarity_check.py",
        "required": False,
        "description": "Find identical fingerprint pairs for QC (validation only, no data modification)"
    },
    4: {
        "name": "Scaffold Split",
        "script": "04_scaffold_split.py",
        "required": True,
        "description": "Scaffold-based train/test splitting (80/20, prevents data leakage)"
    }
}

# ============================================================================
# Pipeline Execution
# ============================================================================

def run_step(step_num, endpoint):
    """
    Run a single preprocessing step by executing its script.

    Args:
        step_num: Step number (1-4)
        endpoint: 'IC50' or 'EC50'

    Returns:
        True if successful, False otherwise
    """
    step_info = STEPS[step_num]
    script_path = SCRIPT_DIR / step_info['script']

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    logging.info(f"Running Step {step_num}: {step_info['name']} for {endpoint}")

    # Build command
    cmd = [sys.executable, str(script_path), "--endpoint", endpoint]

    # Add extra arguments if specified (e.g., --apply for deduplication)
    if "extra_args" in step_info:
        cmd.extend(step_info["extra_args"])

    # Run the script
    try:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=False,
            text=True,
            check=True
        )
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Step {step_num} failed for {endpoint}")
        logging.error(f"Command: {' '.join(cmd)}")
        logging.error(f"Return code: {e.returncode}")
        return False


def run_pipeline(endpoints, steps=None, include_dedup=False, include_qc=False):
    """
    Run the complete preprocessing pipeline.

    Args:
        endpoints: List of endpoints to process ['IC50', 'EC50']
        steps: List of step numbers to run (None = all required steps)
        include_dedup: If True, include step 2 (deduplication)
        include_qc: If True, include step 3 (similarity check)
    """
    t0 = time.time()

    # Determine which steps to run
    if steps is None:
        # Default: run required steps (1 and 4)
        steps_to_run = [1, 4]
        if include_dedup:
            steps_to_run.insert(1, 2)  # Add dedup after standardization
        if include_qc:
            steps_to_run.insert(-1, 3)  # Add QC before scaffold split
    else:
        steps_to_run = sorted(steps)

    # Validate steps
    for step_num in steps_to_run:
        if step_num not in STEPS:
            raise ValueError(f"Invalid step number: {step_num}. Must be 1-4.")

    # Print pipeline configuration
    logging.info("=" * 80)
    logging.info("TRPV1 ML BENCHMARK - PREPROCESSING PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Endpoints to process: {', '.join(endpoints)}")
    logging.info(f"Steps to run: {', '.join(str(s) for s in steps_to_run)}")
    logging.info("")

    for step_num in steps_to_run:
        step_info = STEPS[step_num]
        status = "REQUIRED" if step_info["required"] else "OPTIONAL"
        logging.info(f"  Step {step_num} [{status}]: {step_info['name']}")
        logging.info(f"           {step_info['description']}")

    logging.info("")
    logging.info("=" * 80)
    logging.info("")

    # Run pipeline for each endpoint
    results = {}

    for endpoint in endpoints:
        results[endpoint] = {}

        logging.info(f"\n{'#' * 80}")
        logging.info(f"# Processing {endpoint} Endpoint")
        logging.info(f"{'#' * 80}\n")

        for step_num in steps_to_run:
            success = run_step(step_num, endpoint)
            results[endpoint][step_num] = {"success": success}

            if not success:
                logging.error(f"Pipeline failed at step {step_num} for {endpoint}")
                # Continue with other endpoints
                break

    # Print summary
    elapsed = time.time() - t0
    print_summary(results, steps_to_run, elapsed)

    return results


def print_summary(results, steps_run, elapsed):
    """
    Print pipeline execution summary.

    Args:
        results: Dictionary of results by endpoint and step
        steps_run: List of steps that were executed
        elapsed: Total elapsed time in seconds
    """
    logging.info("")
    logging.info("=" * 80)
    logging.info("PIPELINE EXECUTION SUMMARY")
    logging.info("=" * 80)

    for endpoint, endpoint_results in results.items():
        logging.info(f"\n{endpoint} Endpoint:")
        logging.info("-" * 40)

        for step_num in steps_run:
            step_name = STEPS[step_num]["name"]

            if step_num in endpoint_results:
                step_result = endpoint_results[step_num]

                if step_result["success"]:
                    logging.info(f"  ✓ Step {step_num} ({step_name}): SUCCESS")
                else:
                    logging.info(f"  ✗ Step {step_num} ({step_name}): FAILED")
            else:
                logging.info(f"  - Step {step_num} ({step_name}): SKIPPED")

    logging.info("")
    logging.info(f"Total execution time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logging.info("=" * 80)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Master preprocessing pipeline for TRPV1 ML benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full required pipeline for both endpoints
  python run_preprocessing.py

  # Run for IC50 only
  python run_preprocessing.py --endpoints IC50

  # Run for EC50 only
  python run_preprocessing.py --endpoints EC50

  # Run specific steps only (1 and 4)
  python run_preprocessing.py --steps 1 4

  # Include optional deduplication step
  python run_preprocessing.py --include-dedup

  # Include QC similarity check
  python run_preprocessing.py --include-qc

  # Run all steps for both endpoints
  python run_preprocessing.py --include-dedup --include-qc
        """
    )

    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=["IC50", "EC50"],
        default=["IC50", "EC50"],
        help="Endpoints to process (default: both IC50 and EC50)"
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Specific steps to run (default: 1 and 4 - required steps only)"
    )

    parser.add_argument(
        "--include-dedup",
        action="store_true",
        help="Include step 2 (advanced deduplication) - optional step"
    )

    parser.add_argument(
        "--include-qc",
        action="store_true",
        help="Include step 3 (similarity check QC) - validation only"
    )

    args = parser.parse_args()

    # Run pipeline
    run_pipeline(
        endpoints=args.endpoints,
        steps=args.steps,
        include_dedup=args.include_dedup,
        include_qc=args.include_qc
    )


if __name__ == "__main__":
    main()
