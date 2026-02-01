"""
Script: Validate Model Metrics
Ensures trained models meet minimum quality thresholds
Used in CI/CD pipeline for automated validation
"""

import os
import sys
import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def validate_metrics(
    experiment_name: str,
    min_accuracy: float = 0.85,
    min_precision: float = 0.80,
    min_recall: float = 0.70,
    min_f1: float = 0.75
) -> bool:
    """
    Validate that the latest run meets minimum metric thresholds
    
    Args:
        experiment_name: Name of MLflow experiment
        min_accuracy: Minimum required accuracy
        min_precision: Minimum required precision
        min_recall: Minimum required recall
        min_f1: Minimum required F1 score
    
    Returns:
        True if all metrics meet thresholds, False otherwise
    """
    
    print("=" * 80)
    print("Model Metrics Validation")
    print("=" * 80)
    
    # Initialize MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    print(f"\n✓ MLflow URI: {mlflow_uri}")
    print(f"✓ Experiment: {experiment_name}")
    
    # Get experiment
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"✗ Experiment '{experiment_name}' not found")
            return False
        
        exp_id = experiment.experiment_id
        print(f"✓ Experiment ID: {exp_id}")
        
    except Exception as e:
        print(f"✗ Failed to get experiment: {e}")
        return False
    
    # Get latest run
    try:
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            print(f"✗ No finished runs found in experiment")
            return False
        
        run = runs[0]
        run_id = run.info.run_id
        print(f"✓ Latest Run ID: {run_id}")
        print(f"  Run Name: {run.info.run_name}")
        
    except Exception as e:
        print(f"✗ Failed to get runs: {e}")
        return False
    
    # Get metrics
    metrics = run.data.metrics
    
    print("\n" + "-" * 80)
    print("Validation Results")
    print("-" * 80)
    
    # Define validation checks
    checks = [
        ("val_accuracy", min_accuracy, "Accuracy"),
        ("val_precision", min_precision, "Precision"),
        ("val_recall", min_recall, "Recall"),
        ("val_f1_score", min_f1, "F1 Score")
    ]
    
    passed = True
    results = []
    
    for metric_name, threshold, display_name in checks:
        value = metrics.get(metric_name)
        
        if value is None:
            print(f"✗ {display_name}: MISSING")
            passed = False
            results.append({
                "metric": display_name,
                "value": None,
                "threshold": threshold,
                "passed": False
            })
        elif value >= threshold:
            print(f"✓ {display_name}: {value:.4f} (>= {threshold:.4f}) PASS")
            results.append({
                "metric": display_name,
                "value": value,
                "threshold": threshold,
                "passed": True
            })
        else:
            print(f"✗ {display_name}: {value:.4f} (< {threshold:.4f}) FAIL")
            passed = False
            results.append({
                "metric": display_name,
                "value": value,
                "threshold": threshold,
                "passed": False
            })
    
    print("-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    if passed:
        print("✓ VALIDATION PASSED: All metrics meet minimum thresholds")
        print("=" * 80)
        return True
    else:
        print("✗ VALIDATION FAILED: Some metrics below minimum thresholds")
        print("=" * 80)
        
        # Print failed metrics
        failed = [r for r in results if not r["passed"]]
        print("\nFailed Checks:")
        for r in failed:
            if r["value"] is None:
                print(f"  - {r['metric']}: MISSING (required >= {r['threshold']:.4f})")
            else:
                diff = r["threshold"] - r["value"]
                print(f"  - {r['metric']}: {r['value']:.4f} (needs +{diff:.4f} to pass)")
        
        return False


def main():
    """Main validation script"""
    
    parser = argparse.ArgumentParser(description="Validate model metrics")
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="fraud-detection",
        help="Name of MLflow experiment"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.85,
        help="Minimum accuracy threshold"
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.80,
        help="Minimum precision threshold"
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.70,
        help="Minimum recall threshold"
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.75,
        help="Minimum F1 score threshold"
    )
    
    args = parser.parse_args()
    
    # Run validation
    success = validate_metrics(
        experiment_name=args.experiment_name,
        min_accuracy=args.min_accuracy,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
