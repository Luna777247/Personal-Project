"""
Script: Export Best Model
Exports the best performing model from an experiment
Used in CI/CD pipeline for model deployment
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def export_best_model(
    experiment_name: str,
    metric_name: str = "val_f1_score",
    output_dir: str = "artifacts",
    model_name: str = "best_model"
) -> bool:
    """
    Export the best model from an experiment based on a metric
    
    Args:
        experiment_name: Name of MLflow experiment
        metric_name: Metric to optimize (higher is better)
        output_dir: Directory to save exported model
        model_name: Name for exported model
    
    Returns:
        True if export successful, False otherwise
    """
    
    print("=" * 80)
    print("Export Best Model")
    print("=" * 80)
    
    # Initialize MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    print(f"\n✓ MLflow URI: {mlflow_uri}")
    print(f"✓ Experiment: {experiment_name}")
    print(f"✓ Optimization Metric: {metric_name}")
    
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
    
    # Find best run
    try:
        print(f"\n[1/3] Searching for best run by {metric_name}...")
        
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="status = 'FINISHED'",
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        if not runs:
            print("✗ No finished runs found")
            return False
        
        best_run = runs[0]
        run_id = best_run.info.run_id
        
        print(f"✓ Best Run Found:")
        print(f"  Run ID: {run_id}")
        print(f"  Run Name: {best_run.info.run_name}")
        print(f"  {metric_name}: {best_run.data.metrics.get(metric_name, 'N/A')}")
        
        # Print all metrics
        print(f"\n  All Metrics:")
        for key, value in sorted(best_run.data.metrics.items()):
            print(f"    - {key}: {value:.4f}")
        
    except Exception as e:
        print(f"✗ Failed to find best run: {e}")
        return False
    
    # Download model artifacts
    try:
        print(f"\n[2/3] Downloading model artifacts...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download artifacts
        artifact_path = client.download_artifacts(run_id, "", dst_path=str(output_path))
        
        print(f"✓ Artifacts downloaded to: {artifact_path}")
        
        # List downloaded files
        print(f"\n  Downloaded Files:")
        for item in Path(artifact_path).rglob("*"):
            if item.is_file():
                size = item.stat().st_size
                print(f"    - {item.relative_to(artifact_path)} ({size:,} bytes)")
        
    except Exception as e:
        print(f"✗ Failed to download artifacts: {e}")
        return False
    
    # Copy model files
    try:
        print(f"\n[3/3] Organizing model files...")
        
        # Find model directory
        model_dir = Path(artifact_path) / "model"
        
        if model_dir.exists():
            # Copy to output with custom name
            dest_model_dir = output_path / model_name
            
            if dest_model_dir.exists():
                shutil.rmtree(dest_model_dir)
            
            shutil.copytree(model_dir, dest_model_dir)
            
            print(f"✓ Model copied to: {dest_model_dir}")
            
            # Create metadata file
            metadata = {
                "run_id": run_id,
                "run_name": best_run.info.run_name,
                "experiment_name": experiment_name,
                "metrics": dict(best_run.data.metrics),
                "params": dict(best_run.data.params),
                "artifact_uri": best_run.info.artifact_uri,
                "exported_at": mlflow.utils.time.get_current_time_millis()
            }
            
            import json
            metadata_file = dest_model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Metadata saved to: {metadata_file}")
            
        else:
            print(f"✗ Model directory not found in artifacts")
            return False
        
    except Exception as e:
        print(f"✗ Failed to organize model files: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ EXPORT COMPLETE")
    print("=" * 80)
    print(f"\nExported Model:")
    print(f"  Location: {dest_model_dir}")
    print(f"  Run ID: {run_id}")
    print(f"  Performance: {metric_name} = {best_run.data.metrics.get(metric_name, 'N/A')}")
    print("\nNext Steps:")
    print(f"  1. Review model in: {dest_model_dir}")
    print(f"  2. Export to Triton: python scripts/export_to_triton.py")
    print(f"  3. Deploy model to production")
    
    return True


def main():
    """Main export script"""
    
    parser = argparse.ArgumentParser(description="Export best model from experiment")
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="fraud-detection",
        help="Name of MLflow experiment"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_f1_score",
        help="Metric to optimize (higher is better)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts",
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="best_model",
        help="Name for exported model directory"
    )
    
    args = parser.parse_args()
    
    # Run export
    success = export_best_model(
        experiment_name=args.experiment_name,
        metric_name=args.metric,
        output_dir=args.output,
        model_name=args.model_name
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
