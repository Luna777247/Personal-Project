"""
Script: Export Model to Triton
Converts trained model to Triton-compatible format
Used in CI/CD pipeline for automated deployment
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
from src.deployment.triton_exporter import TritonModelExporter


def export_to_triton(
    model_path: str,
    model_name: str,
    version: int,
    output_dir: str,
    max_batch_size: int = 128
) -> bool:
    """
    Export model to Triton Inference Server format
    
    Args:
        model_path: Path to trained model (MLflow format)
        model_name: Name for Triton model
        version: Model version number
        output_dir: Triton model repository path
        max_batch_size: Maximum batch size for inference
    
    Returns:
        True if export successful, False otherwise
    """
    
    print("=" * 80)
    print("Export Model to Triton Inference Server")
    print("=" * 80)
    
    print(f"\n✓ Model Path: {model_path}")
    print(f"✓ Model Name: {model_name}")
    print(f"✓ Version: {version}")
    print(f"✓ Output Directory: {output_dir}")
    
    # Initialize exporter
    try:
        print(f"\n[1/4] Initializing Triton exporter...")
        exporter = TritonModelExporter(model_repository_path=output_dir)
        print(f"✓ Exporter initialized")
        
    except Exception as e:
        print(f"✗ Failed to initialize exporter: {e}")
        return False
    
    # Load model from MLflow
    try:
        print(f"\n[2/4] Loading model from MLflow...")
        
        model_uri = f"file://{os.path.abspath(model_path)}"
        model = mlflow.sklearn.load_model(model_uri)
        
        print(f"✓ Model loaded: {type(model).__name__}")
        
        # Get model info
        if hasattr(model, 'n_features_in_'):
            print(f"  Input Features: {model.n_features_in_}")
        if hasattr(model, 'classes_'):
            print(f"  Output Classes: {len(model.classes_)}")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Export to ONNX
    try:
        print(f"\n[3/4] Exporting to ONNX format...")
        
        # Determine input features
        if hasattr(model, 'n_features_in_'):
            input_features = model.n_features_in_
        else:
            print("⚠ Could not determine input features, using default: 20")
            input_features = 20
        
        # Export
        exporter.export_sklearn_to_onnx(
            model=model,
            model_name=model_name,
            version=version,
            input_shape=(input_features,),
            max_batch_size=max_batch_size
        )
        
        print(f"✓ Model exported to ONNX")
        
    except Exception as e:
        print(f"✗ Failed to export model: {e}")
        return False
    
    # Verify export
    try:
        print(f"\n[4/4] Verifying exported model...")
        
        model_dir = Path(output_dir) / model_name / str(version)
        
        # Check files
        onnx_file = model_dir / "model.onnx"
        config_file = model_dir.parent / "config.pbtxt"
        
        if not onnx_file.exists():
            print(f"✗ ONNX file not found: {onnx_file}")
            return False
        
        if not config_file.exists():
            print(f"✗ Config file not found: {config_file}")
            return False
        
        print(f"✓ Export verified")
        print(f"\n  Files Created:")
        print(f"    - {onnx_file} ({onnx_file.stat().st_size:,} bytes)")
        print(f"    - {config_file} ({config_file.stat().st_size:,} bytes)")
        
        # Print config
        print(f"\n  Triton Config:")
        with open(config_file, 'r') as f:
            config_content = f.read()
            for line in config_content.split('\n')[:15]:  # First 15 lines
                print(f"    {line}")
            print(f"    ...")
        
    except Exception as e:
        print(f"✗ Failed to verify export: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ EXPORT TO TRITON COMPLETE")
    print("=" * 80)
    print(f"\nTriton Model:")
    print(f"  Repository: {output_dir}")
    print(f"  Model: {model_name}")
    print(f"  Version: {version}")
    print(f"  Path: {model_dir}")
    print("\nNext Steps:")
    print(f"  1. Copy repository to Triton server: {output_dir}")
    print(f"  2. Restart Triton or reload model repository")
    print(f"  3. Test inference: python src/inference/triton_client.py")
    print(f"  4. Deploy to production")
    
    return True


def main():
    """Main export script"""
    
    parser = argparse.ArgumentParser(description="Export model to Triton format")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (MLflow format)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fraud_detector",
        help="Name for Triton model"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Model version number"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Triton model repository path"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Maximum batch size for inference"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"✗ Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Run export
    success = export_to_triton(
        model_path=args.model_path,
        model_name=args.model_name,
        version=args.version,
        output_dir=args.output,
        max_batch_size=args.max_batch_size
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
