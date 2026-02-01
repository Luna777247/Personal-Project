"""
Triton Model Exporter
Exports trained models to Triton-compatible formats
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import pickle

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import torch

from loguru import logger


class TritonModelExporter:
    """Export models to Triton Inference Server format"""
    
    def __init__(
        self,
        model_repository: str = "./models",
        mlflow_tracking_uri: str = "http://localhost:5000"
    ):
        """
        Initialize exporter
        
        Args:
            model_repository: Path to Triton model repository
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.model_repository = Path(model_repository)
        self.model_repository.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient()
        
        logger.info(f"Triton model repository: {self.model_repository}")
    
    def export_sklearn_to_onnx(
        self,
        model,
        model_name: str,
        version: int = 1,
        input_shape: tuple = (None, 10),
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export sklearn model to ONNX format for Triton
        
        Args:
            model: Trained sklearn model
            model_name: Model name in Triton
            version: Model version
            input_shape: Input tensor shape (first dim should be None for batch)
            config: Triton model config overrides
        
        Returns:
            Path to exported model
        """
        logger.info(f"Exporting {model_name} to ONNX format...")
        
        # Create model directory structure
        model_dir = self.model_repository / model_name
        version_dir = model_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to ONNX
        initial_type = [('float_input', FloatTensorType(input_shape))]
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12
        )
        
        # Save ONNX model
        model_path = version_dir / "model.onnx"
        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        logger.info(f"ONNX model saved: {model_path}")
        
        # Generate Triton config
        self._generate_triton_config(
            model_name=model_name,
            platform="onnxruntime_onnx",
            input_shape=input_shape,
            config=config
        )
        
        # Verify ONNX model
        self._verify_onnx_model(model_path)
        
        return model_path
    
    def export_from_mlflow(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = "Production",
        triton_model_name: Optional[str] = None,
        triton_version: int = 1,
        input_shape: tuple = (None, 10),
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export model from MLflow registry to Triton
        
        Args:
            model_name: MLflow registered model name
            version: Model version (if None, uses stage)
            stage: Model stage (if version is None)
            triton_model_name: Name in Triton (defaults to model_name)
            triton_version: Version in Triton
            input_shape: Input tensor shape
            config: Triton config overrides
        
        Returns:
            Path to exported model
        """
        # Get model URI
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
        
        logger.info(f"Loading model from MLflow: {model_uri}")
        
        # Load model
        model = mlflow.sklearn.load_model(model_uri)
        
        # Export to Triton
        triton_model_name = triton_model_name or model_name
        return self.export_sklearn_to_onnx(
            model=model,
            model_name=triton_model_name,
            version=triton_version,
            input_shape=input_shape,
            config=config
        )
    
    def _generate_triton_config(
        self,
        model_name: str,
        platform: str,
        input_shape: tuple,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate Triton model configuration file
        
        Args:
            model_name: Model name
            platform: Triton platform (e.g., onnxruntime_onnx)
            input_shape: Input tensor shape
            config: Additional config options
        
        Returns:
            Path to config file
        """
        model_dir = self.model_repository / model_name
        config_path = model_dir / "config.pbtxt"
        
        # Default configuration
        default_config = {
            "name": model_name,
            "platform": platform,
            "max_batch_size": 128,
            "input": [
                {
                    "name": "float_input",
                    "data_type": "TYPE_FP32",
                    "dims": list(input_shape[1:])  # Exclude batch dimension
                }
            ],
            "output": [
                {
                    "name": "output_label",
                    "data_type": "TYPE_INT64",
                    "dims": [1]
                },
                {
                    "name": "output_probability",
                    "data_type": "TYPE_FP32",
                    "dims": [-1]  # Variable size for class probabilities
                }
            ],
            "dynamic_batching": {
                "preferred_batch_size": [16, 32, 64],
                "max_queue_delay_microseconds": 100
            },
            "instance_group": [
                {
                    "count": 2,
                    "kind": "KIND_CPU"
                }
            ]
        }
        
        # Merge with custom config
        if config:
            default_config.update(config)
        
        # Write config in protobuf text format
        config_content = self._dict_to_pbtxt(default_config)
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Triton config generated: {config_path}")
        return config_path
    
    def _dict_to_pbtxt(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Convert dictionary to protobuf text format"""
        lines = []
        indent_str = "  " * indent
        
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key} {{")
                lines.append(self._dict_to_pbtxt(value, indent + 1))
                lines.append(f"{indent_str}}}")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{indent_str}{key} {{")
                        lines.append(self._dict_to_pbtxt(item, indent + 1))
                        lines.append(f"{indent_str}}}")
                    else:
                        lines.append(f"{indent_str}{key}: {self._format_value(item)}")
            else:
                lines.append(f"{indent_str}{key}: {self._format_value(value)}")
        
        return "\n".join(lines)
    
    def _format_value(self, value: Any) -> str:
        """Format value for protobuf text format"""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        else:
            return str(value)
    
    def _verify_onnx_model(self, model_path: Path) -> None:
        """Verify ONNX model can be loaded"""
        try:
            # Check ONNX model
            onnx_model = onnx.load(str(model_path))
            onnx.checker.check_model(onnx_model)
            
            # Try to create ONNX Runtime session
            session = ort.InferenceSession(str(model_path))
            
            logger.info(f"✓ ONNX model verified successfully")
            logger.info(f"  Inputs: {[i.name for i in session.get_inputs()]}")
            logger.info(f"  Outputs: {[o.name for o in session.get_outputs()]}")
            
        except Exception as e:
            logger.error(f"✗ ONNX model verification failed: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all models in Triton repository
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        if not self.model_repository.exists():
            return models
        
        for model_dir in self.model_repository.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_info = {
                "name": model_dir.name,
                "versions": []
            }
            
            # List versions
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir() and version_dir.name.isdigit():
                    version_info = {
                        "version": int(version_dir.name),
                        "files": [f.name for f in version_dir.iterdir()]
                    }
                    model_info["versions"].append(version_info)
            
            # Sort versions
            model_info["versions"].sort(key=lambda v: v["version"], reverse=True)
            
            # Check for config
            config_path = model_dir / "config.pbtxt"
            model_info["has_config"] = config_path.exists()
            
            models.append(model_info)
        
        return models
    
    def delete_model(self, model_name: str) -> None:
        """
        Delete a model from Triton repository
        
        Args:
            model_name: Model name to delete
        """
        model_dir = self.model_repository / model_name
        
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Model deleted: {model_name}")
        else:
            logger.warning(f"Model not found: {model_name}")


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate dummy data and train a model
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Initialize exporter
    exporter = TritonModelExporter(model_repository="./models")
    
    # Export to Triton
    model_path = exporter.export_sklearn_to_onnx(
        model=model,
        model_name="fraud_detector",
        version=1,
        input_shape=(None, 10)
    )
    
    print(f"\nModel exported to: {model_path}")
    
    # List all models
    models = exporter.list_models()
    print(f"\nModels in repository: {json.dumps(models, indent=2)}")
