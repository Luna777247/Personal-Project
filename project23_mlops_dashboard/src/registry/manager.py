"""
Model Registry Manager
Manages model versioning, staging, and promotion in MLflow Registry
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from loguru import logger


class ModelRegistryManager:
    """Manager for MLflow Model Registry operations"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize registry manager
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.tracking_uri = tracking_uri
        logger.info(f"Connected to MLflow: {tracking_uri}")
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> ModelVersion:
        """
        Register a model from a run
        
        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            artifact_path: Path to model artifact in run
            description: Model description
            tags: Model tags
        
        Returns:
            ModelVersion object
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        logger.info(f"Registering model: {model_name} from run {run_id}")
        
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Update description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        logger.info(
            f"Model registered: {model_name} v{model_version.version} "
            f"(stage: {model_version.current_stage})"
        )
        
        return model_version
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None
    ) -> ModelVersion:
        """
        Get a specific model version
        
        Args:
            model_name: Registered model name
            version: Version number (if None, gets latest)
            stage: Stage name (if provided, gets latest in stage)
        
        Returns:
            ModelVersion object
        """
        if stage:
            # Get latest version in stage
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"No model version found in stage: {stage}")
            return versions[0]
        
        elif version:
            # Get specific version
            return self.client.get_model_version(model_name, version)
        
        else:
            # Get latest version
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model: {model_name}")
            
            # Sort by version number (descending)
            versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
            return versions[0]
    
    def list_model_versions(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> List[ModelVersion]:
        """
        List all versions of a model
        
        Args:
            model_name: Registered model name
            stage: Filter by stage (optional)
        
        Returns:
            List of ModelVersion objects
        """
        if stage:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
        else:
            versions = self.client.search_model_versions(f"name='{model_name}'")
        
        # Sort by version number (descending)
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        
        return versions
    
    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ) -> ModelVersion:
        """
        Transition model version to a new stage
        
        Args:
            model_name: Registered model name
            version: Version number
            stage: Target stage (None, Staging, Production, Archived)
            archive_existing: Archive existing versions in target stage
        
        Returns:
            Updated ModelVersion object
        """
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
        
        logger.info(f"Transitioning {model_name} v{version} to {stage}")
        
        # Transition to new stage
        model_version = self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        
        logger.info(
            f"Model transitioned: {model_name} v{version} -> {stage}"
        )
        
        return model_version
    
    def promote_to_production(
        self,
        model_name: str,
        version: int,
        archive_existing: bool = True
    ) -> ModelVersion:
        """
        Promote a model version to production
        
        Args:
            model_name: Registered model name
            version: Version number
            archive_existing: Archive existing production versions
        
        Returns:
            Updated ModelVersion object
        """
        return self.transition_stage(
            model_name, version, "Production", archive_existing
        )
    
    def promote_to_staging(
        self,
        model_name: str,
        version: int
    ) -> ModelVersion:
        """
        Promote a model version to staging
        
        Args:
            model_name: Registered model name
            version: Version number
        
        Returns:
            Updated ModelVersion object
        """
        return self.transition_stage(model_name, version, "Staging", False)
    
    def archive_version(
        self,
        model_name: str,
        version: int
    ) -> ModelVersion:
        """
        Archive a model version
        
        Args:
            model_name: Registered model name
            version: Version number
        
        Returns:
            Updated ModelVersion object
        """
        return self.transition_stage(model_name, version, "Archived", False)
    
    def delete_model_version(
        self,
        model_name: str,
        version: int
    ) -> None:
        """
        Delete a model version (cannot be undone!)
        
        Args:
            model_name: Registered model name
            version: Version number
        """
        logger.warning(f"Deleting {model_name} v{version}")
        self.client.delete_model_version(model_name, version)
        logger.info(f"Model version deleted: {model_name} v{version}")
    
    def get_model_metadata(
        self,
        model_name: str,
        version: int
    ) -> Dict:
        """
        Get comprehensive metadata for a model version
        
        Args:
            model_name: Registered model name
            version: Version number
        
        Returns:
            Dictionary with model metadata
        """
        model_version = self.client.get_model_version(model_name, version)
        
        # Get run info
        run = self.client.get_run(model_version.run_id)
        
        metadata = {
            "name": model_name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "creation_timestamp": datetime.fromtimestamp(
                model_version.creation_timestamp / 1000
            ).isoformat(),
            "last_updated_timestamp": datetime.fromtimestamp(
                model_version.last_updated_timestamp / 1000
            ).isoformat(),
            "run_id": model_version.run_id,
            "run_name": run.info.run_name,
            "source": model_version.source,
            "status": model_version.status,
            "tags": model_version.tags,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        
        return metadata
    
    def compare_versions(
        self,
        model_name: str,
        version1: int,
        version2: int,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare metrics between two model versions
        
        Args:
            model_name: Registered model name
            version1: First version number
            version2: Second version number
            metrics: List of metrics to compare (if None, compares all)
        
        Returns:
            Dictionary with comparison results
        """
        # Get metadata for both versions
        meta1 = self.get_model_metadata(model_name, version1)
        meta2 = self.get_model_metadata(model_name, version2)
        
        # Compare metrics
        comparison = {
            "model_name": model_name,
            "version1": {
                "version": version1,
                "stage": meta1["stage"],
                "creation_time": meta1["creation_timestamp"],
                "metrics": {},
            },
            "version2": {
                "version": version2,
                "stage": meta2["stage"],
                "creation_time": meta2["creation_timestamp"],
                "metrics": {},
            },
            "differences": {},
        }
        
        # Get all metrics if not specified
        if metrics is None:
            metrics = set(meta1["metrics"].keys()) | set(meta2["metrics"].keys())
        
        for metric in metrics:
            val1 = meta1["metrics"].get(metric)
            val2 = meta2["metrics"].get(metric)
            
            comparison["version1"]["metrics"][metric] = val1
            comparison["version2"]["metrics"][metric] = val2
            
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else None
                comparison["differences"][metric] = {
                    "absolute": diff,
                    "percentage": pct_change,
                    "better": "v2" if diff > 0 else "v1" if diff < 0 else "equal"
                }
        
        return comparison
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get current production model version
        
        Args:
            model_name: Registered model name
        
        Returns:
            ModelVersion object or None if no production version
        """
        try:
            return self.get_model_version(model_name, stage="Production")
        except ValueError:
            return None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None
    ):
        """
        Load a model for inference
        
        Args:
            model_name: Registered model name
            version: Version number (if None, loads from stage)
            stage: Stage name (default: Production)
        
        Returns:
            Loaded model object
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            stage = stage or "Production"
            model_uri = f"models:/{model_name}/{stage}"
        
        logger.info(f"Loading model: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        return model


# Example usage
if __name__ == "__main__":
    # Initialize registry manager
    registry = ModelRegistryManager()
    
    # Example: Register a model from a run
    # run_id = "your-run-id-here"
    # model_version = registry.register_model(
    #     run_id=run_id,
    #     model_name="fraud-detector",
    #     description="Random Forest model for fraud detection",
    #     tags={
    #         "team": "ml-team",
    #         "project": "fraud-prevention",
    #         "framework": "sklearn"
    #     }
    # )
    
    # Example: List all versions
    # versions = registry.list_model_versions("fraud-detector")
    # for v in versions:
    #     print(f"Version {v.version}: {v.current_stage}")
    
    # Example: Promote to production
    # registry.promote_to_production("fraud-detector", version=2)
    
    # Example: Compare versions
    # comparison = registry.compare_versions("fraud-detector", 1, 2)
    # print(comparison)
    
    # Example: Load production model
    # model = registry.load_model("fraud-detector", stage="Production")
    # predictions = model.predict(test_data)
    
    print("Model Registry Manager ready!")
