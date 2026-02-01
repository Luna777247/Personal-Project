"""
MLflow utilities for model tracking and registry
"""
import mlflow
import mlflow.sklearn
import mlflow.keras
from mlflow.tracking import MlflowClient
import logging
from typing import Dict, Any, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking and model registry"""
    
    def __init__(self, 
                 tracking_uri: str = "mlflow/",
                 experiment_name: str = "fraud_detection"):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient(tracking_uri=tracking_uri)
        
        logger.info(f"MLflow tracking initialized: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    
    def log_sklearn_model(self, model, artifact_path: str = "model", **kwargs):
        """Log scikit-learn model"""
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        logger.info(f"Logged sklearn model to {artifact_path}")
    
    def log_keras_model(self, model, artifact_path: str = "model", **kwargs):
        """Log Keras model"""
        mlflow.keras.log_model(model, artifact_path, **kwargs)
        logger.info(f"Logged keras model to {artifact_path}")
    
    def register_model(self, 
                      model_uri: str, 
                      model_name: str,
                      tags: Optional[Dict] = None,
                      description: Optional[str] = None):
        """
        Register model to MLflow Model Registry
        
        Args:
            model_uri: URI of the model
            model_name: Name for the registered model
            tags: Tags for the model version
            description: Description of the model
            
        Returns:
            ModelVersion object
        """
        result = mlflow.register_model(model_uri, model_name)
        
        # Add tags and description
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    model_name, result.version, key, value
                )
        
        if description:
            self.client.update_model_version(
                model_name, result.version, description=description
            )
        
        logger.info(f"Registered model: {model_name} version {result.version}")
        return result
    
    def transition_model_stage(self, 
                              model_name: str, 
                              version: str, 
                              stage: str):
        """
        Transition model to a different stage
        
        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Target stage (Staging, Production, Archived)
        """
        self.client.transition_model_version_stage(
            model_name, version, stage, archive_existing_versions=False
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_model_version(self, model_name: str, stage: str = "Production"):
        """
        Get model version by stage
        
        Args:
            model_name: Name of the registered model
            stage: Stage to retrieve (Production, Staging, etc.)
            
        Returns:
            Model version object
        """
        versions = self.client.get_latest_versions(model_name, stages=[stage])
        if versions:
            return versions[0]
        return None
    
    def load_model(self, model_uri: str):
        """Load model from MLflow"""
        return mlflow.pyfunc.load_model(model_uri)
    
    def compare_runs(self, run_ids: list, metric_names: list):
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs
            metric_names: List of metric names to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                'name': run.data.tags.get('mlflow.runName', 'Unknown'),
                'metrics': {metric: run.data.metrics.get(metric) 
                          for metric in metric_names}
            }
        
        return comparison
    
    def search_runs(self, filter_string: str = "", max_results: int = 10):
        """
        Search runs in the experiment
        
        Args:
            filter_string: Filter string (e.g., "metrics.roc_auc > 0.9")
            max_results: Maximum number of results
            
        Returns:
            List of runs
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=["metrics.roc_auc DESC"]
        )
        return runs
    
    def delete_run(self, run_id: str):
        """Delete a run"""
        self.client.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")


def setup_mlflow(tracking_uri: str = None, 
                 experiment_name: str = None) -> MLflowTracker:
    """
    Setup MLflow tracking
    
    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name
        
    Returns:
        MLflowTracker instance
    """
    tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'mlflow/')
    experiment_name = experiment_name or os.getenv('MLFLOW_EXPERIMENT_NAME', 'fraud_detection')
    
    return MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)


if __name__ == "__main__":
    # Test MLflow tracking
    tracker = setup_mlflow()
    
    # Start a test run
    with tracker.start_run(run_name="test_run"):
        # Log parameters
        tracker.log_params({
            'model_type': 'isolation_forest',
            'contamination': 0.01,
            'n_estimators': 100
        })
        
        # Log metrics
        tracker.log_metrics({
            'roc_auc': 0.95,
            'f1_score': 0.85,
            'precision': 0.90
        })
        
        print("Test run logged successfully")
    
    # Search runs
    runs = tracker.search_runs(max_results=5)
    print(f"\nFound {len(runs)} runs")
