"""
MLflow Model Trainer
Trains models with automatic experiment tracking and artifact logging
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import pickle

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

from loguru import logger


class ModelTrainer:
    """Base trainer class with MLflow integration"""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://localhost:5000",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize trainer with MLflow configuration
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: S3 or local path for artifacts
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                self.experiment = mlflow.get_experiment(experiment_id)
            
            mlflow.set_experiment(experiment_name)
            logger.info(f"Using experiment: {experiment_name} (ID: {self.experiment.experiment_id})")
            
        except Exception as e:
            logger.error(f"Failed to setup experiment: {e}")
            raise
        
        self.model = None
        self.model_name = None
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    def log_artifacts(self, artifact_dir: Path) -> None:
        """Log directory of artifacts to MLflow"""
        if artifact_dir.exists():
            mlflow.log_artifacts(str(artifact_dir))
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for binary classification)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add AUC for binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def save_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        artifact_dir: Path
    ) -> None:
        """Save confusion matrix as artifact"""
        cm = confusion_matrix(y_true, y_pred)
        
        artifact_dir.mkdir(parents=True, exist_ok=True)
        cm_path = artifact_dir / "confusion_matrix.txt"
        
        with open(cm_path, 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(y_true, y_pred))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Train model (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)
        
        return metrics


class RandomForestTrainer(ModelTrainer):
    """Random Forest model trainer"""
    
    def __init__(
        self,
        experiment_name: str = "random-forest-classifier",
        tracking_uri: str = "http://localhost:5000",
        **kwargs
    ):
        super().__init__(experiment_name, tracking_uri, **kwargs)
        self.model_name = "random_forest"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        random_state: int = 42,
        run_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train Random Forest model with MLflow logging
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            max_features: Features to consider for split
            random_state: Random seed
            run_name: Custom run name
        
        Returns:
            Dictionary of metrics
        """
        with mlflow.start_run(run_name=run_name or f"rf-{n_estimators}-{max_depth}"):
            # Log parameters
            params = {
                "model_type": "RandomForest",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "random_state": random_state,
            }
            self.log_params(params)
            
            # Train model
            logger.info("Training Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            logger.info("Training completed")
            
            # Evaluate on training set
            y_train_pred = self.model.predict(X_train)
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            train_metrics = self.compute_metrics(y_train, y_train_pred, y_train_proba)
            
            # Log training metrics
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                y_val_proba = self.model.predict_proba(X_val)[:, 1]
                val_metrics = self.compute_metrics(y_val, y_val_pred, y_val_proba)
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"val_{key}", value)
                
                # Save confusion matrix
                artifact_dir = Path("artifacts")
                self.save_confusion_matrix(y_val, y_val_pred, artifact_dir)
                self.log_artifacts(artifact_dir)
                
                logger.info(f"Validation metrics: {val_metrics}")
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=self.experiment_name
            )
            
            # Log feature importances
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                artifact_dir = Path("artifacts")
                artifact_dir.mkdir(exist_ok=True)
                feature_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
                self.log_artifacts(artifact_dir)
            
            logger.info(f"Model logged to MLflow. Run ID: {mlflow.active_run().info.run_id}")
            
            return val_metrics if X_val is not None else train_metrics


class XGBoostTrainer(ModelTrainer):
    """XGBoost model trainer"""
    
    def __init__(
        self,
        experiment_name: str = "xgboost-classifier",
        tracking_uri: str = "http://localhost:5000",
        **kwargs
    ):
        super().__init__(experiment_name, tracking_uri, **kwargs)
        self.model_name = "xgboost"
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        run_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train XGBoost model with MLflow logging
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Feature subsample ratio
            random_state: Random seed
            run_name: Custom run name
        
        Returns:
            Dictionary of metrics
        """
        # Enable MLflow autologging for XGBoost
        mlflow.xgboost.autolog()
        
        with mlflow.start_run(run_name=run_name or f"xgb-{learning_rate}-{max_depth}"):
            # Log parameters
            params = {
                "model_type": "XGBoost",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "random_state": random_state,
            }
            self.log_params(params)
            
            # Train model
            logger.info("Training XGBoost model...")
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            # Prepare eval set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            logger.info("Training completed")
            
            # Evaluate
            if X_val is not None and y_val is not None:
                y_val_pred = self.model.predict(X_val)
                y_val_proba = self.model.predict_proba(X_val)[:, 1]
                val_metrics = self.compute_metrics(y_val, y_val_pred, y_val_proba)
                
                # Save confusion matrix
                artifact_dir = Path("artifacts")
                self.save_confusion_matrix(y_val, y_val_pred, artifact_dir)
                self.log_artifacts(artifact_dir)
                
                logger.info(f"Validation metrics: {val_metrics}")
                return val_metrics
            
            # Evaluate on training set if no validation set
            y_train_pred = self.model.predict(X_train)
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            train_metrics = self.compute_metrics(y_train, y_train_pred, y_train_proba)
            
            return train_metrics


# Example usage
if __name__ == "__main__":
    # Generate dummy data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    rf_trainer = RandomForestTrainer(experiment_name="fraud-detection")
    rf_metrics = rf_trainer.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        n_estimators=100,
        max_depth=10,
        run_name="rf-baseline"
    )
    
    print(f"\nRandom Forest Metrics: {rf_metrics}")
    
    # Train XGBoost
    xgb_trainer = XGBoostTrainer(experiment_name="fraud-detection")
    xgb_metrics = xgb_trainer.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        run_name="xgb-baseline"
    )
    
    print(f"\nXGBoost Metrics: {xgb_metrics}")
