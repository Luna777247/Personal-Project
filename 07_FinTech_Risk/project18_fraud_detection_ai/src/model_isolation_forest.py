"""
Isolation Forest Model for Fraud Detection
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsolationForestFraudDetector:
    """Isolation Forest for anomaly-based fraud detection"""
    
    def __init__(self, 
                 contamination: float = 0.01,
                 n_estimators: int = 200,
                 max_samples: int = 256,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize Isolation Forest model
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of trees
            max_samples: Number of samples to draw
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        self.feature_names = None
        self.threshold = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the model
        
        Args:
            X: Feature matrix
            y: Labels (not used, for compatibility)
        """
        logger.info("Training Isolation Forest model...")
        self.feature_names = X.columns.tolist()
        self.model.fit(X)
        logger.info("Training completed")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud labels
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0: normal, 1: fraud)
        """
        predictions = self.model.predict(X)
        # Convert: -1 (outlier/fraud) to 1, 1 (inlier/normal) to 0
        return np.where(predictions == -1, 1, 0)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability scores
        
        Args:
            X: Feature matrix
            
        Returns:
            Anomaly scores (higher = more likely fraud)
        """
        # Get anomaly scores (negative scores = outliers)
        scores = self.model.score_samples(X)
        # Convert to probability-like scores (0 to 1)
        # More negative = more anomalous = higher fraud probability
        proba = 1 / (1 + np.exp(scores))  # Sigmoid transformation
        return proba
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        # Get predictions and probabilities
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y, y_proba),
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        })
        
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test with sample data
    from feature_engineering import generate_sample_data, TransactionFeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Generate data
    df = generate_sample_data(n_samples=5000)
    
    # Create features
    feature_engineer = TransactionFeatureEngineer()
    df_features = feature_engineer.create_all_features(df)
    
    # Select numeric features
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'customer_id', 'merchant_id']]
    
    X = df_features[numeric_cols].fillna(0)
    y = df_features['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = IsolationForestFraudDetector(contamination=0.01)
    model.fit(X_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
