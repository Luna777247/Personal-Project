"""
AutoEncoder Model for Fraud Detection
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoEncoderFraudDetector:
    """Deep AutoEncoder for anomaly-based fraud detection"""
    
    def __init__(self,
                 input_dim: int,
                 encoding_dim: int = 32,
                 hidden_layers: List[int] = [64, 32, 16],
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize AutoEncoder model
        
        Args:
            input_dim: Input feature dimension
            encoding_dim: Encoding layer dimension
            hidden_layers: Hidden layer dimensions
            learning_rate: Learning rate
            random_state: Random seed
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.threshold = None
        self.model = self._build_model()
        
    def _build_model(self) -> keras.Model:
        """Build AutoEncoder architecture"""
        
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = input_layer
        
        for units in self.hidden_layers:
            encoded = layers.Dense(units, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(0.2)(encoded)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(encoded)
        
        # Decoder
        decoded = encoded
        
        for units in reversed(self.hidden_layers):
            decoded = layers.Dense(units, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(0.2)(decoded)
        
        # Output
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Model
        autoencoder = keras.Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series = None,
            epochs: int = 50,
            batch_size: int = 256,
            validation_split: float = 0.2):
        """
        Train the AutoEncoder on normal transactions
        
        Args:
            X: Feature matrix (should contain mostly normal transactions)
            y: Labels (optional, used to filter normal transactions)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        logger.info("Training AutoEncoder model...")
        
        # Train only on normal transactions if labels provided
        if y is not None:
            X_normal = X[y == 0].values
            logger.info(f"Training on {len(X_normal)} normal transactions")
        else:
            X_normal = X.values
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = self.model.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold on normal data
        reconstructions = self.model.predict(X_normal, verbose=0)
        mse = np.mean(np.power(X_normal - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile
        
        logger.info(f"Training completed. Threshold: {self.threshold:.4f}")
        
        return history
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict fraud labels
        
        Args:
            X: Feature matrix
            threshold: Custom threshold (uses trained threshold if None)
            
        Returns:
            Binary predictions (0: normal, 1: fraud)
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get reconstruction error
        reconstructions = self.model.predict(X_array, verbose=0)
        mse = np.mean(np.power(X_array - reconstructions, 2), axis=1)
        
        # Use threshold
        threshold = threshold or self.threshold
        if threshold is None:
            raise ValueError("Threshold not set. Train the model first or provide a threshold.")
        
        return (mse > threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability scores
        
        Args:
            X: Feature matrix
            
        Returns:
            Anomaly scores (reconstruction error)
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Get reconstruction error
        reconstructions = self.model.predict(X_array, verbose=0)
        mse = np.mean(np.power(X_array - reconstructions, 2), axis=1)
        
        # Normalize to 0-1 range
        if self.threshold is not None:
            scores = np.clip(mse / (self.threshold * 2), 0, 1)
        else:
            scores = mse / (np.max(mse) + 1e-8)
        
        return scores
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Custom threshold
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        # Get predictions and probabilities
        y_pred = self.predict(X, threshold=threshold)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y, y_proba),
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'threshold': threshold or self.threshold
        }
        
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk"""
        self.model.save(filepath)
        # Save threshold separately
        with open(f"{filepath}_threshold.txt", 'w') as f:
            f.write(str(self.threshold))
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
        # Load threshold
        try:
            with open(f"{filepath}_threshold.txt", 'r') as f:
                self.threshold = float(f.read())
        except FileNotFoundError:
            logger.warning("Threshold file not found")
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
    model = AutoEncoderFraudDetector(
        input_dim=X_train.shape[1],
        encoding_dim=32,
        hidden_layers=[64, 32, 16]
    )
    
    model.fit(X_train, y_train, epochs=20, batch_size=256)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
