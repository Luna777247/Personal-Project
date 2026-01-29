"""
LSTM AutoEncoder Model for Fraud Detection with Time Series
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


class LSTMAutoEncoderFraudDetector:
    """LSTM AutoEncoder for sequence-based fraud detection"""
    
    def __init__(self,
                 n_features: int,
                 sequence_length: int = 10,
                 lstm_units: List[int] = [64, 32],
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize LSTM AutoEncoder model
        
        Args:
            n_features: Number of features per timestep
            sequence_length: Length of input sequences
            lstm_units: LSTM layer units
            learning_rate: Learning rate
            random_state: Random seed
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.threshold = None
        self.model = self._build_model()
        
    def _build_model(self) -> keras.Model:
        """Build LSTM AutoEncoder architecture"""
        
        # Input
        input_layer = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Encoder
        encoded = input_layer
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            encoded = layers.LSTM(units, return_sequences=return_sequences, 
                                 dropout=0.2, recurrent_dropout=0.2)(encoded)
        
        # Repeat vector to create decoder input
        encoded = layers.RepeatVector(self.sequence_length)(encoded)
        
        # Decoder
        decoded = encoded
        for units in reversed(self.lstm_units):
            decoded = layers.LSTM(units, return_sequences=True,
                                 dropout=0.2, recurrent_dropout=0.2)(decoded)
        
        # Output
        decoded = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
        
        # Model
        model = keras.Model(inputs=input_layer, outputs=decoded, name='lstm_autoencoder')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(self, X: np.ndarray, customer_ids: np.ndarray = None) -> np.ndarray:
        """
        Create sequences from transaction data
        
        Args:
            X: Feature matrix
            customer_ids: Customer IDs for grouping transactions
            
        Returns:
            Sequence array of shape (n_sequences, sequence_length, n_features)
        """
        if customer_ids is None:
            # Simple sliding window without grouping
            sequences = []
            for i in range(len(X) - self.sequence_length + 1):
                sequences.append(X[i:i + self.sequence_length])
            return np.array(sequences)
        else:
            # Create sequences per customer
            sequences = []
            unique_customers = np.unique(customer_ids)
            
            for customer in unique_customers:
                customer_mask = customer_ids == customer
                customer_data = X[customer_mask]
                
                # Create sequences for this customer
                for i in range(len(customer_data) - self.sequence_length + 1):
                    sequences.append(customer_data[i:i + self.sequence_length])
            
            return np.array(sequences)
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series = None,
            customer_ids: np.ndarray = None,
            epochs: int = 50,
            batch_size: int = 128,
            validation_split: float = 0.2):
        """
        Train the LSTM AutoEncoder
        
        Args:
            X: Feature matrix
            y: Labels (optional, used to filter normal transactions)
            customer_ids: Customer IDs for sequence grouping
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        logger.info("Training LSTM AutoEncoder model...")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Train only on normal transactions if labels provided
        if y is not None:
            y_array = y.values if isinstance(y, pd.Series) else y
            normal_mask = y_array == 0
            X_normal = X_array[normal_mask]
            if customer_ids is not None:
                customer_ids_normal = customer_ids[normal_mask]
            else:
                customer_ids_normal = None
        else:
            X_normal = X_array
            customer_ids_normal = customer_ids
        
        # Create sequences
        X_sequences = self.create_sequences(X_normal, customer_ids_normal)
        logger.info(f"Created {len(X_sequences)} sequences")
        
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
            X_sequences, X_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold
        reconstructions = self.model.predict(X_sequences, verbose=0)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, 95)
        
        logger.info(f"Training completed. Threshold: {self.threshold:.4f}")
        
        return history
    
    def predict(self, X: pd.DataFrame, customer_ids: np.ndarray = None, threshold: float = None) -> np.ndarray:
        """
        Predict fraud labels
        
        Args:
            X: Feature matrix
            customer_ids: Customer IDs for sequence grouping
            threshold: Custom threshold
            
        Returns:
            Binary predictions
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Create sequences
        X_sequences = self.create_sequences(X_array, customer_ids)
        
        # Get reconstruction error
        reconstructions = self.model.predict(X_sequences, verbose=0)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=(1, 2))
        
        # Use threshold
        threshold = threshold or self.threshold
        if threshold is None:
            raise ValueError("Threshold not set. Train the model first or provide a threshold.")
        
        predictions = (mse > threshold).astype(int)
        
        # Map predictions back to original length
        # For simplicity, use last prediction for each position
        full_predictions = np.zeros(len(X_array))
        for i, pred in enumerate(predictions):
            full_predictions[i + self.sequence_length - 1] = pred
        
        return full_predictions.astype(int)
    
    def predict_proba(self, X: pd.DataFrame, customer_ids: np.ndarray = None) -> np.ndarray:
        """
        Predict fraud probability scores
        
        Args:
            X: Feature matrix
            customer_ids: Customer IDs for sequence grouping
            
        Returns:
            Anomaly scores
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Create sequences
        X_sequences = self.create_sequences(X_array, customer_ids)
        
        # Get reconstruction error
        reconstructions = self.model.predict(X_sequences, verbose=0)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=(1, 2))
        
        # Normalize
        if self.threshold is not None:
            scores = np.clip(mse / (self.threshold * 2), 0, 1)
        else:
            scores = mse / (np.max(mse) + 1e-8)
        
        # Map scores back to original length
        full_scores = np.zeros(len(X_array))
        for i, score in enumerate(scores):
            full_scores[i + self.sequence_length - 1] = score
        
        return full_scores
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, customer_ids: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            customer_ids: Customer IDs
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred = self.predict(X, customer_ids)
        y_proba = self.predict_proba(X, customer_ids)
        
        # Only evaluate on positions where we have predictions
        valid_mask = y_proba > 0
        
        if valid_mask.sum() == 0:
            logger.warning("No valid predictions available")
            return {}
        
        y_true = y.values[valid_mask] if isinstance(y, pd.Series) else y[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        y_proba_valid = y_proba[valid_mask]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_proba_valid),
            'f1_score': f1_score(y_true, y_pred_valid),
            'precision': precision_score(y_true, y_pred_valid),
            'recall': recall_score(y_true, y_pred_valid),
        }
        
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk"""
        self.model.save(filepath)
        with open(f"{filepath}_threshold.txt", 'w') as f:
            f.write(str(self.threshold))
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
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
    df = df.sort_values(['customer_id', 'timestamp'])
    
    # Create features
    feature_engineer = TransactionFeatureEngineer()
    df_features = feature_engineer.create_all_features(df)
    
    # Select numeric features
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['is_fraud']]
    
    X = df_features[numeric_cols].fillna(0)
    y = df_features['is_fraud']
    customer_ids = df_features['customer_id'].values
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    cust_train, cust_test = customer_ids[:split_idx], customer_ids[split_idx:]
    
    # Train model
    model = LSTMAutoEncoderFraudDetector(
        n_features=X_train.shape[1],
        sequence_length=5,
        lstm_units=[32, 16]
    )
    
    model.fit(X_train, y_train, cust_train, epochs=10, batch_size=64)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test, cust_test)
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
