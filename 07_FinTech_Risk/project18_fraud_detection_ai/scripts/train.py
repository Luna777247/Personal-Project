"""
Training script for fraud detection models
Train and evaluate multiple models with MLflow tracking
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import argparse
import logging
from datetime import datetime

from src.data_generator import FraudDataGenerator
from src.feature_engineering import TransactionFeatureEngineer
from src.model_isolation_forest import IsolationForestFraudDetector
from src.model_autoencoder import AutoEncoderFraudDetector
from src.model_lstm_autoencoder import LSTMAutoEncoderFraudDetector
from src.mlflow_utils import setup_mlflow
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_or_load_data(config, force_generate=False):
    """Generate or load transaction data"""
    raw_data_path = os.path.join(config['data']['raw_data_path'], 'transactions.csv')
    
    if os.path.exists(raw_data_path) and not force_generate:
        logger.info(f"Loading data from {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        logger.info("Generating synthetic transaction data...")
        generator = FraudDataGenerator(random_seed=config['data']['random_state'])
        df = generator.generate_transactions(n_samples=50000, fraud_ratio=0.02)
        
        # Save data
        os.makedirs(config['data']['raw_data_path'], exist_ok=True)
        df.to_csv(raw_data_path, index=False)
        logger.info(f"Data saved to {raw_data_path}")
    
    return df


def train_isolation_forest(X_train, X_test, y_train, y_test, config, tracker):
    """Train Isolation Forest model"""
    logger.info("=" * 60)
    logger.info("Training Isolation Forest Model")
    logger.info("=" * 60)
    
    # Get model config
    model_config = config['models']['isolation_forest']
    
    # Start MLflow run
    with tracker.start_run(run_name="isolation_forest"):
        # Log parameters
        tracker.log_params({
            'model_type': 'isolation_forest',
            **model_config,
            'n_samples': len(X_train)
        })
        
        # Train model
        model = IsolationForestFraudDetector(**model_config)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        tracker.log_metrics(metrics)
        
        # Log model
        model_path = f"models/isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        tracker.log_artifact(model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
    return model, metrics


def train_autoencoder(X_train, X_test, y_train, y_test, config, tracker):
    """Train AutoEncoder model"""
    logger.info("=" * 60)
    logger.info("Training AutoEncoder Model")
    logger.info("=" * 60)
    
    # Get model config
    model_config = config['models']['autoencoder']
    
    # Start MLflow run
    with tracker.start_run(run_name="autoencoder"):
        # Log parameters
        tracker.log_params({
            'model_type': 'autoencoder',
            **model_config,
            'n_samples': len(X_train),
            'input_dim': X_train.shape[1]
        })
        
        # Train model
        model = AutoEncoderFraudDetector(
            input_dim=X_train.shape[1],
            encoding_dim=model_config['encoding_dim'],
            hidden_layers=model_config['hidden_layers'],
            learning_rate=model_config['learning_rate']
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            validation_split=model_config['validation_split']
        )
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        tracker.log_metrics(metrics)
        
        # Log training history
        for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], 
                                                     history.history['val_loss'])):
            tracker.log_metrics({
                'train_loss': loss,
                'val_loss': val_loss
            }, step=epoch)
        
        # Log model
        model_path = f"models/autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        tracker.log_artifact(model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
    return model, metrics


def train_lstm_autoencoder(X_train, X_test, y_train, y_test, df_train, df_test, config, tracker):
    """Train LSTM AutoEncoder model"""
    logger.info("=" * 60)
    logger.info("Training LSTM AutoEncoder Model")
    logger.info("=" * 60)
    
    # Get model config
    model_config = config['models']['lstm_autoencoder']
    
    # Start MLflow run
    with tracker.start_run(run_name="lstm_autoencoder"):
        # Log parameters
        tracker.log_params({
            'model_type': 'lstm_autoencoder',
            **model_config,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        })
        
        # Train model
        model = LSTMAutoEncoderFraudDetector(
            n_features=X_train.shape[1],
            sequence_length=model_config['sequence_length'],
            lstm_units=model_config['lstm_units'],
            learning_rate=model_config['learning_rate']
        )
        
        # Get customer IDs
        customer_ids_train = df_train['customer_id'].values
        customer_ids_test = df_test['customer_id'].values
        
        history = model.fit(
            X_train, y_train,
            customer_ids=customer_ids_train,
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size']
        )
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test, customer_ids=customer_ids_test)
        tracker.log_metrics(metrics)
        
        # Log model
        model_path = f"models/lstm_autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        tracker.log_artifact(model_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
    return model, metrics


def main(args):
    """Main training pipeline"""
    logger.info("Starting Fraud Detection Model Training")
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup MLflow
    tracker = setup_mlflow()
    
    # Generate or load data
    df = generate_or_load_data(config, force_generate=args.generate_data)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    
    # Feature engineering
    logger.info("Creating features...")
    feature_engineer = TransactionFeatureEngineer()
    df_features = feature_engineer.create_all_features(
        df, 
        fit=True, 
        config=config['features']
    )
    
    # Save feature engineer
    os.makedirs("models", exist_ok=True)
    joblib.dump(feature_engineer, "models/feature_engineer.pkl")
    logger.info("Feature engineer saved")
    
    # Select features
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'customer_id', 'merchant_id']]
    
    X = df_features[numeric_cols].fillna(0)
    y = df_features['is_fraud']
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(numeric_cols)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    # Also split original dataframe for LSTM
    df_train = df_features.iloc[X_train.index]
    df_test = df_features.iloc[X_test.index]
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train models
    results = {}
    
    if args.model == 'all' or args.model == 'isolation_forest':
        model_if, metrics_if = train_isolation_forest(
            X_train, X_test, y_train, y_test, config, tracker
        )
        results['isolation_forest'] = metrics_if
    
    if args.model == 'all' or args.model == 'autoencoder':
        model_ae, metrics_ae = train_autoencoder(
            X_train, X_test, y_train, y_test, config, tracker
        )
        results['autoencoder'] = metrics_ae
    
    if args.model == 'all' or args.model == 'lstm':
        model_lstm, metrics_lstm = train_lstm_autoencoder(
            X_train, X_test, y_train, y_test, df_train, df_test, config, tracker
        )
        results['lstm_autoencoder'] = metrics_lstm
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
    logger.info(f"\nBest Model: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")
    
    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'isolation_forest', 'autoencoder', 'lstm'],
                       help='Model to train')
    parser.add_argument('--generate-data', action='store_true',
                       help='Force generate new data')
    
    args = parser.parse_args()
    main(args)
