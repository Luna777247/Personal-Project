"""
Example: Train Fraud Detection Model
Demonstrates complete training workflow with MLflow
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import mlflow

from src.training.trainer import RandomForestTrainer, XGBoostTrainer


def generate_fraud_data(n_samples: int = 10000):
    """Generate synthetic fraud detection dataset"""
    
    # Generate classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],  # Imbalanced (95% normal, 5% fraud)
        random_state=42
    )
    
    # Create feature names
    feature_names = [
        'amount', 'hour_of_day', 'day_of_week', 'is_international',
        'merchant_category', 'card_type', 'transaction_type',
        'device_score', 'ip_risk_score', 'email_risk_score',
        'velocity_1h', 'velocity_24h', 'avg_amount_30d',
        'merchant_risk_score', 'distance_from_home',
        'unusual_time', 'card_present', 'cvv_provided',
        'address_match', 'account_age_days'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
    df['is_fraud'] = y
    
    return df


def main():
    """Main training pipeline"""
    
    print("=" * 80)
    print("Fraud Detection Model Training Pipeline")
    print("=" * 80)
    
    # Set MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"\n✓ MLflow Tracking URI: {mlflow_uri}")
    
    # Generate data
    print("\n[1/5] Generating synthetic fraud detection data...")
    df = generate_fraud_data(n_samples=10000)
    print(f"✓ Generated {len(df)} samples")
    print(f"  - Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"  - Normal cases: {(1-df['is_fraud']).sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")
    
    # Split data
    print("\n[2/5] Splitting data...")
    X = df.drop('is_fraud', axis=1).values
    y = df['is_fraud'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train Random Forest
    print("\n[3/5] Training Random Forest model...")
    rf_trainer = RandomForestTrainer(
        experiment_name="fraud-detection",
        tracking_uri=mlflow_uri
    )
    
    rf_metrics = rf_trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        run_name="random-forest-baseline"
    )
    
    print(f"✓ Random Forest trained")
    print(f"  Validation Metrics:")
    for metric, value in rf_metrics.items():
        print(f"    - {metric}: {value:.4f}")
    
    # Train XGBoost
    print("\n[4/5] Training XGBoost model...")
    xgb_trainer = XGBoostTrainer(
        experiment_name="fraud-detection",
        tracking_uri=mlflow_uri
    )
    
    xgb_metrics = xgb_trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        run_name="xgboost-baseline"
    )
    
    print(f"✓ XGBoost trained")
    print(f"  Validation Metrics:")
    for metric, value in xgb_metrics.items():
        print(f"    - {metric}: {value:.4f}")
    
    # Evaluate on test set
    print("\n[5/5] Final evaluation on test set...")
    
    # RF test metrics
    rf_test_metrics = rf_trainer.evaluate(X_test, y_test)
    print(f"\nRandom Forest Test Metrics:")
    for metric, value in rf_test_metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    # XGB test metrics
    xgb_test_metrics = xgb_trainer.evaluate(X_test, y_test)
    print(f"\nXGBoost Test Metrics:")
    for metric, value in xgb_test_metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Compare models
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'Metric': list(rf_test_metrics.keys()),
        'Random Forest': list(rf_test_metrics.values()),
        'XGBoost': list(xgb_test_metrics.values())
    })
    print(comparison.to_string(index=False))
    
    # Determine best model
    best_model = "XGBoost" if xgb_test_metrics['f1_score'] > rf_test_metrics['f1_score'] else "Random Forest"
    print(f"\n✓ Best Model: {best_model}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"View results in MLflow UI: {mlflow_uri}")
    print("=" * 80)


if __name__ == "__main__":
    main()
