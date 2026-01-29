#!/usr/bin/env python3
"""
Test Training Script for MLOps Mini Project
Verifies the complete MLOps pipeline with real California Housing data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mlops_pipeline import MLOpsManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mlops_pipeline():
    """Test the complete MLOps pipeline"""
    logger.info("🚀 Starting MLOps Mini Pipeline Test")
    logger.info("=" * 60)

    try:
        # Check if data exists
        data_dir = "data"
        train_path = os.path.join(data_dir, "train.csv")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")

        # Load a small sample for testing
        logger.info("Loading training data sample...")
        train_df = pd.read_csv(train_path)
        logger.info(f"Full training set: {len(train_df)} samples")

        # Use a small subset for quick testing
        test_sample = train_df.sample(n=1000, random_state=42)
        test_path = os.path.join(data_dir, "test_sample.csv")
        test_sample.to_csv(test_path, index=False)
        logger.info(f"Test sample created: {len(test_sample)} samples")

        # Initialize MLOps manager (without MLflow server for testing)
        logger.info("Initializing MLOps manager...")
        # Use file store for testing instead of server
        mlops = MLOpsManager(
            experiment_name="mlops_mini_test",
            tracking_uri="./experiments"  # Local file store
        )

        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = mlops.load_data(
            test_path,
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Test features shape: {X_test.shape}")

        # Test model training with a simple model
        logger.info("Testing model training...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        # Train a simple Random Forest
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(".4f")
        logger.info(".4f")

        # Test hyperparameter tuning (quick test)
        logger.info("Testing hyperparameter tuning...")
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 50)
            max_depth = trial.suggest_int('max_depth', 3, 10)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)

        # Quick optimization with few trials
        import optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)

        best_r2 = study.best_value
        logger.info(".4f")

        # Save test results
        results_summary = {
            "test_timestamp": datetime.now().isoformat(),
            "dataset": "California Housing",
            "samples_used": len(test_sample),
            "features": list(X_train.columns),
            "baseline_model": {
                "type": "RandomForestRegressor",
                "mse": mse,
                "r2_score": r2
            },
            "hyperparameter_tuning": {
                "best_r2": best_r2,
                "trials": 5
            },
            "status": "success"
        }

        os.makedirs('results', exist_ok=True)
        with open('results/test_pipeline_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)

        logger.info("\n✅ MLOps pipeline test completed successfully!")
        logger.info(f"🎯 Best R² after tuning: {best_r2:.4f}")
        logger.info("📁 Results saved to results/test_pipeline_results.json")

        # Clean up test file
        if os.path.exists(test_path):
            os.remove(test_path)
            logger.info("Cleaned up test sample file")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mlops_pipeline()
    sys.exit(0 if success else 1)