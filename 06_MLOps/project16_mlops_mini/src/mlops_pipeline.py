import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLOpsManager:
    def __init__(self, experiment_name="mlops_mini_experiment", tracking_uri="http://localhost:5000"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")

    def load_data(self, filepath, target_column='target', test_size=0.2, random_state=42):
        """Load and split dataset"""
        logger.info(f"Loading data from {filepath}")

        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical variables
        X = self._preprocess_features(X)

        # Split data - only stratify for classification tasks
        if y.dtype in ['int64', 'int32'] and len(y.unique()) < 20:  # Likely classification
            stratify_param = y
            logger.info("Detected classification task - using stratification")
        else:  # Regression or too many classes
            stratify_param = None
            logger.info("Detected regression task or too many classes - not using stratification")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def _preprocess_features(self, X):
        """Preprocess features: encode categorical, scale numerical"""
        X_processed = X.copy()

        for column in X_processed.columns:
            if X_processed[column].dtype == 'object':
                # Label encode categorical features
                le = LabelEncoder()
                X_processed[column] = le.fit_transform(X_processed[column].astype(str))
            elif X_processed[column].dtype in ['int64', 'float64']:
                # Scale numerical features
                scaler = StandardScaler()
                X_processed[column] = scaler.fit_transform(X_processed[column].values.reshape(-1, 1))

        return X_processed

    def train_and_log_model(self, model, model_name, X_train, X_test, y_train, y_test,
                          hyperparameters=None):
        """Train model and log to MLflow"""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            run_id = run.info.run_id
            logger.info(f"Started run: {run_id}")

            # Log hyperparameters
            if hyperparameters:
                mlflow.log_params(hyperparameters)
            else:
                mlflow.log_param("model_type", model_name)

            # Log dataset info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("features", X_train.shape[1])

            # Measure training time
            start_time = time.time()

            # Train model
            model.fit(X_train, y_train)

            training_time = time.time() - start_time
            mlflow.log_metric("training_time", training_time)

            # Make predictions
            inference_start = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            inference_time = time.time() - inference_start

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log model
            mlflow.sklearn.log_model(model, f"{model_name}_model")

            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Save feature importance plot
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
                plt.title(f'Feature Importance - {model_name}')
                plt.tight_layout()
                plt.savefig('feature_importance.png')
                mlflow.log_artifact('feature_importance.png')
                plt.close()

            # Log confusion matrix
            self._log_confusion_matrix(y_test, y_pred, model_name)

            logger.info(f"Completed run {run_id} - Accuracy: {metrics['accuracy']:.4f}")

            return run_id, metrics

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])

        return metrics

    def _log_confusion_matrix(self, y_true, y_pred, model_name):
        """Log confusion matrix as artifact"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()

    def hyperparameter_tuning(self, model_class, X_train, y_train, param_space,
                            n_trials=50, metric='f1_score'):
        """Perform hyperparameter tuning with Optuna"""
        def objective(trial):
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name,
                                                         param_config['low'],
                                                         param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name,
                                                           param_config['low'],
                                                           param_config['high'],
                                                           log=param_config.get('log', False))
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name,
                                                                 param_config['choices'])

            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring=metric)
            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best {metric}: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        return study.best_params, study.best_value

    def compare_models(self, models_dict, X_train, X_test, y_train, y_test):
        """Compare multiple models and log results"""
        results = {}

        for model_name, model_config in models_dict.items():
            logger.info(f"Training {model_name}...")

            model_class = model_config['class']
            params = model_config.get('params', {})

            model = model_class(**params)

            run_id, metrics = self.train_and_log_model(
                model, model_name, X_train, X_test, y_train, y_test, params
            )

            results[model_name] = {
                'run_id': run_id,
                'metrics': metrics
            }

        # Create comparison table
        comparison_df = pd.DataFrame({
            model_name: results[model_name]['metrics']
            for model_name in results.keys()
        }).T

        comparison_df = comparison_df.round(4)
        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string())

        # Save comparison to file
        comparison_df.to_csv('model_comparison.csv')
        mlflow.log_artifact('model_comparison.csv')

        return results

    def deploy_best_model(self, experiment_name=None, metric='f1_score'):
        """Deploy the best performing model"""
        if experiment_name is None:
            experiment_name = self.experiment_name

        # Get all runs from experiment
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])

        if runs.empty:
            logger.error("No runs found in experiment")
            return None

        # Find best run based on metric
        best_run = runs.loc[runs[metric].idxmax()]
        best_run_id = best_run.run_id

        logger.info(f"Best model run ID: {best_run_id} with {metric}: {best_run[metric]:.4f}")

        # Load best model
        model_uri = f"runs:/{best_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        # Save model for deployment
        model_path = f"models/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(model_path, exist_ok=True)

        mlflow.sklearn.save_model(model, model_path)

        logger.info(f"Best model saved to {model_path}")

        return model, model_path

    def create_model_registry_entry(self, model_path, model_name, description=""):
        """Register model in MLflow Model Registry"""
        try:
            model_version = mlflow.register_model(model_path, model_name)
            logger.info(f"Model registered: {model_version.name} v{model_version.version}")

            # Add description
            if description:
                self.client.update_model_version(
                    name=model_version.name,
                    version=model_version.version,
                    description=description
                )

            return model_version
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            return None

def get_default_models():
    """Get default model configurations for comparison"""
    return {
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {
                'random_state': 42,
                'max_iter': 1000
            }
        },
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'LightGBM': {
            'class': lgb.LGBMClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
    }

def main():
    """Main MLOps pipeline"""
    # Initialize MLOps manager
    mlops = MLOpsManager()

    # Load data (replace with your dataset)
    try:
        X_train, X_test, y_train, y_test = mlops.load_data('data/train.csv')
    except FileNotFoundError:
        logger.error("Dataset not found. Please add your dataset to data/train.csv")
        logger.info("Expected format: CSV with features and 'target' column")
        return

    # Get default models for comparison
    models = get_default_models()

    # Compare models
    logger.info("Starting model comparison...")
    results = mlops.compare_models(models, X_train, X_test, y_train, y_test)

    # Deploy best model
    logger.info("Deploying best model...")
    best_model, model_path = mlops.deploy_best_model()

    # Register model
    if best_model is not None:
        model_version = mlops.create_model_registry_entry(
            model_path,
            "best_classifier",
            "Best performing model from automated comparison"
        )

    logger.info("MLOps pipeline completed!")

if __name__ == "__main__":
    main()