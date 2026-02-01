"""
Model training for credit scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import joblib
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train credit scoring models"""
    
    def __init__(self, config: Dict):
        """
        Initialize model trainer
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.model_type = config.get('model', {}).get('type', 'xgboost')
        self.cv_scores = None
        
        # MLflow setup
        if config.get('monitoring', {}).get('mlflow_tracking', True):
            mlflow.set_tracking_uri(config.get('mlflow_uri', 'file:./mlruns'))
            mlflow.set_experiment(config.get('experiment_name', 'credit_scoring'))
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """
        Train credit scoring model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_type} model")
        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape if X_val is not None else 'None'}")
        
        with mlflow.start_run(run_name=f"{self.model_type}_training"):
            # Log parameters
            mlflow.log_params(self._get_model_params())
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", X_train.shape[0])
            
            # Train model
            if self.model_type == 'xgboost':
                metrics = self._train_xgboost(X_train, y_train, X_val, y_val)
            elif self.model_type == 'lightgbm':
                metrics = self._train_lightgbm(X_train, y_train, X_val, y_val)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Cross-validation
            cv_metrics = self._cross_validate(X_train, y_train)
            metrics.update(cv_metrics)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            if self.model_type == 'xgboost':
                mlflow.xgboost.log_model(self.model, "model")
            elif self.model_type == 'lightgbm':
                mlflow.lightgbm.log_model(self.model, "model")
            
            logger.info(f"Training completed. AUC: {metrics.get('auc', 0):.4f}")
            
            return metrics
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict:
        """Train XGBoost model"""
        params = self.config.get('model', {}).get('xgboost', {})
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Evaluation list
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=evals,
            early_stopping_rounds=params.get('early_stopping_rounds', 10),
            verbose_eval=False
        )
        
        # Predictions
        y_train_pred = self.model.predict(dtrain)
        metrics = {
            'train_auc': roc_auc_score(y_train, y_train_pred)
        }
        
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(dval)
            metrics.update({
                'val_auc': roc_auc_score(y_val, y_val_pred),
                'auc': roc_auc_score(y_val, y_val_pred)
            })
        else:
            metrics['auc'] = metrics['train_auc']
        
        return metrics
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict:
        """Train LightGBM model"""
        params = self.config.get('model', {}).get('lightgbm', {})
        
        # Convert to Dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Validation data
        valid_sets = [train_data]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
        
        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 100),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 10)),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        metrics = {
            'train_auc': roc_auc_score(y_train, y_train_pred)
        }
        
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            metrics.update({
                'val_auc': roc_auc_score(y_val, y_val_pred),
                'auc': roc_auc_score(y_val, y_val_pred)
            })
        else:
            metrics['auc'] = metrics['train_auc']
        
        return metrics
    
    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Perform cross-validation
        
        Returns:
            CV metrics
        """
        cv = StratifiedKFold(
            n_splits=self.config.get('model', {}).get('cv_folds', 5),
            shuffle=True,
            random_state=42
        )
        
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]
            
            # Train model
            if self.model_type == 'xgboost':
                dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
                dval = xgb.DMatrix(X_val_cv, label=y_val_cv)
                params = self.config.get('model', {}).get('xgboost', {})
                
                model_cv = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=params.get('n_estimators', 100),
                    verbose_eval=False
                )
                
                y_pred = model_cv.predict(dval)
                
            else:  # lightgbm
                train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
                params = self.config.get('model', {}).get('lightgbm', {})
                
                model_cv = lgb.train(
                    params,
                    train_data,
                    num_boost_round=params.get('n_estimators', 100),
                    callbacks=[lgb.log_evaluation(period=0)]
                )
                
                y_pred = model_cv.predict(X_val_cv)
            
            auc = roc_auc_score(y_val_cv, y_pred)
            auc_scores.append(auc)
        
        self.cv_scores = auc_scores
        
        return {
            'cv_auc_mean': np.mean(auc_scores),
            'cv_auc_std': np.std(auc_scores),
            'cv_auc_min': np.min(auc_scores),
            'cv_auc_max': np.max(auc_scores)
        }
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions (probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            return self.model.predict(dmatrix)
        else:  # lightgbm
            return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'xgboost':
            self.model.save_model(str(filepath))
        else:  # lightgbm
            self.model.save_model(str(filepath))
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'n_features': self.model.num_features() if self.model_type == 'xgboost' else self.model.num_feature(),
            'cv_scores': self.cv_scores
        }
        
        with open(filepath.parent / f"{filepath.stem}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        filepath = Path(filepath)
        
        # Load metadata
        with open(filepath.parent / f"{filepath.stem}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.cv_scores = metadata.get('cv_scores')
        
        # Load model
        if self.model_type == 'xgboost':
            self.model = xgb.Booster()
            self.model.load_model(str(filepath))
        else:  # lightgbm
            self.model = lgb.Booster(model_file=str(filepath))
        
        logger.info(f"Model loaded from {filepath}")
    
    def _get_model_params(self) -> Dict:
        """Get model parameters"""
        if self.model_type == 'xgboost':
            return self.config.get('model', {}).get('xgboost', {})
        else:
            return self.config.get('model', {}).get('lightgbm', {})


if __name__ == "__main__":
    from src.data import generate_credit_data
    from src.features import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Generate data
    df = generate_credit_data(n_samples=5000)
    
    # Engineer features
    engineer = FeatureEngineer({})
    df_engineered = engineer.engineer_features(df, fit=True)
    
    # Prepare features
    feature_cols = [col for col in df_engineered.columns 
                    if col not in ['loan_status', 'customer_id']]
    X = df_engineered[feature_cols].fillna(0)
    y = df_engineered['loan_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train XGBoost
    config = {
        'model': {
            'type': 'xgboost',
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'early_stopping_rounds': 10
            },
            'cv_folds': 5
        }
    }
    
    trainer = ModelTrainer(config)
    metrics = trainer.train(X_train, y_train, X_test, y_test)
    
    print("\nTraining Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save model
    trainer.save_model("models/xgboost_model.json")
