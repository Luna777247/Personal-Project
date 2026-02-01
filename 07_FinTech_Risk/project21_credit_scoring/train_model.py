"""
Complete training pipeline for credit scoring model
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from sklearn.model_selection import train_test_split

from src.data import generate_credit_data, DataCleaner, DataScaler
from src.features import FeatureEngineer, FeatureSelector
from src.models import ModelTrainer, ModelEvaluator
from src.explainability import SHAPAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete training pipeline"""
    
    # Load configuration
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    logger.info("=" * 60)
    logger.info("CREDIT SCORING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Generate/Load Data
    logger.info("\n[Step 1/9] Loading data...")
    data_path = Path("data/raw/credit_data.csv")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        logger.info(f"Loaded existing data: {df.shape}")
    else:
        df = generate_credit_data(n_samples=10000, default_rate=0.2)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.info(f"Generated new data: {df.shape}")
    
    # Step 2: Data Cleaning
    logger.info("\n[Step 2/9] Cleaning data...")
    cleaner = DataCleaner(config.get('data', {}))
    df_clean, cleaning_report = cleaner.clean_data(df)
    logger.info(f"Cleaned data: {df_clean.shape}")
    
    # Step 3: Feature Engineering
    logger.info("\n[Step 3/9] Engineering features...")
    engineer = FeatureEngineer(config.get('feature_engineering', {}))
    df_engineered = engineer.engineer_features(df_clean, fit=True)
    logger.info(f"Engineered features: {df_engineered.shape}")
    
    # Step 4: Feature Selection
    logger.info("\n[Step 4/9] Selecting features...")
    feature_cols = [col for col in df_engineered.columns 
                    if col not in ['loan_status', 'customer_id', 'age_group']]
    X = df_engineered[feature_cols].fillna(0)
    y = df_engineered['loan_status']
    
    selector = FeatureSelector(config.get('feature_engineering', {}))
    X_selected, selected_features = selector.select_features(
        X, y, method='all', n_features=30
    )
    logger.info(f"Selected {len(selected_features)} features")
    
    # Step 5: Data Scaling
    logger.info("\n[Step 5/9] Scaling features...")
    scaler = DataScaler(scaler_type='standard')
    X_scaled = scaler.fit_transform(X_selected, X_selected.columns.tolist())
    
    # Save scaler
    scaler_path = Path("models/scaler.pkl")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save_scaler(str(scaler_path))
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Step 6: Train/Test Split
    logger.info("\n[Step 6/9] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 7: Model Training
    logger.info("\n[Step 7/9] Training models...")
    
    # XGBoost
    logger.info("\nTraining XGBoost...")
    xgb_config = {
        'model': {
            'type': 'xgboost',
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 10
            },
            'cv_folds': 5
        },
        'monitoring': {
            'mlflow_tracking': True
        }
    }
    
    xgb_trainer = ModelTrainer(xgb_config)
    xgb_metrics = xgb_trainer.train(X_train, y_train, X_test, y_test)
    
    logger.info(f"XGBoost AUC: {xgb_metrics['auc']:.4f}")
    logger.info(f"XGBoost CV AUC: {xgb_metrics['cv_auc_mean']:.4f} ± {xgb_metrics['cv_auc_std']:.4f}")
    
    # Save XGBoost model
    xgb_model_path = Path("models/xgboost_model.json")
    xgb_trainer.save_model(str(xgb_model_path))
    logger.info(f"XGBoost model saved to {xgb_model_path}")
    
    # LightGBM
    logger.info("\nTraining LightGBM...")
    lgb_config = {
        'model': {
            'type': 'lightgbm',
            'lightgbm': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'objective': 'binary',
                'metric': 'auc',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'early_stopping_rounds': 10
            },
            'cv_folds': 5
        },
        'monitoring': {
            'mlflow_tracking': True
        }
    }
    
    lgb_trainer = ModelTrainer(lgb_config)
    lgb_metrics = lgb_trainer.train(X_train, y_train, X_test, y_test)
    
    logger.info(f"LightGBM AUC: {lgb_metrics['auc']:.4f}")
    logger.info(f"LightGBM CV AUC: {lgb_metrics['cv_auc_mean']:.4f} ± {lgb_metrics['cv_auc_std']:.4f}")
    
    # Save LightGBM model
    lgb_model_path = Path("models/lightgbm_model.txt")
    lgb_trainer.save_model(str(lgb_model_path))
    logger.info(f"LightGBM model saved to {lgb_model_path}")
    
    # Select best model
    best_trainer = xgb_trainer if xgb_metrics['auc'] >= lgb_metrics['auc'] else lgb_trainer
    best_model_type = 'xgboost' if xgb_metrics['auc'] >= lgb_metrics['auc'] else 'lightgbm'
    logger.info(f"\nBest model: {best_model_type.upper()}")
    
    # Step 8: Model Evaluation
    logger.info("\n[Step 8/9] Evaluating model...")
    evaluator = ModelEvaluator({'threshold': 0.5})
    
    y_pred_proba = best_trainer.predict(X_test)
    eval_metrics = evaluator.evaluate(y_test.values, y_pred_proba)
    
    logger.info("\nEvaluation Metrics:")
    logger.info(f"  AUC: {eval_metrics['auc']:.4f}")
    logger.info(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {eval_metrics['precision']:.4f}")
    logger.info(f"  Recall: {eval_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {eval_metrics['f1_score']:.4f}")
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
        y_test.values, y_pred_proba, metric='f1'
    )
    logger.info(f"\nOptimal Threshold: {optimal_threshold:.2f} (F1: {optimal_f1:.4f})")
    
    # Generate evaluation plots
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    evaluator.plot_roc_curve(y_test.values, y_pred_proba, 
                              str(results_dir / "roc_curve.png"))
    evaluator.plot_precision_recall_curve(y_test.values, y_pred_proba,
                                           str(results_dir / "pr_curve.png"))
    evaluator.plot_confusion_matrix(y_test.values, (y_pred_proba >= 0.5).astype(int),
                                     str(results_dir / "confusion_matrix.png"))
    evaluator.plot_threshold_analysis(y_test.values, y_pred_proba,
                                       str(results_dir / "threshold_analysis.png"))
    
    logger.info(f"Evaluation plots saved to {results_dir}")
    
    # Step 9: SHAP Analysis
    logger.info("\n[Step 9/9] Generating SHAP explanations...")
    analyzer = SHAPAnalyzer(best_trainer.model, model_type=best_model_type)
    analyzer.create_explainer(X_train, n_samples=100)
    
    # Generate SHAP report
    explanations_dir = Path("models/explanations")
    analyzer.generate_report(X_test, output_dir=str(explanations_dir))
    
    # Feature importance
    importance = analyzer.get_feature_importance(X_test, top_n=20)
    logger.info("\nTop 10 Features by SHAP:")
    for idx, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"\nModels saved to: models/")
    logger.info(f"Results saved to: {results_dir}/")
    logger.info(f"Explanations saved to: {explanations_dir}/")
    logger.info(f"\nBest Model: {best_model_type.upper()}")
    logger.info(f"Test AUC: {eval_metrics['auc']:.4f}")
    logger.info(f"Optimal Threshold: {optimal_threshold:.2f}")


if __name__ == "__main__":
    main()
