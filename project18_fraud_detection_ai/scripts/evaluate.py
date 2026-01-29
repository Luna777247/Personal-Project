"""
Evaluation script for fraud detection models
Comprehensive model evaluation with multiple metrics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import joblib
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_data(model_path: str, data_path: str):
    """Load model and test data"""
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    return model, df


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive metrics"""
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'accuracy': (y_pred == y_true).mean()
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
    })
    
    # Precision at K
    k_values = [10, 50, 100, 500]
    for k in k_values:
        if len(y_true) >= k:
            top_k_indices = np.argsort(y_proba)[-k:]
            precision_at_k = y_true[top_k_indices].mean()
            metrics[f'precision_at_{k}'] = precision_at_k
    
    return metrics


def plot_roc_curve(y_true, y_proba, output_path='reports/roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")


def plot_precision_recall_curve(y_true, y_proba, output_path='reports/pr_curve.png'):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Precision-Recall curve saved to {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_path='reports/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_score_distribution(y_true, y_proba, output_path='reports/score_distribution.png'):
    """Plot fraud score distribution"""
    df_scores = pd.DataFrame({
        'fraud_probability': y_proba,
        'is_fraud': y_true
    })
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df_scores[df_scores['is_fraud'] == 0]['fraud_probability'], 
             bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(df_scores[df_scores['is_fraud'] == 1]['fraud_probability'], 
             bins=50, alpha=0.7, label='Fraud', color='red')
    plt.xlabel('Fraud Probability')
    plt.ylabel('Count')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([
        df_scores[df_scores['is_fraud'] == 0]['fraud_probability'],
        df_scores[df_scores['is_fraud'] == 1]['fraud_probability']
    ], labels=['Normal', 'Fraud'])
    plt.ylabel('Fraud Probability')
    plt.title('Score Distribution (Box Plot)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Score distribution saved to {output_path}")


def generate_report(metrics, output_path='reports/evaluation_report.txt'):
    """Generate evaluation report"""
    report = []
    report.append("=" * 80)
    report.append("FRAUD DETECTION MODEL EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    report.append("PERFORMANCE METRICS")
    report.append("-" * 80)
    report.append(f"ROC-AUC Score:        {metrics['roc_auc']:.4f}")
    report.append(f"F1 Score:             {metrics['f1_score']:.4f}")
    report.append(f"Precision:            {metrics['precision']:.4f}")
    report.append(f"Recall:               {metrics['recall']:.4f}")
    report.append(f"Accuracy:             {metrics['accuracy']:.4f}")
    report.append("")
    
    report.append("CONFUSION MATRIX")
    report.append("-" * 80)
    report.append(f"True Positives:       {metrics['true_positives']}")
    report.append(f"True Negatives:       {metrics['true_negatives']}")
    report.append(f"False Positives:      {metrics['false_positives']}")
    report.append(f"False Negatives:      {metrics['false_negatives']}")
    report.append(f"False Positive Rate:  {metrics['false_positive_rate']:.4f}")
    report.append(f"True Positive Rate:   {metrics['true_positive_rate']:.4f}")
    report.append("")
    
    report.append("PRECISION @ K")
    report.append("-" * 80)
    for k in [10, 50, 100, 500]:
        key = f'precision_at_{k}'
        if key in metrics:
            report.append(f"Precision @ {k:3d}:      {metrics[key]:.4f}")
    report.append("")
    
    report.append("=" * 80)
    
    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Report saved to {output_path}")
    
    # Print to console
    print('\n'.join(report))


def main(args):
    """Main evaluation pipeline"""
    logger.info("Starting Model Evaluation")
    
    # Load model and data
    model, df = load_model_and_data(args.model_path, args.data_path)
    
    # Feature engineering
    from src.feature_engineering import TransactionFeatureEngineer
    
    feature_engineer_path = args.feature_engineer or "models/feature_engineer.pkl"
    feature_engineer = joblib.load(feature_engineer_path)
    
    df_features = feature_engineer.create_all_features(df, fit=False)
    
    # Select features
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'customer_id', 'merchant_id']]
    
    X = df_features[numeric_cols].fillna(0)
    y = df_features['is_fraud']
    
    logger.info(f"Evaluation dataset shape: {X.shape}")
    logger.info(f"Fraud ratio: {y.mean():.2%}")
    
    # Predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(y, y_pred, y_proba)
    
    # Save metrics
    metrics_path = 'reports/metrics.json'
    os.makedirs('reports', exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate plots
    logger.info("Generating visualizations...")
    plot_roc_curve(y, y_proba)
    plot_precision_recall_curve(y, y_proba)
    plot_confusion_matrix(y, y_pred)
    plot_score_distribution(y, y_proba)
    
    # Generate report
    logger.info("Generating evaluation report...")
    generate_report(metrics)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--feature-engineer', type=str,
                       help='Path to feature engineer pickle file')
    
    args = parser.parse_args()
    main(args)
