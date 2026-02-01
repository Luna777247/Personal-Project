"""
Evaluation Metrics for Fraud Detection
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from typing import Dict


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_score: Prediction scores (for AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_score is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_score)
            metrics['auc_pr'] = average_precision_score(y_true, y_score)
        except ValueError:
            # Handle case with only one class
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print metrics"""
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    if 'auc_pr' in metrics:
        print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Print confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Normal  Fraud")
    print(f"  Normal    {cm[0,0]:8d}  {cm[0,1]:6d}")
    print(f"  Fraud     {cm[1,0]:8d}  {cm[1,1]:6d}")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """Print classification report"""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
