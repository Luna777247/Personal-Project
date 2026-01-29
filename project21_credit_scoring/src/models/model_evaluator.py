"""
Model evaluation for credit scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate credit scoring model performance"""
    
    def __init__(self, config: Dict):
        """
        Initialize model evaluator
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.threshold = config.get('threshold', 0.5)
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = None
    ) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Evaluation metrics dictionary
        """
        if threshold is None:
            threshold = self.threshold
        
        # Binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Classification metrics
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'threshold': threshold
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        # Business metrics
        metrics.update(self._compute_business_metrics(y_true, y_pred, y_pred_proba))
        
        self.results = metrics
        
        logger.info(f"Evaluation completed. AUC: {metrics['auc']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def _compute_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Compute business-specific metrics
        
        Returns:
            Business metrics dictionary
        """
        # Approval rate
        approval_rate = np.mean(y_pred == 0)  # 0 = approved
        
        # Default rate in approved loans
        approved_mask = (y_pred == 0)
        if approved_mask.sum() > 0:
            default_rate_approved = np.mean(y_true[approved_mask])
        else:
            default_rate_approved = 0
        
        # False positive rate (approve bad loans)
        fpr = np.sum((y_pred == 0) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        
        # False negative rate (reject good loans)
        fnr = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
        
        # Expected loss (assuming $100 per false positive, $20 per false negative)
        fp_cost = 100
        fn_cost = 20
        expected_loss = (np.sum((y_pred == 0) & (y_true == 1)) * fp_cost + 
                         np.sum((y_pred == 1) & (y_true == 0)) * fn_cost)
        
        return {
            'approval_rate': float(approval_rate),
            'default_rate_approved': float(default_rate_approved),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'expected_loss': float(expected_loss)
        }
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Optimization metric ('f1', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, optimal_score)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.2f} ({metric}: {optimal_score:.4f})")
        
        return optimal_threshold, optimal_score
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = None
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Credit Scoring Model')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = None
    ):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Approved (0)', 'Denied (1)'],
            yticklabels=['Good (0)', 'Bad (1)']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = None
    ):
        """Plot metrics vs threshold"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        
        precisions = []
        recalls = []
        f1_scores = []
        approval_rates = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))
            approval_rates.append(np.mean(y_pred == 0))
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        plt.plot(thresholds, approval_rates, label='Approval Rate', linewidth=2, linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Classification Threshold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis saved to {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        output_dir: str = 'results'
    ):
        """Generate comprehensive evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([self.results])
        metrics_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
        
        logger.info(f"Evaluation report saved to {output_dir}")
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """Print detailed classification report"""
        print("\nClassification Report:")
        print("=" * 60)
        print(classification_report(
            y_true,
            y_pred,
            target_names=['Good Credit (0)', 'Bad Credit (1)']
        ))


if __name__ == "__main__":
    # Example usage
    from src.models.model_trainer import ModelTrainer
    from src.data import generate_credit_data
    from src.features import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Generate and prepare data
    df = generate_credit_data(n_samples=5000)
    engineer = FeatureEngineer({})
    df_engineered = engineer.engineer_features(df, fit=True)
    
    feature_cols = [col for col in df_engineered.columns 
                    if col not in ['loan_status', 'customer_id']]
    X = df_engineered[feature_cols].fillna(0)
    y = df_engineered['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train model
    config = {
        'model': {
            'type': 'xgboost',
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            }
        }
    }
    
    trainer = ModelTrainer(config)
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    y_pred_proba = trainer.predict(X_test)
    
    evaluator = ModelEvaluator({'threshold': 0.5})
    metrics = evaluator.evaluate(y_test.values, y_pred_proba)
    
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(
        y_test.values,
        y_pred_proba,
        metric='f1'
    )
    
    # Generate plots
    evaluator.plot_roc_curve(y_test.values, y_pred_proba, 'results/roc_curve.png')
    evaluator.plot_confusion_matrix(y_test.values, (y_pred_proba >= 0.5).astype(int), 
                                     'results/confusion_matrix.png')
