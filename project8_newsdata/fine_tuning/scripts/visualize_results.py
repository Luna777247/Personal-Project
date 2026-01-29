#!/usr/bin/env python3
"""
Visualization Script for Fine-tuning Results
Create comprehensive visualizations for model training and evaluation
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_training_logs(log_dir: str) -> Dict[str, pd.DataFrame]:
    """Load training logs from different models"""
    logs = {}

    for model_dir in Path(log_dir).iterdir():
        if model_dir.is_dir():
            log_file = model_dir / "training_logs.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)
                logs[model_dir.name] = df

    return logs


def plot_training_curves(logs: Dict[str, pd.DataFrame], output_dir: str):
    """Plot training curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16)

    # Loss curves
    ax = axes[0, 0]
    for model_name, df in logs.items():
        if 'train_loss' in df.columns and 'eval_loss' in df.columns:
            ax.plot(df['epoch'], df['train_loss'], label=f'{model_name} - Train', linestyle='--')
            ax.plot(df['epoch'], df['eval_loss'], label=f'{model_name} - Eval', linestyle='-')
    ax.set_title('Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score curves
    ax = axes[0, 1]
    for model_name, df in logs.items():
        if 'eval_f1' in df.columns:
            ax.plot(df['epoch'], df['eval_f1'], label=model_name)
    ax.set_title('F1 Score Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[1, 0]
    for model_name, df in logs.items():
        if 'eval_accuracy' in df.columns:
            ax.plot(df['epoch'], df['eval_accuracy'], label=model_name)
    ax.set_title('Accuracy Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    for model_name, df in logs.items():
        if 'learning_rate' in df.columns:
            ax.plot(df['epoch'], df['learning_rate'], label=model_name)
    ax.set_title('Learning Rate Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(evaluation_results: Dict[str, Dict[str, Any]], output_dir: str):
    """Plot model comparison metrics"""
    models = list(evaluation_results.keys())
    metrics = ['accuracy', 'macro_f1', 'weighted_f1']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [evaluation_results[model].get(metric, 0) for model in models]

        bars = ax.bar(models, values, alpha=0.8)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(evaluation_results: Dict[str, Dict[str, Any]], output_dir: str):
    """Plot confusion matrices for all models"""
    n_models = len(evaluation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for i, (model_name, results) in enumerate(evaluation_results.items()):
        ax = axes[i]
        cm = np.array(results.get('confusion_matrix', []))

        if cm.size > 0:
            # Get labels from classification report
            report = results.get('classification_report', {})
            labels = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f'{model_name.upper()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_performance(evaluation_results: Dict[str, Dict[str, Any]], output_dir: str):
    """Plot per-class performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Per-Class Performance Analysis', fontsize=16)

    metrics = ['precision', 'recall', 'f1-score', 'support']
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        all_classes = set()

        # Collect all classes across models
        for results in evaluation_results.values():
            report = results.get('classification_report', {})
            all_classes.update([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])

        all_classes = sorted(list(all_classes))

        # Plot bars for each class
        x = np.arange(len(all_classes))
        width = 0.8 / len(evaluation_results)

        for j, (model_name, results) in enumerate(evaluation_results.items()):
            report = results.get('classification_report', {})
            values = [report.get(cls, {}).get(metric, 0) for cls in all_classes]

            ax.bar(x + j*width, values, width, label=model_name, alpha=0.8)

        ax.set_title(f'{metric.title()} by Class')
        ax.set_xlabel('Classes')
        ax.set_ylabel(metric.title())
        ax.set_xticks(x + width * (len(evaluation_results)-1) / 2)
        ax.set_xticklabels(all_classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_summary(evaluation_results: Dict[str, Dict[str, Any]], output_dir: str):
    """Create performance summary table"""
    summary_data = []

    for model_name, results in evaluation_results.items():
        row = {
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Macro F1': results.get('macro_f1', 0),
            'Weighted F1': results.get('weighted_f1', 0)
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df = df.round(4)

    # Save as CSV
    df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)

    # Create markdown table
    markdown_table = df.to_markdown(index=False)

    with open(os.path.join(output_dir, 'performance_summary.md'), 'w') as f:
        f.write("# Model Performance Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(markdown_table)
        f.write("\n\n## Best Performing Models\n\n")

        # Add best models
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        best_macro_f1 = df.loc[df['Macro F1'].idxmax()]
        best_weighted_f1 = df.loc[df['Weighted F1'].idxmax()]

        f.write(f"- **Highest Accuracy**: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})\n")
        f.write(f"- **Highest Macro F1**: {best_macro_f1['Model']} ({best_macro_f1['Macro F1']:.4f})\n")
        f.write(f"- **Highest Weighted F1**: {best_weighted_f1['Model']} ({best_weighted_f1['Weighted F1']:.4f})\n")


def plot_training_time_analysis(logs: Dict[str, pd.DataFrame], output_dir: str):
    """Analyze training time and efficiency"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training Time Analysis', fontsize=16)

    # Training time per epoch
    ax = axes[0]
    for model_name, df in logs.items():
        if 'epoch_time' in df.columns:
            ax.plot(df['epoch'], df['epoch_time'], label=model_name, marker='o')
    ax.set_title('Training Time per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative training time
    ax = axes[1]
    for model_name, df in logs.items():
        if 'epoch_time' in df.columns:
            cumulative_time = df['epoch_time'].cumsum()
            ax.plot(df['epoch'], cumulative_time, label=model_name, marker='o')
    ax.set_title('Cumulative Training Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Time (seconds)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_report(logs: Dict[str, pd.DataFrame],
                               evaluation_results: Dict[str, Dict[str, Any]],
                               output_dir: str):
    """Create comprehensive training and evaluation report"""
    report = {
        "training_summary": {},
        "evaluation_summary": {},
        "recommendations": []
    }

    # Training summary
    for model_name, df in logs.items():
        if not df.empty:
            final_epoch = df.iloc[-1]
            report["training_summary"][model_name] = {
                "final_train_loss": final_epoch.get("train_loss", 0),
                "final_eval_loss": final_epoch.get("eval_loss", 0),
                "final_f1": final_epoch.get("eval_f1", 0),
                "final_accuracy": final_epoch.get("eval_accuracy", 0),
                "total_epochs": len(df),
                "best_epoch": df["eval_f1"].idxmax() + 1 if "eval_f1" in df.columns else 0
            }

    # Evaluation summary
    report["evaluation_summary"] = evaluation_results

    # Generate recommendations
    best_model = max(evaluation_results.keys(),
                    key=lambda x: evaluation_results[x].get("weighted_f1", 0))

    report["recommendations"].append(f"Use {best_model} for production deployment")

    for model_name, metrics in evaluation_results.items():
        f1_score = metrics.get("weighted_f1", 0)
        if f1_score < 0.7:
            report["recommendations"].append(f"Consider retraining {model_name} with more data or hyperparameter tuning")
        elif f1_score > 0.9:
            report["recommendations"].append(f"{model_name} shows excellent performance and is ready for deployment")

    # Save report
    with open(os.path.join(output_dir, 'comprehensive_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Create HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Disaster Information Extraction - Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Disaster Information Extraction - Training Report</h1>
    <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="section">
        <h2>Model Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>Weighted F1</th>
            </tr>
"""

    for model_name, metrics in evaluation_results.items():
        html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics.get('accuracy', 0):.4f}</td>
                <td>{metrics.get('macro_f1', 0):.4f}</td>
                <td>{metrics.get('weighted_f1', 0):.4f}</td>
            </tr>
"""

    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>Training Curves</h2>
        <img src="training_curves.png" alt="Training Curves">
    </div>

    <div class="section">
        <h2>Model Comparison</h2>
        <img src="model_comparison.png" alt="Model Comparison">
    </div>

    <div class="section">
        <h2>Confusion Matrices</h2>
        <img src="confusion_matrices.png" alt="Confusion Matrices">
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
"""

    for rec in report["recommendations"]:
        html_content += f"<li>{rec}</li>"

    html_content += """
        </ul>
    </div>
</body>
</html>
"""

    with open(os.path.join(output_dir, 'training_report.html'), 'w') as f:
        f.write(html_content)


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description="Visualize Fine-tuning Results")
    parser.add_argument("--logs-dir", required=True, help="Directory containing training logs")
    parser.add_argument("--evaluation-dir", required=True, help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="visualizations/", help="Output directory for visualizations")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training logs
    logs = load_training_logs(args.logs_dir)
    logger.info(f"Loaded training logs for {len(logs)} models")

    # Load evaluation results
    evaluation_results = {}
    for eval_file in Path(args.evaluation_dir).glob("*_evaluation.json"):
        model_name = eval_file.stem.replace("_evaluation", "")
        with open(eval_file, 'r') as f:
            evaluation_results[model_name] = json.load(f)

    logger.info(f"Loaded evaluation results for {len(evaluation_results)} models")

    # Create visualizations
    if logs:
        plot_training_curves(logs, args.output_dir)
        plot_training_time_analysis(logs, args.output_dir)

    if evaluation_results:
        plot_model_comparison(evaluation_results, args.output_dir)
        plot_confusion_matrices(evaluation_results, args.output_dir)
        plot_class_performance(evaluation_results, args.output_dir)
        create_performance_summary(evaluation_results, args.output_dir)

    # Create comprehensive report
    create_comprehensive_report(logs, evaluation_results, args.output_dir)

    logger.info(f"Visualizations and reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()