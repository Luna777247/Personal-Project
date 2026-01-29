"""
YOLOv8 Model Evaluation and Benchmarking
=========================================

Comprehensive evaluation suite for YOLOv8 object detection models.

Features:
- Standard object detection metrics (mAP, precision, recall)
- Model comparison across different configurations
- Performance benchmarking (latency, throughput)
- Confusion matrix and detailed analysis
- Export evaluation results and visualizations

Author: AI Assistant
Date: 2025
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOEvaluator:
    """
    YOLOv8 Model Evaluator

    Comprehensive evaluation suite for object detection models.
    """

    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize evaluator

        Args:
            model_path (str): Path to model weights
            device (str): Device to run evaluation on
        """
        self.model_path = model_path
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []

        if model_path:
            self.load_model(model_path)

        logger.info(f"Initialized YOLO evaluator on {self.device}")

    def load_model(self, model_path: str):
        """
        Load YOLOv8 model

        Args:
            model_path (str): Path to model weights
        """
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.class_names = list(self.model.names.values()) if hasattr(self.model, 'names') else []
            self.model_path = model_path
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def evaluate_on_dataset(self, data_yaml_path: str, conf: float = 0.25, iou: float = 0.6,
                          save_dir: str = 'runs/evaluate'):
        """
        Evaluate model on dataset using YOLO's built-in evaluation

        Args:
            data_yaml_path (str): Path to data.yaml
            conf (float): Confidence threshold
            iou (float): IoU threshold
            save_dir (str): Directory to save results

        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Evaluating model on dataset: {data_yaml_path}")

        # Run evaluation
        metrics = self.model.val(
            data=data_yaml_path,
            conf=conf,
            iou=iou,
            save_dir=save_dir,
            device=self.device,
            plots=True,
            save_json=True
        )

        # Extract key metrics
        results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'fitness': float(metrics.fitness) if hasattr(metrics, 'fitness') else 0.0,
            'class_metrics': {}
        }

        # Per-class metrics
        if hasattr(metrics.box, 'ap_class_index'):
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
                results['class_metrics'][class_name] = {
                    'precision': float(metrics.box.p[i]),
                    'recall': float(metrics.box.r[i]),
                    'mAP50': float(metrics.box.ap50[i]),
                    'mAP50-95': float(metrics.box.ap[i])
                }

        logger.info(".4f")
        logger.info(".4f")

        return results

    def benchmark_inference_speed(self, test_images_dir: str, num_runs: int = 100,
                                batch_sizes: List[int] = None):
        """
        Benchmark inference speed and throughput

        Args:
            test_images_dir (str): Directory containing test images
            num_runs (int): Number of inference runs
            batch_sizes (List[int]): Batch sizes to test

        Returns:
            dict: Benchmarking results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if batch_sizes is None:
            batch_sizes = [1, 4, 8]

        logger.info(f"Benchmarking inference speed with {num_runs} runs")

        # Get test images
        test_images = list(Path(test_images_dir).glob('*.jpg')) + \
                     list(Path(test_images_dir).glob('*.png'))

        if not test_images:
            raise ValueError(f"No test images found in {test_images_dir}")

        # Limit to first 50 images for benchmarking
        test_images = test_images[:min(50, len(test_images))]

        benchmark_results = {}

        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size {batch_size}")

            # Prepare batches
            batches = []
            for i in range(0, len(test_images), batch_size):
                batch = test_images[i:i + batch_size]
                batch_images = []
                for img_path in batch:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        batch_images.append(img)
                if batch_images:
                    batches.append(batch_images)

            # Warm up
            logger.info("Warming up...")
            for _ in range(5):
                self.model(batches[0][0], verbose=False)

            # Benchmark
            start_time = time.time()
            total_inferences = 0

            for _ in tqdm(range(num_runs), desc=f"Batch size {batch_size}"):
                for batch in batches:
                    for img in batch:
                        self.model(img, verbose=False)
                        total_inferences += 1

            total_time = time.time() - start_time

            benchmark_results[f'batch_{batch_size}'] = {
                'total_time': total_time,
                'total_inferences': total_inferences,
                'avg_time_per_inference': total_time / total_inferences,
                'fps': total_inferences / total_time,
                'batch_size': batch_size
            }

            logger.info(f"Batch {batch_size}: {benchmark_results[f'batch_{batch_size}']['fps']:.2f} FPS")

        return benchmark_results

    def analyze_predictions(self, test_images_dir: str, test_labels_dir: str = None,
                          conf: float = 0.25, iou: float = 0.6, save_analysis: bool = True):
        """
        Analyze model predictions in detail

        Args:
            test_images_dir (str): Directory containing test images
            test_labels_dir (str): Directory containing ground truth labels (optional)
            conf (float): Confidence threshold
            iou (float): IoU threshold
            save_analysis (bool): Whether to save analysis results

        Returns:
            dict: Detailed analysis results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Analyzing model predictions...")

        test_images = list(Path(test_images_dir).glob('*.jpg')) + \
                     list(Path(test_images_dir).glob('*.png'))

        predictions = []
        ground_truths = []

        for img_path in tqdm(test_images, desc="Analyzing predictions"):
            # Run inference
            results = self.model(str(img_path), conf=conf, iou=iou, verbose=False)

            # Extract predictions
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        pred = {
                            'image_path': str(img_path),
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'class_id': int(box.cls[0].cpu().numpy()),
                            'confidence': float(box.conf[0].cpu().numpy()),
                            'class_name': self.class_names[int(box.cls[0].cpu().numpy())] if self.class_names else f"class_{int(box.cls[0].cpu().numpy())}"
                        }
                        predictions.append(pred)

            # Load ground truth if available
            if test_labels_dir:
                label_path = Path(test_labels_dir) / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                gt = {
                                    'image_path': str(img_path),
                                    'bbox': None,  # Would need conversion from YOLO format
                                    'class_id': int(parts[0]),
                                    'class_name': self.class_names[int(parts[0])] if self.class_names else f"class_{int(parts[0])}"
                                }
                                ground_truths.append(gt)

        analysis = {
            'total_predictions': len(predictions),
            'predictions_per_class': {},
            'confidence_distribution': [],
            'bbox_sizes': []
        }

        # Analyze predictions
        for pred in predictions:
            class_name = pred['class_name']
            analysis['predictions_per_class'][class_name] = analysis['predictions_per_class'].get(class_name, 0) + 1
            analysis['confidence_distribution'].append(pred['confidence'])

            # Calculate bbox size
            bbox = pred['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            analysis['bbox_sizes'].append({'width': width, 'height': height, 'area': width * height})

        # Convert to numpy arrays for statistics
        analysis['confidence_stats'] = {
            'mean': float(np.mean(analysis['confidence_distribution'])),
            'std': float(np.std(analysis['confidence_distribution'])),
            'min': float(np.min(analysis['confidence_distribution'])),
            'max': float(np.max(analysis['confidence_distribution']))
        }

        bbox_sizes = pd.DataFrame(analysis['bbox_sizes'])
        analysis['bbox_stats'] = {
            'mean_width': float(bbox_sizes['width'].mean()),
            'mean_height': float(bbox_sizes['height'].mean()),
            'mean_area': float(bbox_sizes['area'].mean()),
            'std_width': float(bbox_sizes['width'].std()),
            'std_height': float(bbox_sizes['height'].std()),
            'std_area': float(bbox_sizes['area'].std())
        }

        if save_analysis:
            self.save_analysis_results(analysis, 'analysis_results.json')

        logger.info(f"Analysis complete. {len(predictions)} predictions analyzed.")
        return analysis

    def compare_models(self, model_paths: List[str], data_yaml_path: str,
                      conf: float = 0.25, iou: float = 0.6):
        """
        Compare multiple models on the same dataset

        Args:
            model_paths (List[str]): List of model paths to compare
            data_yaml_path (str): Path to data.yaml
            conf (float): Confidence threshold
            iou (float): IoU threshold

        Returns:
            dict: Comparison results
        """
        logger.info(f"Comparing {len(model_paths)} models")

        comparison_results = {}

        for model_path in model_paths:
            logger.info(f"Evaluating {Path(model_path).name}")
            self.load_model(model_path)

            results = self.evaluate_on_dataset(data_yaml_path, conf, iou, save_dir=f'runs/compare_{Path(model_path).stem}')
            comparison_results[Path(model_path).name] = results

        # Create comparison summary
        summary = {
            'models': list(comparison_results.keys()),
            'mAP50_scores': [results['mAP50'] for results in comparison_results.values()],
            'mAP50_95_scores': [results['mAP50-95'] for results in comparison_results.values()],
            'precision_scores': [results['precision'] for results in comparison_results.values()],
            'recall_scores': [results['recall'] for results in comparison_results.values()]
        }

        # Find best model
        best_model_idx = np.argmax(summary['mAP50_95_scores'])
        summary['best_model'] = summary['models'][best_model_idx]
        summary['best_mAP50_95'] = summary['mAP50_95_scores'][best_model_idx]

        logger.info(f"Best model: {summary['best_model']} (mAP50-95: {summary['best_mAP50_95']:.4f})")

        return {
            'detailed_results': comparison_results,
            'summary': summary
        }

    def create_visualizations(self, analysis_results: dict, save_dir: str = 'visualizations'):
        """
        Create visualizations for evaluation results

        Args:
            analysis_results (dict): Analysis results from analyze_predictions
            save_dir (str): Directory to save visualizations
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(analysis_results['confidence_distribution'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Predictions per class
        plt.figure(figsize=(12, 6))
        classes = list(analysis_results['predictions_per_class'].keys())
        counts = list(analysis_results['predictions_per_class'].values())
        plt.bar(classes, counts, alpha=0.7, edgecolor='black')
        plt.xlabel('Class')
        plt.ylabel('Number of Predictions')
        plt.title('Predictions per Class')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / 'predictions_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Bounding box sizes
        bbox_data = pd.DataFrame(analysis_results['bbox_sizes'])
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 3, 1)
        plt.hist(bbox_data['width'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Width')
        plt.ylabel('Frequency')
        plt.title('Bounding Box Width Distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.hist(bbox_data['height'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Height')
        plt.ylabel('Frequency')
        plt.title('Bounding Box Height Distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.hist(bbox_data['area'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Area')
        plt.ylabel('Frequency')
        plt.title('Bounding Box Area Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / 'bbox_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to {save_path}")

    def save_analysis_results(self, results: dict, filename: str):
        """
        Save analysis results to JSON file

        Args:
            results (dict): Results to save
            filename (str): Output filename
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Analysis results saved to {filename}")

    def export_model_metrics(self, evaluation_results: dict, output_file: str = 'model_metrics.json'):
        """
        Export model evaluation metrics to file

        Args:
            evaluation_results (dict): Evaluation results
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        logger.info(f"Model metrics exported to {output_file}")


def run_comprehensive_evaluation(model_path: str, data_yaml_path: str, test_images_dir: str,
                               output_dir: str = 'evaluation_results'):
    """
    Run comprehensive model evaluation

    Args:
        model_path (str): Path to model weights
        data_yaml_path (str): Path to data.yaml
        test_images_dir (str): Directory containing test images
        output_dir (str): Output directory for results
    """
    logger.info("Starting comprehensive model evaluation")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize evaluator
    evaluator = YOLOEvaluator(model_path)

    # 1. Standard evaluation
    logger.info("Running standard evaluation...")
    eval_results = evaluator.evaluate_on_dataset(data_yaml_path, save_dir=str(output_path / 'standard_eval'))

    # 2. Performance benchmarking
    logger.info("Running performance benchmarking...")
    benchmark_results = evaluator.benchmark_inference_speed(test_images_dir)

    # 3. Detailed prediction analysis
    logger.info("Running prediction analysis...")
    analysis_results = evaluator.analyze_predictions(test_images_dir)

    # 4. Create visualizations
    logger.info("Creating visualizations...")
    evaluator.create_visualizations(analysis_results, str(output_path / 'visualizations'))

    # 5. Compile comprehensive report
    report = {
        'model_path': model_path,
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'standard_metrics': eval_results,
        'benchmarking': benchmark_results,
        'prediction_analysis': analysis_results,
        'summary': {
            'mAP50': eval_results['mAP50'],
            'mAP50_95': eval_results['mAP50-95'],
            'best_fps': max([b['fps'] for b in benchmark_results.values()]),
            'total_predictions': analysis_results['total_predictions'],
            'avg_confidence': analysis_results['confidence_stats']['mean']
        }
    }

    # Save comprehensive report
    evaluator.save_analysis_results(report, str(output_path / 'comprehensive_report.json'))

    logger.info("Comprehensive evaluation completed")
    logger.info(f"Results saved to {output_path}")

    return report


if __name__ == "__main__":
    # Example usage
    model_path = 'runs/train/yolov8n_custom/weights/best.pt'
    data_yaml = 'data/data.yaml'
    test_images_dir = 'data/test/images'

    if os.path.exists(model_path):
        # Run comprehensive evaluation
        report = run_comprehensive_evaluation(model_path, data_yaml, test_images_dir)
        print("Evaluation Summary:")
        print(f"mAP50: {report['summary']['mAP50']:.4f}")
        print(f"mAP50-95: {report['summary']['mAP50-95']:.4f}")
        print(f"Best FPS: {report['summary']['best_fps']:.2f}")
    else:
        print(f"Model not found at {model_path}. Please train a model first.")

    logger.info("Evaluation pipeline ready")