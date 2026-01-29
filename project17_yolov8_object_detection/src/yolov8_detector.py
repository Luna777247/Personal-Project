"""
YOLOv8 Object Detection Training and Inference Pipeline
======================================================

This module provides a complete pipeline for training YOLOv8 models on custom datasets,
performing inference, and evaluating model performance.

Features:
- Custom dataset training with data augmentation
- Model evaluation with mAP metrics
- Real-time inference with video streams
- ONNX export for optimized deployment
- Comprehensive logging and visualization

Author: AI Assistant
Date: 2025
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import yaml
from tqdm import tqdm
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv8Detector:
    """
    YOLOv8 Object Detection Pipeline

    Handles training, inference, and evaluation of YOLOv8 models.
    """

    def __init__(self, model_size='yolov8n', device='auto'):
        """
        Initialize YOLOv8 detector

        Args:
            model_size (str): Model size ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            device (str): Device to run on ('cpu', 'cuda', 'auto')
        """
        self.model_size = model_size
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.results = None
        self.class_names = []  # Initialize class names list

        logger.info(f"Initialized YOLOv8 detector with {model_size} on {self.device}")

    def load_model(self, weights_path=None):
        """
        Load YOLOv8 model

        Args:
            weights_path (str): Path to custom weights file (optional)
        """
        try:
            if weights_path and os.path.exists(weights_path):
                self.model = YOLO(weights_path)
                logger.info(f"Loaded custom model from {weights_path}")
            else:
                self.model = YOLO(f'{self.model_size}.pt')
                logger.info(f"Loaded pretrained {self.model_size} model")

            # Set class names from the model
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = []

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def prepare_dataset(self, data_yaml_path):
        """
        Prepare dataset configuration

        Args:
            data_yaml_path (str): Path to data.yaml file
        """
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Data configuration file not found: {data_yaml_path}")

        with open(data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)

        logger.info(f"Dataset prepared with {len(self.data_config['names'])} classes")

    def train(self, data_yaml_path, epochs=100, batch_size=16, img_size=640, **kwargs):
        """
        Train YOLOv8 model

        Args:
            data_yaml_path (str): Path to data.yaml
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            img_size (int): Image size
            **kwargs: Additional training arguments
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")

        # Default training arguments
        train_args = {
            'data': data_yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': self.device,
            'workers': 4,
            'patience': 50,
            'save': True,
            'project': 'runs/train',
            'name': f'{self.model_size}_custom',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }

        # Update with custom arguments
        train_args.update(kwargs)

        self.results = self.model.train(**train_args)
        logger.info("Training completed successfully")

        return self.results

    def validate(self, data_yaml_path=None, conf=0.25, iou=0.6):
        """
        Validate model on validation set

        Args:
            data_yaml_path (str): Path to data.yaml (optional if already prepared)
            conf (float): Confidence threshold
            iou (float): IoU threshold
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if data_yaml_path:
            self.prepare_dataset(data_yaml_path)

        logger.info("Starting validation...")

        metrics = self.model.val(
            data=data_yaml_path,
            conf=conf,
            iou=iou,
            device=self.device
        )

        logger.info(f"Validation mAP50: {metrics.box.map50:.4f}")
        logger.info(f"Validation mAP50-95: {metrics.box.map:.4f}")

        return metrics

    def predict(self, source, conf=0.25, iou=0.6, save=True, **kwargs):
        """
        Run inference on images/videos

        Args:
            source (str): Path to image/video or URL
            conf (float): Confidence threshold
            iou (float): IoU threshold
            save (bool): Save results
            **kwargs: Additional prediction arguments
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Running inference on {source}")

        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            device=self.device,
            **kwargs
        )

        return results

    def export_model(self, format='onnx', **kwargs):
        """
        Export model to different formats

        Args:
            format (str): Export format ('onnx', 'torchscript', 'openvino', etc.)
            **kwargs: Additional export arguments
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info(f"Exporting model to {format} format")

        export_args = {
            'format': format,
            'device': self.device,
        }
        export_args.update(kwargs)

        exported_path = self.model.export(**export_args)
        logger.info(f"Model exported to {exported_path}")

        return exported_path

    def visualize_results(self, results, save_path=None):
        """
        Visualize detection results

        Args:
            results: YOLO results object
            save_path (str): Path to save visualization
        """
        if not results:
            logger.warning("No results to visualize")
            return

        # Plot results
        for result in results:
            img = result.plot()
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()

    def benchmark_performance(self, test_images_path, num_runs=100):
        """
        Benchmark model performance

        Args:
            test_images_path (str): Path to test images
            num_runs (int): Number of inference runs for timing
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Starting performance benchmark...")

        # Get test images
        test_images = list(Path(test_images_path).glob('*.jpg')) + \
                     list(Path(test_images_path).glob('*.png'))

        if not test_images:
            raise ValueError(f"No test images found in {test_images_path}")

        # Warm up
        logger.info("Warming up model...")
        for img_path in test_images[:5]:
            self.predict(str(img_path), save=False, verbose=False)

        # Benchmark
        logger.info(f"Running benchmark with {num_runs} inferences...")
        import time

        start_time = time.time()
        for _ in tqdm(range(num_runs)):
            img_path = np.random.choice(test_images)
            self.predict(str(img_path), save=False, verbose=False)

        total_time = time.time() - start_time
        avg_time = total_time / num_runs

        logger.info(".4f")
        logger.info(".2f")

        return {
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'fps': 1.0 / avg_time
        }


def create_data_yaml(train_path, val_path, test_path, classes, save_path='data/data.yaml'):
    """
    Create data.yaml configuration file

    Args:
        train_path (str): Path to training images
        val_path (str): Path to validation images
        test_path (str): Path to test images
        classes (list): List of class names
        save_path (str): Path to save data.yaml
    """
    data_config = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(classes),
        'names': classes
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    logger.info(f"Data configuration saved to {save_path}")
    return save_path


if __name__ == "__main__":
    # Example usage
    detector = YOLOv8Detector(model_size='yolov8n')

    # Create sample data configuration
    classes = ['person', 'car', 'truck', 'bus', 'motorcycle']
    create_data_yaml(
        train_path='data/train/images',
        val_path='data/val/images',
        test_path='data/test/images',
        classes=classes
    )

    # Train model (uncomment to run)
    # detector.train('data/data.yaml', epochs=50, batch_size=8)

    # Load trained model and run inference
    # detector.load_model('runs/train/yolov8n_custom/weights/best.pt')
    # results = detector.predict('data/test/images/sample.jpg')
    # detector.visualize_results(results)

    logger.info("YOLOv8 pipeline initialized. Ready for training or inference.")