"""
YOLOv8 Data Preprocessing Pipeline
==================================

This module provides comprehensive data preprocessing for YOLOv8 object detection training,
including dataset conversion, augmentation, validation, and visualization.

Features:
- Convert COCO, Pascal VOC, and custom datasets to YOLO format
- Advanced data augmentation techniques
- Dataset validation and statistics
- Visualization tools for annotations
- Train/val/test split functionality

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
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
import logging
import yaml
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODataPreprocessor:
    """
    YOLOv8 Data Preprocessing Pipeline

    Handles dataset preparation, augmentation, and validation for YOLOv8 training.
    """

    def __init__(self, base_path='data'):
        """
        Initialize data preprocessor

        Args:
            base_path (str): Base path for data operations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.augmentation_pipeline = None

        logger.info(f"Initialized data preprocessor with base path: {self.base_path}")

    def setup_augmentation(self, augmentation_config=None):
        """
        Setup data augmentation pipeline

        Args:
            augmentation_config (dict): Custom augmentation configuration
        """
        if augmentation_config is None:
            augmentation_config = {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.1,
                'rotate_limit': 15,
                'scale_limit': 0.1,
                'brightness_contrast': 0.2,
                'hue_saturation': 0.1,
                'blur': 0.1,
                'noise': 0.1
            }

        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=augmentation_config['horizontal_flip']),
            A.VerticalFlip(p=augmentation_config['vertical_flip']),
            A.Rotate(limit=augmentation_config['rotate_limit'], p=0.5),
            A.RandomScale(scale_limit=augmentation_config['scale_limit'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=augmentation_config['brightness_contrast'],
                contrast_limit=augmentation_config['brightness_contrast'],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=augmentation_config['hue_saturation'] * 20,
                sat_shift_limit=augmentation_config['hue_saturation'] * 30,
                val_shift_limit=augmentation_config['hue_saturation'] * 20,
                p=0.5
            ),
            A.Blur(blur_limit=3, p=augmentation_config['blur']),
            A.GaussNoise(var_limit=(10, 50), p=augmentation_config['noise']),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        logger.info("Data augmentation pipeline configured")

    def convert_coco_to_yolo(self, coco_json_path, output_path, class_mapping=None):
        """
        Convert COCO format annotations to YOLO format

        Args:
            coco_json_path (str): Path to COCO annotations JSON
            output_path (str): Output directory for YOLO annotations
            class_mapping (dict): Optional class ID mapping
        """
        logger.info(f"Converting COCO to YOLO format: {coco_json_path}")

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Create category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        if class_mapping:
            categories = {k: class_mapping.get(v, v) for k, v in categories.items()}

        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        # Process each image
        for image in tqdm(coco_data['images'], desc="Converting annotations"):
            image_id = image['id']
            image_filename = image['file_name']
            image_width = image['width']
            image_height = image['height']

            # Find annotations for this image
            image_annotations = [ann for ann in coco_data['annotations']
                               if ann['image_id'] == image_id]

            if not image_annotations:
                continue

            # Create YOLO annotation file
            annotation_file = output_path / f"{Path(image_filename).stem}.txt"
            with open(annotation_file, 'w') as f:
                for ann in image_annotations:
                    category_id = ann['category_id']
                    bbox = ann['bbox']  # [x, y, width, height]

                    # Convert to YOLO format [class_id, x_center, y_center, width, height]
                    x_center = (bbox[0] + bbox[2] / 2) / image_width
                    y_center = (bbox[1] + bbox[3] / 2) / image_height
                    width = bbox[2] / image_width
                    height = bbox[3] / image_height

                    # Ensure values are within [0, 1]
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    width = np.clip(width, 0, 1)
                    height = np.clip(height, 0, 1)

                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        logger.info(f"Conversion completed. Annotations saved to {output_path}")

    def convert_voc_to_yolo(self, voc_dir, output_path, class_mapping=None):
        """
        Convert Pascal VOC format to YOLO format

        Args:
            voc_dir (str): Directory containing VOC annotations
            output_path (str): Output directory for YOLO annotations
            class_mapping (dict): Optional class name mapping
        """
        logger.info(f"Converting VOC to YOLO format: {voc_dir}")

        voc_path = Path(voc_dir)
        annotations_dir = voc_path / 'Annotations'
        images_dir = voc_path / 'JPEGImages'

        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        # Get all annotation files
        xml_files = list(annotations_dir.glob('*.xml'))

        for xml_file in tqdm(xml_files, desc="Converting VOC annotations"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get image dimensions
            size = root.find('size')
            image_width = int(size.find('width').text)
            image_height = int(size.find('height').text)

            # Create YOLO annotation file
            annotation_file = output_path / f"{xml_file.stem}.txt"

            with open(annotation_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    class_id = class_name if class_mapping is None else class_mapping.get(class_name, class_name)

                    # Get bounding box
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)

                    # Convert to YOLO format
                    x_center = ((xmin + xmax) / 2) / image_width
                    y_center = ((ymin + ymax) / 2) / image_height
                    width = (xmax - xmin) / image_width
                    height = (ymax - ymin) / image_height

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        logger.info(f"VOC conversion completed. Annotations saved to {output_path}")

    def split_dataset(self, images_dir, annotations_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Split dataset into train/val/test sets

        Args:
            images_dir (str): Directory containing images
            annotations_dir (str): Directory containing annotations
            train_ratio (float): Training set ratio
            val_ratio (float): Validation set ratio
            test_ratio (float): Test set ratio
        """
        logger.info("Splitting dataset into train/val/test sets")

        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir)

        # Get all image files
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + list(images_path.glob('*.jpeg'))

        if not image_files:
            raise ValueError(f"No image files found in {images_dir}")

        # Get corresponding annotation files
        annotation_files = []
        for img_file in image_files:
            ann_file = annotations_path / f"{img_file.stem}.txt"
            if ann_file.exists():
                annotation_files.append((img_file, ann_file))

        logger.info(f"Found {len(annotation_files)} image-annotation pairs")

        # Split the data
        train_val, test = train_test_split(annotation_files, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)

        # Create output directories
        splits = {
            'train': train,
            'val': val,
            'test': test
        }

        for split_name, split_data in splits.items():
            split_images_dir = self.base_path / split_name / 'images'
            split_labels_dir = self.base_path / split_name / 'labels'

            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            for img_file, ann_file in split_data:
                # Copy image
                shutil.copy2(img_file, split_images_dir / img_file.name)
                # Copy annotation
                shutil.copy2(ann_file, split_labels_dir / ann_file.name)

        logger.info(f"Dataset split completed:")
        logger.info(f"  Train: {len(train)} samples")
        logger.info(f"  Val: {len(val)} samples")
        logger.info(f"  Test: {len(test)} samples")

        return {
            'train': len(train),
            'val': len(val),
            'test': len(test)
        }

    def validate_dataset(self, images_dir, labels_dir):
        """
        Validate dataset integrity and generate statistics

        Args:
            images_dir (str): Directory containing images
            labels_dir (str): Directory containing labels
        """
        logger.info("Validating dataset...")

        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        # Get all image files
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + list(images_path.glob('*.jpeg'))
        label_files = list(labels_path.glob('*.txt'))

        logger.info(f"Found {len(image_files)} images and {len(label_files)} label files")

        # Check for missing annotations
        image_names = {f.stem for f in image_files}
        label_names = {f.stem for f in label_files}

        missing_labels = image_names - label_names
        missing_images = label_names - image_names

        if missing_labels:
            logger.warning(f"Images without labels: {len(missing_labels)}")
            logger.warning(f"Sample missing: {list(missing_labels)[:5]}")

        if missing_images:
            logger.warning(f"Labels without images: {len(missing_images)}")
            logger.warning(f"Sample missing: {list(missing_images)[:5]}")

        # Analyze annotations
        class_counts = {}
        bbox_stats = []

        for label_file in tqdm(label_files, desc="Analyzing annotations"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:]))

                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        bbox_stats.append(bbox)

        # Generate statistics
        stats = {
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'missing_labels': len(missing_labels),
            'missing_images': len(missing_images),
            'class_distribution': class_counts,
            'bbox_stats': {
                'count': len(bbox_stats),
                'avg_width': np.mean([b[2] for b in bbox_stats]) if bbox_stats else 0,
                'avg_height': np.mean([b[3] for b in bbox_stats]) if bbox_stats else 0,
                'min_width': np.min([b[2] for b in bbox_stats]) if bbox_stats else 0,
                'max_width': np.max([b[2] for b in bbox_stats]) if bbox_stats else 0,
                'min_height': np.min([b[3] for b in bbox_stats]) if bbox_stats else 0,
                'max_height': np.max([b[3] for b in bbox_stats]) if bbox_stats else 0,
            }
        }

        logger.info("Dataset validation completed")
        logger.info(f"Class distribution: {stats['class_distribution']}")

        return stats

    def visualize_annotations(self, images_dir, labels_dir, class_names, num_samples=5, save_path=None):
        """
        Visualize annotations on sample images

        Args:
            images_dir (str): Directory containing images
            labels_dir (str): Directory containing labels
            class_names (list): List of class names
            num_samples (int): Number of samples to visualize
            save_path (str): Path to save visualization
        """
        logger.info(f"Visualizing {num_samples} sample annotations")

        images_path = Path(images_dir)
        labels_path = Path(labels_dir)

        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))

        if len(image_files) < num_samples:
            num_samples = len(image_files)

        selected_images = np.random.choice(image_files, num_samples, replace=False)

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

        for i, img_file in enumerate(selected_images):
            # Read image
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # Read annotations
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])

                            # Convert back to pixel coordinates
                            x1 = int((x_center - bbox_w/2) * w)
                            y1 = int((y_center - bbox_h/2) * h)
                            x2 = int((x_center + bbox_w/2) * w)
                            y2 = int((y_center + bbox_h/2) * h)

                            # Draw bounding box
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                            # Add label
                            label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if num_samples == 1:
                axes.imshow(image)
            else:
                axes[i].imshow(image)
                axes[i].set_title(f"Sample {i+1}")
                axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

    def augment_dataset(self, input_images_dir, input_labels_dir, output_images_dir, output_labels_dir,
                       num_augmentations=2):
        """
        Augment dataset with transformations

        Args:
            input_images_dir (str): Input images directory
            input_labels_dir (str): Input labels directory
            output_images_dir (str): Output images directory
            output_labels_dir (str): Output labels directory
            num_augmentations (int): Number of augmentations per image
        """
        if self.augmentation_pipeline is None:
            self.setup_augmentation()

        logger.info(f"Augmenting dataset with {num_augmentations} augmentations per image")

        input_images_path = Path(input_images_dir)
        input_labels_path = Path(input_labels_dir)
        output_images_path = Path(output_images_dir)
        output_labels_path = Path(output_labels_dir)

        output_images_path.mkdir(parents=True, exist_ok=True)
        output_labels_path.mkdir(parents=True, exist_ok=True)

        image_files = list(input_images_path.glob('*.jpg')) + list(input_images_path.glob('*.png'))

        for img_file in tqdm(image_files, desc="Augmenting images"):
            # Read image
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read labels
            label_file = input_labels_path / f"{img_file.stem}.txt"
            bboxes = []
            class_labels = []

            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = list(map(float, parts[1:]))
                            bboxes.append(bbox)
                            class_labels.append(class_id)

            # Save original
            shutil.copy2(img_file, output_images_path / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, output_labels_path / label_file.name)

            # Generate augmentations
            for aug_idx in range(num_augmentations):
                augmented = self.augmentation_pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                # Save augmented image
                aug_filename = f"{img_file.stem}_aug_{aug_idx}{img_file.suffix}"
                aug_image_path = output_images_path / aug_filename
                cv2.imwrite(str(aug_image_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

                # Save augmented labels
                aug_label_filename = f"{img_file.stem}_aug_{aug_idx}.txt"
                aug_label_path = output_labels_path / aug_label_filename

                with open(aug_label_path, 'w') as f:
                    for bbox, label in zip(aug_bboxes, aug_labels):
                        f.write(f"{label} {' '.join(f'{x:.6f}' for x in bbox)}\n")

        logger.info("Dataset augmentation completed")


def create_sample_dataset(output_dir='data/sample_dataset'):
    """
    Create a sample dataset for testing

    Args:
        output_dir (str): Output directory for sample dataset
    """
    logger.info("Creating sample dataset...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create sample images and annotations
    classes = ['person', 'car', 'dog']

    # Create data.yaml
    data_config = {
        'train': str(output_path / 'train' / 'images'),
        'val': str(output_path / 'val' / 'images'),
        'test': str(output_path / 'test' / 'images'),
        'nc': len(classes),
        'names': classes
    }

    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    # Create dummy images and annotations
    for split in ['train', 'val', 'test']:
        img_dir = output_path / split / 'images'
        label_dir = output_path / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):  # 10 images per split
            # Create dummy image (100x100 RGB)
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = img_dir / f'{split}_{i}.jpg'
            cv2.imwrite(str(img_path), image)

            # Create dummy annotation
            label_path = label_dir / f'{split}_{i}.txt'
            with open(label_path, 'w') as f:
                # Random bounding box
                class_id = np.random.randint(0, len(classes))
                x_center = np.random.uniform(0.2, 0.8)
                y_center = np.random.uniform(0.2, 0.8)
                width = np.random.uniform(0.1, 0.3)
                height = np.random.uniform(0.1, 0.3)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    logger.info(f"Sample dataset created at {output_path}")
    return str(output_path / 'data.yaml')


if __name__ == "__main__":
    # Example usage
    preprocessor = YOLODataPreprocessor()

    # Create sample dataset for testing
    data_yaml = create_sample_dataset()

    # Validate dataset
    stats = preprocessor.validate_dataset('data/sample_dataset/train/images', 'data/sample_dataset/train/labels')
    print("Dataset Statistics:", stats)

    # Visualize sample annotations
    preprocessor.visualize_annotations(
        'data/sample_dataset/train/images',
        'data/sample_dataset/train/labels',
        ['person', 'car', 'dog'],
        num_samples=3
    )

    logger.info("Data preprocessing pipeline ready")