#!/usr/bin/env python3
"""
Real Data Fetcher for YOLOv8 Object Detection Project
Downloads COCO128 dataset for object detection training
"""

import os
import zipfile
import requests
from pathlib import Path
import yaml
import shutil
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_coco128_dataset():
    """Download COCO128 dataset from Ultralytics"""
    logger.info("Downloading COCO128 dataset...")

    # COCO128 is a small subset of COCO dataset (128 images)
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = "coco128.zip"

    try:
        # Download the dataset
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded {zip_path}")

        # Extract the dataset
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        # Clean up zip file
        os.remove(zip_path)

        # Move to data directory
        if os.path.exists("coco128"):
            data_dir = "data/coco128"
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            shutil.move("coco128", data_dir)
            logger.info(f"Dataset moved to {data_dir}")
            return data_dir

    except Exception as e:
        logger.error(f"Failed to download COCO128: {e}")
        return None

def create_data_config(data_dir):
    """Create data.yaml configuration file"""
    data_yaml = {
        'names': [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ],
        'nc': 80,  # number of classes
        'train': f'{data_dir}/images/train2017',
        'val': f'{data_dir}/images/train2017',  # Using train for both for small dataset
        'test': f'{data_dir}/images/train2017'
    }

    config_path = os.path.join(data_dir, "data.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    logger.info(f"Created data configuration at {config_path}")
    return config_path

def create_dataset_info(data_dir, config_path):
    """Create dataset information file"""
    info = {
        "dataset_name": "COCO128",
        "description": "Small subset of COCO dataset with 128 images for YOLOv8 training",
        "source": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
        "num_classes": 80,
        "classes": [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ],
        "download_timestamp": datetime.now().isoformat(),
        "data_directory": data_dir,
        "config_file": config_path
    }

    info_path = os.path.join(data_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        import json
        json.dump(info, f, indent=2)

    logger.info(f"Dataset info saved to {info_path}")

def main():
    """Main function to fetch and prepare COCO128 dataset"""
    logger.info("🚀 Starting COCO128 dataset fetch for YOLOv8...")

    try:
        # Create data directory
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # Download and extract dataset
        dataset_dir = download_coco128_dataset()
        if not dataset_dir:
            raise Exception("Failed to download dataset")

        # Create data configuration
        config_path = create_data_config(dataset_dir)

        # Create dataset info
        create_dataset_info(dataset_dir, config_path)

        # Verify the dataset
        images_dir = os.path.join(dataset_dir, "images", "train2017")
        labels_dir = os.path.join(dataset_dir, "labels", "train2017")

        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            num_images = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
            num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

            logger.info("✅ COCO128 dataset fetch completed successfully!")
            logger.info(f"📊 Dataset: {num_images} images, {num_labels} label files")
            logger.info(f"🎯 Classes: 80 COCO object classes")
            logger.info(f"📁 Dataset location: {dataset_dir}")
            logger.info(f"⚙️  Config file: {config_path}")

            return True
        else:
            raise Exception("Dataset structure verification failed")

    except Exception as e:
        logger.error(f"❌ Dataset fetch failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)