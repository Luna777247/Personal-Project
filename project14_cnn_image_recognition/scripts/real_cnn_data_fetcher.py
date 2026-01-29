#!/usr/bin/env python3
"""
Real Data Fetcher for CNN Image Recognition Project
Downloads CIFAR-10 dataset and prepares it for training
"""

import os
import sys
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import shutil
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class CNNDataFetcher:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_fetcher.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def download_cifar10(self):
        """Download CIFAR-10 dataset using TensorFlow Datasets"""
        self.logger.info("🔄 Downloading CIFAR-10 dataset...")

        try:
            # Download CIFAR-10 dataset
            (ds_train, ds_test), ds_info = tfds.load(
                'cifar10',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
                download=True
            )

            self.logger.info("✅ CIFAR-10 dataset downloaded successfully")
            return ds_train, ds_test, ds_info

        except Exception as e:
            self.logger.error(f"❌ Failed to download CIFAR-10: {str(e)}")
            return None, None, None

    def save_images_to_directory(self, dataset, split_name, class_names):
        """Save dataset images to directory structure"""
        self.logger.info(f"💾 Saving {split_name} images to directory structure...")

        split_dir = self.raw_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create class directories
        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)

        # Save images
        image_count = 0
        for image, label in dataset:
            class_name = class_names[label.numpy()]
            class_dir = split_dir / class_name

            # Convert tensor to PIL Image
            image_array = image.numpy()
            pil_image = Image.fromarray(image_array)

            # Save image
            image_path = class_dir / f"{image_count:05d}.png"
            pil_image.save(image_path)
            image_count += 1

            if image_count % 1000 == 0:
                self.logger.info(f"   Saved {image_count} images...")

        self.logger.info(f"✅ Saved {image_count} {split_name} images")
        return image_count

    def prepare_dataset(self):
        """Main function to download and prepare the dataset"""
        self.logger.info("🚀 Starting CIFAR-10 dataset preparation...")

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Download dataset
        ds_train, ds_test, ds_info = self.download_cifar10()
        if ds_train is None:
            return False

        # Get class names
        class_names = ds_info.features['label'].names
        self.logger.info(f"📋 Dataset classes: {class_names}")

        # Save training images
        train_count = self.save_images_to_directory(ds_train, 'train', class_names)

        # Save test images
        test_count = self.save_images_to_directory(ds_test, 'test', class_names)

        # Create validation split from training data (10% of training)
        self.create_validation_split(class_names)

        self.logger.info("✅ Dataset preparation completed!")
        self.logger.info(f"📊 Training images: {train_count}")
        self.logger.info(f"📊 Test images: {test_count}")
        self.logger.info(f"📁 Data saved to: {self.raw_dir}")

        return True

    def create_validation_split(self, class_names):
        """Create validation split from training data"""
        self.logger.info("🔄 Creating validation split...")

        train_dir = self.raw_dir / 'train'
        val_dir = self.raw_dir / 'val'
        val_dir.mkdir(exist_ok=True)

        val_ratio = 0.1  # 10% for validation

        for class_name in class_names:
            class_train_dir = train_dir / class_name
            class_val_dir = val_dir / class_name
            class_val_dir.mkdir(exist_ok=True)

            if not class_train_dir.exists():
                continue

            # Get all images in class
            images = list(class_train_dir.glob('*.png'))
            val_count = int(len(images) * val_ratio)

            # Move validation images
            val_images = images[:val_count]
            for img_path in val_images:
                shutil.move(str(img_path), str(class_val_dir / img_path.name))

        self.logger.info("✅ Validation split created")

    def verify_dataset(self):
        """Verify the downloaded dataset"""
        self.logger.info("🔍 Verifying dataset...")

        splits = ['train', 'val', 'test']
        class_counts = {}

        for split in splits:
            split_dir = self.raw_dir / split
            if not split_dir.exists():
                self.logger.warning(f"⚠️  {split} directory not found")
                continue

            split_count = 0
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    image_count = len(list(class_dir.glob('*.png')))

                    if class_name not in class_counts:
                        class_counts[class_name] = {}
                    class_counts[class_name][split] = image_count
                    split_count += image_count

            self.logger.info(f"   {split}: {split_count} images")

        # Print summary
        self.logger.info("📊 Dataset Summary:")
        for class_name, counts in class_counts.items():
            total = sum(counts.values())
            self.logger.info(f"   {class_name}: {total} images")

        return True

def main():
    """Main execution function"""
    print("🖼️  CNN Image Recognition - Real Data Fetcher")
    print("=" * 50)

    fetcher = CNNDataFetcher()

    try:
        # Prepare dataset
        success = fetcher.prepare_dataset()

        if success:
            # Verify dataset
            fetcher.verify_dataset()

            print("\n✅ Dataset preparation completed successfully!")
            print("📁 Check the 'data/raw/' directory for the downloaded images")
            print("🚀 You can now run the data preprocessing script")

        else:
            print("\n❌ Dataset preparation failed!")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())