import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import numpy as np

class DataPreprocessor:
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, 'raw')
        self.processed_dir = os.path.join(base_dir, 'processed')

    def create_directory_structure(self, classes):
        """Create train/val/test directory structure"""
        splits = ['train', 'val', 'test']

        for split in splits:
            split_dir = os.path.join(self.processed_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            for class_name in classes:
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

        print(f"Created directory structure for {len(classes)} classes")

    def get_classes_from_raw_data(self):
        """Automatically detect classes from raw data directory"""
        train_dir = os.path.join(self.raw_dir, 'train')
        if not os.path.exists(train_dir):
            print("Error: Raw train directory not found")
            return []

        classes = []
        for item in os.listdir(train_dir):
            item_path = os.path.join(train_dir, item)
            if os.path.isdir(item_path):
                classes.append(item)

        classes.sort()  # Ensure consistent ordering
        print(f"Detected {len(classes)} classes: {classes}")
        return classes

    def split_dataset(self, source_dir, classes, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test sets"""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

        for class_name in classes:
            class_source_dir = os.path.join(source_dir, class_name)

            if not os.path.exists(class_source_dir):
                print(f"Warning: Class directory {class_source_dir} does not exist")
                continue

            # Get all images in the class
            images = [f for f in os.listdir(class_source_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if len(images) == 0:
                print(f"Warning: No images found in {class_source_dir}")
                continue

            # Split the images
            train_images, temp_images = train_test_split(
                images, train_size=train_ratio, random_state=42
            )

            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_images, test_images = train_test_split(
                temp_images, train_size=val_ratio_adjusted, random_state=42
            )

            # Copy images to respective directories
            self._copy_images(class_source_dir, train_images,
                            os.path.join(self.processed_dir, 'train', class_name))
            self._copy_images(class_source_dir, val_images,
                            os.path.join(self.processed_dir, 'val', class_name))
            self._copy_images(class_source_dir, test_images,
                            os.path.join(self.processed_dir, 'test', class_name))

            print(f"Class {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    def split_cifar_dataset(self, source_dir, classes, val_ratio=0.2):
        """Split CIFAR-10 dataset where train and test are separate subdirectories"""
        train_source_dir = os.path.join(source_dir, 'train')
        test_source_dir = os.path.join(source_dir, 'test')

        if not os.path.exists(train_source_dir):
            print(f"Warning: Train directory {train_source_dir} does not exist")
            return

        if not os.path.exists(test_source_dir):
            print(f"Warning: Test directory {test_source_dir} does not exist")
            return

        for class_name in classes:
            # Process train data
            train_class_dir = os.path.join(train_source_dir, class_name)
            if not os.path.exists(train_class_dir):
                print(f"Warning: Train class directory {train_class_dir} does not exist")
                continue

            train_images = [f for f in os.listdir(train_class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if len(train_images) == 0:
                print(f"Warning: No train images found in {train_class_dir}")
                continue

            # Split train into train and val
            train_split, val_split = train_test_split(
                train_images, test_size=val_ratio, random_state=42
            )

            # Process test data
            test_class_dir = os.path.join(test_source_dir, class_name)
            if not os.path.exists(test_class_dir):
                print(f"Warning: Test class directory {test_class_dir} does not exist")
                continue

            test_images = [f for f in os.listdir(test_class_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            # Copy images to respective directories
            self._copy_images(train_class_dir, train_split,
                            os.path.join(self.processed_dir, 'train', class_name))
            self._copy_images(train_class_dir, val_split,
                            os.path.join(self.processed_dir, 'val', class_name))
            self._copy_images(test_class_dir, test_images,
                            os.path.join(self.processed_dir, 'test', class_name))

            print(f"Class {class_name}: {len(train_split)} train, {len(val_split)} val, {len(test_images)} test")

    def _copy_images(self, source_dir, image_list, dest_dir):
        """Copy images from source to destination"""
        for image in image_list:
            src_path = os.path.join(source_dir, image)
            dest_path = os.path.join(dest_dir, image)
            shutil.copy2(src_path, dest_path)

    def validate_images(self, directory):
        """Validate and clean corrupted images"""
        valid_count = 0
        invalid_count = 0

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()  # Verify image integrity
                        valid_count += 1
                    except Exception as e:
                        print(f"Invalid image {file_path}: {e}")
                        os.remove(file_path)  # Remove corrupted images
                        invalid_count += 1

        print(f"Validation complete: {valid_count} valid, {invalid_count} invalid images removed")
        return valid_count, invalid_count

    def get_dataset_stats(self, directory):
        """Get statistics about the dataset"""
        stats = {}

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(directory, split)
            if os.path.exists(split_dir):
                split_stats = {}
                total_images = 0

                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.isdir(class_dir):
                        num_images = len([f for f in os.listdir(class_dir)
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                        split_stats[class_name] = num_images
                        total_images += num_images

                split_stats['total'] = total_images
                stats[split] = split_stats

        return stats

    def print_stats(self, stats):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)

        for split, split_stats in stats.items():
            print(f"\n{split.upper()} SET:")
            total = split_stats.get('total', 0)
            print(f"  Total images: {total}")

            for class_name, count in split_stats.items():
                if class_name != 'total':
                    percentage = (count / total * 100) if total > 0 else 0
                    print(".1f")

    def create_data_augmentation_samples(self, source_dir, target_dir, num_augmented=5):
        """Create additional augmented samples for classes with few images"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
        import cv2

        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        os.makedirs(target_dir, exist_ok=True)

        for class_name in os.listdir(source_dir):
            class_source_dir = os.path.join(source_dir, class_name)
            class_target_dir = os.path.join(target_dir, class_name)

            if not os.path.isdir(class_source_dir):
                continue

            os.makedirs(class_target_dir, exist_ok=True)

            images = [f for f in os.listdir(class_source_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for image_file in images:
                img_path = os.path.join(class_source_dir, image_file)
                img = load_img(img_path)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # Generate augmented images
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=class_target_dir,
                                        save_prefix=f'aug_{os.path.splitext(image_file)[0]}',
                                        save_format='jpeg'):
                    i += 1
                    if i >= num_augmented:
                        break

        print(f"Created augmented samples in {target_dir}")

def main():
    """Example usage of data preprocessor"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Auto-detect classes from raw data directory
    raw_data_dir = 'data/raw'
    classes = preprocessor.get_classes_from_raw_data()
    
    if not classes:
        print(f"No classes detected in {raw_data_dir}. Please check if data has been downloaded.")
        return

    print(f"Detected classes: {classes}")

    # Create directory structure
    preprocessor.create_directory_structure(classes)

    # Split CIFAR-10 dataset (train and test are separate subdirectories)
    preprocessor.split_cifar_dataset(raw_data_dir, classes)

    # Validate images
    preprocessor.validate_images('data/processed')

    # Get and print statistics
    stats = preprocessor.get_dataset_stats('data/processed')
    preprocessor.print_stats(stats)

    print("\nData preprocessing complete!")
    print("Next steps:")
    print("1. Review the statistics above")
    print("2. If needed, create augmented samples for underrepresented classes")
    print("3. Run the CNN training script")

if __name__ == "__main__":
    main()