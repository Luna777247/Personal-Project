"""
YOLOv8 Object Detection Tests
============================

Unit tests for YOLOv8 object detection components.

Author: AI Assistant
Date: 2025
"""

import unittest
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yolov8_detector import YOLOv8Detector, create_data_yaml
from data_preprocessing import YOLODataPreprocessor

class TestYOLOv8Detector(unittest.TestCase):
    """Test cases for YOLOv8 detector"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = YOLOv8Detector(model_size='yolov8n', device='cpu')
        self.test_image = self._create_test_image()

    def _create_test_image(self):
        """Create a test image for inference"""
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, image)
        self.addCleanup(os.unlink, temp_file.name)
        return temp_file.name

    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.model_size, 'yolov8n')
        self.assertEqual(self.detector.device, 'cpu')

    def test_model_loading(self):
        """Test model loading"""
        # Test loading pretrained model
        self.detector.load_model()
        self.assertIsNotNone(self.detector.model)

    def test_inference(self):
        """Test basic inference"""
        if self.detector.model is None:
            self.detector.load_model()

        results = self.detector.predict(self.test_image, save=False, verbose=False)
        self.assertIsInstance(results, list)

    def test_data_yaml_creation(self):
        """Test data.yaml creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = create_data_yaml(
                train_path='data/train/images',
                val_path='data/val/images',
                test_path='data/test/images',
                classes=['class1', 'class2'],
                save_path=os.path.join(temp_dir, 'data.yaml')
            )

            self.assertTrue(os.path.exists(yaml_path))

class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing"""

    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = YOLODataPreprocessor()

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        self.assertIsNotNone(self.preprocessor)
        self.assertIsNotNone(self.preprocessor.base_path)

    def test_augmentation_setup(self):
        """Test augmentation pipeline setup"""
        self.preprocessor.setup_augmentation()
        self.assertIsNotNone(self.preprocessor.augmentation_pipeline)

class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_full_pipeline(self):
        """Test full detection pipeline"""
        detector = YOLOv8Detector(model_size='yolov8n', device='cpu')

        # Create test image
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, image)

            try:
                # Load model and run inference
                detector.load_model()
                results = detector.predict(temp_file.name, save=False, verbose=False)

                # Check results structure
                self.assertIsInstance(results, list)
                if results:
                    self.assertTrue(hasattr(results[0], 'boxes'))

            finally:
                try:
                    os.unlink(temp_file.name)
                except PermissionError:
                    # File may still be in use on Windows, skip cleanup
                    pass

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestYOLOv8Detector))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running YOLOv8 Object Detection Tests...")
    print("=" * 50)

    success = run_tests()

    if success:
        print("\\n✅ All tests passed!")
    else:
        print("\\n❌ Some tests failed!")
        sys.exit(1)