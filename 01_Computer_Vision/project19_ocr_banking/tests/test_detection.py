"""
Test Detection Module
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.craft_detector import CRAFTDetector, DBNetDetector
from src.detection import BaseTextDetector


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple image with text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST TEXT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img


def test_craft_detector_initialization():
    """Test CRAFT detector initialization"""
    try:
        detector = CRAFTDetector(cuda=False)
        assert detector is not None
        assert detector.device == 'cpu'
    except ImportError:
        pytest.skip("CRAFT not installed")


def test_dbnet_detector_initialization():
    """Test DBNet detector initialization"""
    detector = DBNetDetector()
    assert detector is not None


def test_detection_output_format(sample_image):
    """Test detection output format"""
    try:
        detector = CRAFTDetector(cuda=False)
        boxes = detector.detect(sample_image)
        
        assert isinstance(boxes, list)
        
        if boxes:
            assert isinstance(boxes[0], (list, np.ndarray))
    except ImportError:
        pytest.skip("CRAFT not installed")


def test_base_detector_methods():
    """Test BaseTextDetector utility methods"""
    detector = BaseTextDetector()
    
    # Test box sorting
    boxes = [
        np.array([10, 50, 100, 70]),
        np.array([10, 10, 100, 30]),
        np.array([10, 90, 100, 110])
    ]
    
    sorted_boxes = detector.sort_boxes(boxes, method='top-to-bottom')
    assert len(sorted_boxes) == 3
    
    # Test box grouping
    lines = detector.group_boxes_by_line(boxes, threshold=30)
    assert isinstance(lines, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
