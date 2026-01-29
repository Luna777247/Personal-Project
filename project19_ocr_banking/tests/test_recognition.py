"""
Test Recognition Module
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recognition.vietocr_recognizer import VietOCRRecognizer, PaddleOCRRecognizer
from src.recognition import BaseTextRecognizer, RecognitionResult


@pytest.fixture
def sample_text_image():
    """Create a sample text image"""
    img = np.ones((32, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello World", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def test_vietocr_initialization():
    """Test VietOCR recognizer initialization"""
    try:
        recognizer = VietOCRRecognizer(device='cpu')
        assert recognizer is not None
        assert recognizer.device == 'cpu'
    except ImportError:
        pytest.skip("VietOCR not installed")


def test_recognition_output(sample_text_image):
    """Test recognition output format"""
    try:
        recognizer = VietOCRRecognizer(device='cpu')
        text = recognizer.recognize(sample_text_image)
        
        assert isinstance(text, str)
    except ImportError:
        pytest.skip("VietOCR not installed")


def test_recognition_result():
    """Test RecognitionResult class"""
    result = RecognitionResult(
        text="Test",
        confidence=0.95,
        bbox=np.array([10, 10, 100, 30])
    )
    
    assert result.text == "Test"
    assert result.confidence == 0.95
    assert result.bbox is not None
    
    # Test to_dict
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert 'text' in result_dict
    assert 'confidence' in result_dict


def test_base_recognizer_preprocessing(sample_text_image):
    """Test base recognizer preprocessing"""
    class TestRecognizer(BaseTextRecognizer):
        def recognize(self, image):
            return "test"
        
        def recognize_batch(self, images):
            return ["test"] * len(images)
    
    recognizer = TestRecognizer()
    
    # Test preprocessing
    preprocessed = recognizer.preprocess_image(sample_text_image, target_height=32)
    assert preprocessed is not None
    assert preprocessed.shape[0] == 32
    
    # Test postprocessing
    text = recognizer.postprocess_text("  Test  Text  ")
    assert text == "Test Text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
