"""
Test OCR Pipeline
"""
import pytest
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline import OCRPipeline


@pytest.fixture
def sample_document_image():
    """Create a sample document image"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some text
    cv2.putText(img, "CAN CUOC CONG DAN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "So: 001234567890", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Ho va ten: NGUYEN VAN A", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Ngay sinh: 15/03/1990", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img


def test_pipeline_initialization():
    """Test OCR pipeline initialization"""
    try:
        pipeline = OCRPipeline(
            detector_type='craft',
            recognizer_type='vietocr',
            device='cpu'
        )
        assert pipeline is not None
        assert pipeline.detector is not None
        assert pipeline.recognizer is not None
    except ImportError as e:
        pytest.skip(f"Required library not installed: {e}")


def test_pipeline_process_image(sample_document_image):
    """Test pipeline image processing"""
    try:
        pipeline = OCRPipeline(
            detector_type='craft',
            recognizer_type='vietocr',
            device='cpu'
        )
        
        result = pipeline.process_image_array(sample_document_image, source="test")
        
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'document_type' in result
        assert 'raw_text' in result
        assert 'text_regions' in result
        assert 'extracted_fields' in result
        assert 'confidence' in result
        
    except ImportError as e:
        pytest.skip(f"Required library not installed: {e}")


def test_pipeline_output_structure(sample_document_image):
    """Test pipeline output structure"""
    try:
        pipeline = OCRPipeline(
            detector_type='craft',
            recognizer_type='vietocr',
            device='cpu'
        )
        
        result = pipeline.process_image_array(sample_document_image)
        
        # Check result structure
        assert isinstance(result['text_regions'], list)
        assert isinstance(result['extracted_fields'], dict)
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        
    except ImportError as e:
        pytest.skip(f"Required library not installed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
