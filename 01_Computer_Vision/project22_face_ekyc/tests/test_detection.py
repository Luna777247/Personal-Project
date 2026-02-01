"""
Unit tests for Face Detection
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection import FaceDetector


class TestFaceDetector:
    """Test face detector"""
    
    @pytest.fixture
    def detector(self):
        """Initialize detector"""
        return FaceDetector(model_pack="buffalo_l", ctx_id=-1)  # CPU mode
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image"""
        # Create simple test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None
        assert detector.app is not None
    
    def test_detect_faces_returns_list(self, detector, sample_image):
        """Test detect_faces returns a list"""
        faces = detector.detect_faces(sample_image)
        assert isinstance(faces, list)
    
    def test_get_largest_face(self, detector, sample_image):
        """Test get_largest_face"""
        face = detector.get_largest_face(sample_image)
        # May be None if no face detected in random image
        assert face is None or isinstance(face, dict)
    
    def test_draw_faces(self, detector, sample_image):
        """Test drawing faces on image"""
        faces = [
            {
                'bbox': [100, 100, 200, 200],
                'det_score': 0.95,
                'landmark': [[120, 130], [180, 130], [150, 150], [130, 180], [170, 180]]
            }
        ]
        result = detector.draw_faces(sample_image, faces)
        assert result.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
