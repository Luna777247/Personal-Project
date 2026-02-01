"""
Unit tests for Liveness Detection
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from liveness import BlinkDetector, HeadMovementDetector, TextureAnalyzer, LivenessDetector


class TestBlinkDetector:
    """Test blink detector"""
    
    @pytest.fixture
    def detector(self):
        """Initialize blink detector"""
        return BlinkDetector()
    
    def test_detector_initialization(self, detector):
        """Test initialization"""
        assert detector.ear_threshold > 0
        assert detector.blink_counter == 0
    
    def test_eye_aspect_ratio(self, detector):
        """Test EAR calculation"""
        # Create dummy eye landmarks (6 points)
        eye = np.array([
            [0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]
        ], dtype=np.float32)
        
        ear = detector.eye_aspect_ratio(eye)
        assert isinstance(ear, float)
        assert ear > 0


class TestHeadMovementDetector:
    """Test head movement detector"""
    
    @pytest.fixture
    def detector(self):
        """Initialize head movement detector"""
        return HeadMovementDetector()
    
    def test_detector_initialization(self, detector):
        """Test initialization"""
        assert detector.yaw_threshold > 0
        assert detector.movement_count == 0
    
    def test_reset(self, detector):
        """Test reset"""
        detector.movement_count = 5
        detector.reset()
        assert detector.movement_count == 0


class TestTextureAnalyzer:
    """Test texture analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Initialize texture analyzer"""
        return TextureAnalyzer()
    
    @pytest.fixture
    def sample_face(self):
        """Create sample face image"""
        return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    def test_analyzer_initialization(self, analyzer):
        """Test initialization"""
        assert analyzer.blur_threshold > 0
    
    def test_detect_blur(self, analyzer):
        """Test blur detection"""
        gray = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
        blur_score = analyzer.detect_blur(gray)
        
        assert isinstance(blur_score, (float, np.floating))
        assert blur_score >= 0
    
    def test_analyze_color_diversity(self, analyzer, sample_face):
        """Test color diversity analysis"""
        diversity = analyzer.analyze_color_diversity(sample_face)
        
        assert isinstance(diversity, (float, np.floating))
        assert diversity >= 0
    
    def test_analyze_texture(self, analyzer, sample_face):
        """Test full texture analysis"""
        is_live, details = analyzer.analyze_texture(sample_face)
        
        assert isinstance(is_live, bool)
        assert isinstance(details, dict)
        assert 'blur_score' in details
        assert 'color_diversity' in details


class TestLivenessDetector:
    """Test combined liveness detector"""
    
    @pytest.fixture
    def detector(self):
        """Initialize liveness detector"""
        return LivenessDetector(
            enable_blink=False,  # Disable for testing
            enable_head_movement=False,
            enable_texture=True
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, detector):
        """Test initialization"""
        assert detector is not None
        assert detector.enable_texture == True
    
    def test_reset(self, detector):
        """Test reset"""
        detector.reset()
        # Should not raise exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
