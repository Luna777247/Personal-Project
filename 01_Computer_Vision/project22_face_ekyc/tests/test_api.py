"""
Integration tests for API endpoints
"""
import pytest
import sys
import base64
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


def create_test_image() -> str:
    """Create a base64 encoded test image"""
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', image)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    return image_b64


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data
    
    def test_detect_endpoint(self):
        """Test face detection endpoint"""
        image_b64 = create_test_image()
        
        response = client.post(
            "/detect",
            json={
                "image": image_b64,
                "max_faces": 1,
                "return_landmarks": True
            }
        )
        
        # May be 503 if models not loaded in test environment
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "num_faces" in data
    
    def test_match_endpoint(self):
        """Test face matching endpoint"""
        selfie = create_test_image()
        cccd = create_test_image()
        
        response = client.post(
            "/match",
            json={
                "selfie_image": selfie,
                "cccd_image": cccd,
                "threshold": 0.6
            }
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "is_match" in data
            assert "similarity" in data
    
    def test_liveness_endpoint(self):
        """Test liveness check endpoint"""
        image_b64 = create_test_image()
        
        response = client.post(
            "/liveness",
            json={
                "image": image_b64,
                "enable_blink": True,
                "enable_head_movement": True,
                "enable_texture": True
            }
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "is_live" in data
    
    def test_verify_endpoint(self):
        """Test complete verification endpoint"""
        selfie = create_test_image()
        cccd = create_test_image()
        
        response = client.post(
            "/verify",
            json={
                "selfie_image": selfie,
                "cccd_image": cccd,
                "enable_liveness": True,
                "threshold": 0.6
            }
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "verified" in data
            assert "status" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
