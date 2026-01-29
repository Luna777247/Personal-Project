"""
Unit tests for Face Embedding
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding import FaceEmbedder


class TestFaceEmbedder:
    """Test face embedder"""
    
    @pytest.fixture
    def embedder(self):
        """Initialize embedder"""
        return FaceEmbedder(model_pack="buffalo_l", ctx_id=-1)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_embedder_initialization(self, embedder):
        """Test embedder initializes correctly"""
        assert embedder is not None
        assert embedder.app is not None
    
    def test_compute_similarity_cosine(self, embedder):
        """Test cosine similarity computation"""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        
        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        similarity = embedder.compute_similarity(emb1, emb2, metric="cosine")
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_compute_similarity_euclidean(self, embedder):
        """Test euclidean similarity computation"""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        
        similarity = embedder.compute_similarity(emb1, emb2, metric="euclidean")
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_verify_faces(self, embedder):
        """Test face verification"""
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        
        # Normalize
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        is_same, similarity = embedder.verify_faces(emb1, emb2, threshold=0.5)
        
        assert isinstance(is_same, bool)
        assert isinstance(similarity, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
