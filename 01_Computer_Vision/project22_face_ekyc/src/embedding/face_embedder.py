"""
Face Embedding Module using ArcFace and InsightFace
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
import logging
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available")

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """
    Face embedding extraction using InsightFace (ArcFace)
    """
    
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        ctx_id: int = 0,  # GPU device ID, -1 for CPU
    ):
        """
        Initialize face embedder
        
        Args:
            model_pack: Model pack name (buffalo_l includes ArcFace)
            ctx_id: Device ID (0 for GPU, -1 for CPU)
        """
        self.model_pack = model_pack
        self.ctx_id = ctx_id
        
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is required")
        
        # Initialize FaceAnalysis (includes ArcFace recognition model)
        self.app = FaceAnalysis(
            name=model_pack,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id)
        
        logger.info(f"FaceEmbedder initialized with {model_pack}")
    
    def extract_embedding(
        self,
        image: np.ndarray,
        bbox: Optional[List[int]] = None,
        aligned: bool = False
    ) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Input image (BGR format)
            bbox: Face bounding box [x1, y1, x2, y2] (optional)
            aligned: Whether image is already aligned face
        
        Returns:
            512-dimensional embedding vector or None
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        if aligned and bbox is None:
            # Image is already aligned face, use directly
            faces = self.app.get(image_rgb, max_num=1)
            if faces:
                return faces[0].embedding
            return None
        
        # Detect and extract embedding
        faces = self.app.get(image_rgb, max_num=1)
        
        if not faces:
            logger.warning("No face detected for embedding extraction")
            return None
        
        # Return embedding of first face
        embedding = faces[0].embedding
        
        # Normalize embedding (L2 normalization)
        embedding = embedding / np.linalg.norm(embedding)
        
        logger.debug(f"Extracted embedding with shape {embedding.shape}")
        return embedding
    
    def extract_multiple_embeddings(
        self,
        image: np.ndarray,
        max_num: int = 0
    ) -> List[np.ndarray]:
        """
        Extract embeddings for multiple faces in image
        
        Args:
            image: Input image (BGR)
            max_num: Maximum number of faces (0 for all)
        
        Returns:
            List of embedding vectors
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Detect faces and extract embeddings
        faces = self.app.get(image_rgb, max_num=max_num)
        
        embeddings = []
        for face in faces:
            embedding = face.embedding
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        logger.debug(f"Extracted {len(embeddings)} embeddings")
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ("cosine" or "euclidean")
        
        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity (embeddings are already normalized)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        
        elif metric == "euclidean":
            # Euclidean distance (convert to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert to similarity score (0-1 range)
            similarity = 1.0 / (1.0 + distance)
            return float(similarity)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def verify_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.6,
        metric: str = "cosine"
    ) -> Tuple[bool, float]:
        """
        Verify if two embeddings belong to the same person
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Verification threshold
            metric: Similarity metric
        
        Returns:
            (is_same_person, similarity_score)
        """
        similarity = self.compute_similarity(embedding1, embedding2, metric)
        is_same = similarity >= threshold
        
        logger.debug(f"Verification: similarity={similarity:.4f}, threshold={threshold}, same={is_same}")
        return is_same, similarity
    
    def compare_faces(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.6,
        metric: str = "cosine"
    ) -> Tuple[bool, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compare two face images
        
        Args:
            image1: First face image
            image2: Second face image
            threshold: Verification threshold
            metric: Similarity metric
        
        Returns:
            (is_same_person, similarity, embedding1, embedding2)
        """
        # Extract embeddings
        embedding1 = self.extract_embedding(image1)
        embedding2 = self.extract_embedding(image2)
        
        if embedding1 is None or embedding2 is None:
            logger.warning("Failed to extract embeddings for comparison")
            return False, 0.0, None, None
        
        # Verify
        is_same, similarity = self.verify_faces(embedding1, embedding2, threshold, metric)
        
        return is_same, similarity, embedding1, embedding2


class ArcFaceEmbedder:
    """
    Standalone ArcFace embedder using ONNX model
    Note: InsightFace already includes ArcFace, this is for standalone usage
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (112, 112)
    ):
        """
        Initialize ArcFace embedder with ONNX model
        
        Args:
            model_path: Path to ONNX model
            input_size: Input face size
        """
        import onnxruntime as ort
        
        self.input_size = input_size
        
        # Load ONNX model
        if Path(model_path).exists():
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"ArcFace model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for ArcFace
        
        Args:
            face_image: Aligned face image
        
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        face = cv2.resize(face_image, self.input_size)
        
        # Normalize
        face = (face - 127.5) / 128.0
        
        # Transpose to CHW format
        face = face.transpose(2, 0, 1)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0).astype(np.float32)
        
        return face
    
    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract embedding from aligned face
        
        Args:
            aligned_face: Aligned face image (112x112)
        
        Returns:
            512-dimensional embedding
        """
        # Preprocess
        input_blob = self.preprocess(aligned_face)
        
        # Inference
        embedding = self.session.run(None, {self.input_name: input_blob})[0]
        
        # Squeeze batch dimension
        embedding = embedding.squeeze()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding


def test_face_embedder():
    """Test face embedder"""
    # Create test face images
    test_image1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_image2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize embedder
    embedder = FaceEmbedder(model_pack="buffalo_l", ctx_id=-1)  # CPU mode
    
    # Compare faces
    is_same, similarity, emb1, emb2 = embedder.compare_faces(test_image1, test_image2)
    
    print(f"Same person: {is_same}")
    print(f"Similarity: {similarity:.4f}")
    
    if emb1 is not None:
        print(f"Embedding shape: {emb1.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_face_embedder()
