"""
Face Matching Module for eKYC
Matches selfie with ID card photo (CCCD - Vietnamese Citizen Identity Card)
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path

from ..detection import FaceDetector
from ..embedding import FaceEmbedder

logger = logging.getLogger(__name__)


class FaceMatcher:
    """
    Face matching for eKYC verification
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.6,
        distance_threshold: float = 1.0,
        metric: str = "cosine",
        config: Optional[Dict] = None
    ):
        """
        Initialize face matcher
        
        Args:
            similarity_threshold: Cosine similarity threshold
            distance_threshold: Euclidean distance threshold
            metric: Similarity metric ("cosine" or "euclidean")
            config: Configuration dict
        """
        self.similarity_threshold = similarity_threshold
        self.distance_threshold = distance_threshold
        self.metric = metric
        
        config = config or {}
        
        # Initialize detector and embedder
        ctx_id = 0 if config.get("use_gpu", True) else -1
        model_pack = config.get("model_pack", "buffalo_l")
        
        self.detector = FaceDetector(
            model_pack=model_pack,
            ctx_id=ctx_id
        )
        
        self.embedder = FaceEmbedder(
            model_pack=model_pack,
            ctx_id=ctx_id
        )
        
        logger.info(f"FaceMatcher initialized with metric={metric}, threshold={similarity_threshold}")
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess image for matching
        
        Args:
            image: Input image
            target_size: Target size (width, height)
        
        Returns:
            Preprocessed image
        """
        # Resize if needed
        if target_size is not None:
            image = cv2.resize(image, target_size)
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def extract_face_from_cccd(
        self,
        cccd_image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Extract face from CCCD (ID card) image
        
        Args:
            cccd_image: CCCD image
        
        Returns:
            (face_crop, face_info) or None
        """
        # Preprocess CCCD
        preprocessed = self.preprocess_image(cccd_image)
        
        # Detect face
        face = self.detector.get_largest_face(preprocessed)
        
        if face is None:
            logger.warning("No face detected in CCCD")
            return None
        
        # Extract face region
        bbox = face['bbox']
        face_crop = self.detector.extract_face(preprocessed, bbox, margin=0.2)
        
        logger.info(f"Extracted face from CCCD: bbox={bbox}")
        return face_crop, face
    
    def match_faces(
        self,
        selfie_image: np.ndarray,
        cccd_image: np.ndarray,
        return_details: bool = True
    ) -> Tuple[bool, float, Optional[Dict]]:
        """
        Match selfie with CCCD photo
        
        Args:
            selfie_image: Selfie image
            cccd_image: CCCD (ID card) image
            return_details: Whether to return detailed info
        
        Returns:
            (is_match, similarity, details)
        """
        details = {} if return_details else None
        
        # Extract face from CCCD
        cccd_result = self.extract_face_from_cccd(cccd_image)
        if cccd_result is None:
            return False, 0.0, {"error": "No face in CCCD"}
        
        cccd_face_crop, cccd_face_info = cccd_result
        
        # Detect face in selfie
        selfie_face = self.detector.get_largest_face(selfie_image)
        if selfie_face is None:
            return False, 0.0, {"error": "No face in selfie"}
        
        # Extract embeddings
        cccd_embedding = self.embedder.extract_embedding(cccd_face_crop)
        selfie_embedding = self.embedder.extract_embedding(selfie_image)
        
        if cccd_embedding is None or selfie_embedding is None:
            return False, 0.0, {"error": "Failed to extract embeddings"}
        
        # Compute similarity
        similarity = self.embedder.compute_similarity(
            selfie_embedding,
            cccd_embedding,
            metric=self.metric
        )
        
        # Determine match
        is_match = similarity >= self.similarity_threshold
        
        if return_details:
            details = {
                "similarity": float(similarity),
                "threshold": self.similarity_threshold,
                "metric": self.metric,
                "is_match": is_match,
                "cccd_face": {
                    "bbox": cccd_face_info['bbox'],
                    "confidence": cccd_face_info['det_score']
                },
                "selfie_face": {
                    "bbox": selfie_face['bbox'],
                    "confidence": selfie_face['det_score']
                }
            }
        
        logger.info(f"Face matching: similarity={similarity:.4f}, is_match={is_match}")
        return is_match, similarity, details
    
    def verify_identity(
        self,
        selfie_image: np.ndarray,
        cccd_image: np.ndarray,
        liveness_check: bool = False,
        liveness_detector=None
    ) -> Dict:
        """
        Complete identity verification pipeline
        
        Args:
            selfie_image: Selfie image
            cccd_image: CCCD image
            liveness_check: Whether to perform liveness detection
            liveness_detector: LivenessDetector instance
        
        Returns:
            Verification result dict
        """
        result = {
            "verified": False,
            "confidence": 0.0,
            "face_match": None,
            "liveness": None,
            "errors": []
        }
        
        # Face matching
        is_match, similarity, match_details = self.match_faces(
            selfie_image,
            cccd_image,
            return_details=True
        )
        
        result["face_match"] = match_details
        
        if not is_match:
            result["errors"].append("Face does not match CCCD")
            return result
        
        # Liveness detection (optional)
        if liveness_check:
            if liveness_detector is None:
                result["errors"].append("Liveness detector not provided")
                return result
            
            # Detect face for liveness
            selfie_face = self.detector.get_largest_face(selfie_image)
            if selfie_face:
                # Note: Liveness needs video frames, single image is limited
                is_live, liveness_details = liveness_detector.detect(
                    selfie_image,
                    face_bbox=selfie_face['bbox']
                )
                
                result["liveness"] = liveness_details
                
                if not is_live:
                    result["errors"].append("Liveness check failed")
                    return result
        
        # Verification successful
        result["verified"] = True
        result["confidence"] = float(similarity)
        
        logger.info(f"Identity verification: verified={result['verified']}, confidence={result['confidence']:.4f}")
        return result


class CCCDProcessor:
    """
    CCCD (Vietnamese ID Card) specific processing
    """
    
    def __init__(self):
        """Initialize CCCD processor"""
        self.expected_formats = ["portrait", "landscape"]
        self.min_resolution = (300, 400)  # width x height
        
        logger.info("CCCDProcessor initialized")
    
    def validate_cccd_format(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Dict]:
        """
        Validate CCCD image format
        
        Args:
            image: CCCD image
        
        Returns:
            (is_valid, details)
        """
        height, width = image.shape[:2]
        
        details = {
            "width": width,
            "height": height,
            "resolution_ok": False,
            "aspect_ratio": width / height,
            "format": None
        }
        
        # Check resolution
        if width >= self.min_resolution[0] and height >= self.min_resolution[1]:
            details["resolution_ok"] = True
        
        # Determine format
        if width > height:
            details["format"] = "landscape"
        else:
            details["format"] = "portrait"
        
        is_valid = details["resolution_ok"] and details["format"] in self.expected_formats
        
        logger.debug(f"CCCD validation: {details}")
        return is_valid, details
    
    def detect_cccd_region(
        self,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect and extract CCCD region from photo
        (if CCCD is held in hand or placed on surface)
        
        Args:
            image: Input image containing CCCD
        
        Returns:
            CCCD region or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest rectangular contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        
        # Check if it's a rectangle (4 corners)
        if len(approx) == 4:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(approx)
            
            # Extract region
            cccd_region = image[y:y+h, x:x+w]
            
            logger.info(f"Detected CCCD region: {w}x{h}")
            return cccd_region
        
        return None
    
    def enhance_cccd_quality(
        self,
        cccd_image: np.ndarray
    ) -> np.ndarray:
        """
        Enhance CCCD image quality
        
        Args:
            cccd_image: CCCD image
        
        Returns:
            Enhanced image
        """
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(cccd_image, None, 10, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Adjust brightness and contrast
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced


def test_face_matcher():
    """Test face matcher"""
    # Create test images
    selfie = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cccd = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # Initialize matcher
    matcher = FaceMatcher(ctx_id=-1)  # CPU mode
    
    # Match faces
    is_match, similarity, details = matcher.match_faces(selfie, cccd)
    
    print(f"Match: {is_match}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Details: {details}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_face_matcher()
