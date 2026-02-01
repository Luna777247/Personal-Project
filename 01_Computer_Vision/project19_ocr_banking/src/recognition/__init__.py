"""
Text Recognition Module
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTextRecognizer(ABC):
    """Abstract base class for text recognizers"""
    
    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        """Recognize text from image"""
        pass
    
    @abstractmethod
    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        """Recognize text from multiple images"""
        pass
    
    def preprocess_image(self, image: np.ndarray, 
                        target_height: Optional[int] = 32,
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for recognition
        
        Args:
            image: Input image
            target_height: Target height for resizing
            normalize: Apply normalization
            
        Returns:
            Preprocessed image
        """
        import cv2
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize maintaining aspect ratio
        if target_height:
            h, w = gray.shape
            aspect_ratio = w / h
            new_w = int(target_height * aspect_ratio)
            gray = cv2.resize(gray, (new_w, target_height))
        
        # Normalize
        if normalize:
            gray = gray.astype(np.float32) / 255.0
        
        return gray
    
    def postprocess_text(self, text: str, 
                        remove_spaces: bool = False,
                        lowercase: bool = False) -> str:
        """
        Postprocess recognized text
        
        Args:
            text: Recognized text
            remove_spaces: Remove whitespace
            lowercase: Convert to lowercase
            
        Returns:
            Postprocessed text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove extra spaces
        if remove_spaces:
            text = ''.join(text.split())
        else:
            text = ' '.join(text.split())
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        return text


class RecognitionResult:
    """Result of text recognition"""
    
    def __init__(self, 
                 text: str,
                 confidence: float = 1.0,
                 bbox: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None):
        """
        Initialize recognition result
        
        Args:
            text: Recognized text
            confidence: Confidence score (0-1)
            bbox: Bounding box coordinates
            metadata: Additional metadata
        """
        self.text = text
        self.confidence = confidence
        self.bbox = bbox
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"RecognitionResult(text='{self.text}', confidence={self.confidence:.2f})"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox.tolist() if self.bbox is not None else None,
            'metadata': self.metadata
        }


def merge_recognition_results(results: List[RecognitionResult], 
                              separator: str = ' ') -> RecognitionResult:
    """
    Merge multiple recognition results
    
    Args:
        results: List of recognition results
        separator: Separator between texts
        
    Returns:
        Merged result
    """
    if not results:
        return RecognitionResult("", 0.0)
    
    texts = [r.text for r in results]
    confidences = [r.confidence for r in results]
    
    merged_text = separator.join(texts)
    avg_confidence = sum(confidences) / len(confidences)
    
    return RecognitionResult(merged_text, avg_confidence)


if __name__ == "__main__":
    # Test utilities
    result1 = RecognitionResult("Hello", 0.95)
    result2 = RecognitionResult("World", 0.92)
    
    print(result1)
    print(result2)
    
    merged = merge_recognition_results([result1, result2])
    print(f"Merged: {merged}")
    print(f"Dict: {merged.to_dict()}")
