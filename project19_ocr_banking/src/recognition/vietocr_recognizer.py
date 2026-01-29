"""
Text Recognition using VietOCR (Optimized for Vietnamese)
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging
from PIL import Image

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
except ImportError:
    Predictor = None
    Cfg = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietOCRRecognizer:
    """VietOCR text recognizer for Vietnamese documents"""
    
    def __init__(self,
                 config_name: str = 'vgg_transformer',
                 device: str = 'cpu'):
        """
        Initialize VietOCR recognizer
        
        Args:
            config_name: Model configuration ('vgg_transformer', 'vgg_seq2seq', 'resnet_transformer')
            device: Device to run on ('cpu' or 'cuda')
        """
        if Predictor is None or Cfg is None:
            raise ImportError("vietocr not installed. Run: pip install vietocr")
        
        self.device = device
        self.config_name = config_name
        
        logger.info(f"Initializing VietOCR with config: {config_name} on {device}")
        
        try:
            # Initialize config
            config = Cfg.load_config_from_name(config_name)
            config['device'] = device
            config['predictor']['beamsearch'] = False  # Faster inference
            
            # Initialize predictor
            self.predictor = Predictor(config)
            
            logger.info("VietOCR recognizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VietOCR: {e}")
            raise
    
    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize text from image
        
        Args:
            image: Input image (cropped text region)
            
        Returns:
            Recognized text
        """
        try:
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Recognize
            text = self.predictor.predict(pil_image)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            return ""
    
    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Recognize text from multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of recognized texts
        """
        results = []
        
        for image in images:
            text = self.recognize(image)
            results.append(text)
        
        return results
    
    def recognize_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text with confidence score
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (text, confidence)
        """
        try:
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Note: VietOCR doesn't provide confidence scores by default
            text = self.predictor.predict(pil_image)
            
            # Estimate confidence based on text length and characters
            confidence = self._estimate_confidence(text)
            
            return text.strip(), confidence
            
        except Exception as e:
            logger.error(f"Text recognition with confidence failed: {e}")
            return "", 0.0
    
    @staticmethod
    def _estimate_confidence(text: str) -> float:
        """
        Estimate confidence score based on text characteristics
        
        Args:
            text: Recognized text
            
        Returns:
            Estimated confidence (0-1)
        """
        if not text:
            return 0.0
        
        # Simple heuristic: longer text with alphanumeric chars = higher confidence
        alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)
        
        # Penalize very short text
        length_factor = min(len(text) / 10, 1.0)
        
        confidence = alphanumeric_ratio * 0.7 + length_factor * 0.3
        
        return min(confidence, 1.0)


class PaddleOCRRecognizer:
    """PaddleOCR text recognizer"""
    
    def __init__(self, lang: str = 'vi', use_gpu: bool = False):
        """
        Initialize PaddleOCR recognizer
        
        Args:
            lang: Language code ('vi', 'en', 'ch')
            use_gpu: Use GPU if available
        """
        try:
            from paddleocr import PaddleOCR
            
            logger.info(f"Initializing PaddleOCR for language: {lang}")
            
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
            
            logger.info("PaddleOCR recognizer initialized successfully")
            
        except ImportError:
            raise ImportError("paddleocr not installed. Run: pip install paddleocr")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize text from image
        
        Args:
            image: Input image
            
        Returns:
            Recognized text
        """
        try:
            result = self.ocr.ocr(image, det=False, rec=True, cls=True)
            
            if result and result[0] and len(result[0]) > 0:
                # Extract text from result
                texts = []
                for line in result[0]:
                    if isinstance(line, tuple) and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        texts.append(text)
                
                return ' '.join(texts)
            
            return ""
            
        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {e}")
            return ""
    
    def recognize_batch(self, images: List[np.ndarray]) -> List[str]:
        """
        Recognize text from multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of recognized texts
        """
        results = []
        
        for image in images:
            text = self.recognize(image)
            results.append(text)
        
        return results


class EasyOCRRecognizer:
    """EasyOCR text recognizer"""
    
    def __init__(self, languages: List[str] = ['vi', 'en'], gpu: bool = False):
        """
        Initialize EasyOCR recognizer
        
        Args:
            languages: List of language codes
            gpu: Use GPU if available
        """
        try:
            import easyocr
            
            logger.info(f"Initializing EasyOCR for languages: {languages}")
            
            self.reader = easyocr.Reader(languages, gpu=gpu)
            
            logger.info("EasyOCR recognizer initialized successfully")
            
        except ImportError:
            raise ImportError("easyocr not installed. Run: pip install easyocr")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def recognize(self, image: np.ndarray) -> str:
        """
        Recognize text from image
        
        Args:
            image: Input image
            
        Returns:
            Recognized text
        """
        try:
            results = self.reader.readtext(image, detail=0)
            return ' '.join(results)
            
        except Exception as e:
            logger.error(f"EasyOCR recognition failed: {e}")
            return ""
    
    def recognize_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize text with confidence
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (text, confidence)
        """
        try:
            results = self.reader.readtext(image, detail=1)
            
            if not results:
                return "", 0.0
            
            texts = []
            confidences = []
            
            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)
            
            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return combined_text, avg_confidence
            
        except Exception as e:
            logger.error(f"EasyOCR recognition with confidence failed: {e}")
            return "", 0.0


def create_recognizer(recognizer_type: str = "vietocr", **kwargs):
    """
    Factory function to create text recognizer
    
    Args:
        recognizer_type: Type of recognizer ('vietocr', 'paddleocr', 'easyocr')
        **kwargs: Additional arguments for recognizer
        
    Returns:
        Text recognizer instance
    """
    if recognizer_type.lower() == "vietocr":
        return VietOCRRecognizer(**kwargs)
    elif recognizer_type.lower() == "paddleocr":
        return PaddleOCRRecognizer(**kwargs)
    elif recognizer_type.lower() == "easyocr":
        return EasyOCRRecognizer(**kwargs)
    else:
        raise ValueError(f"Unknown recognizer type: {recognizer_type}")


if __name__ == "__main__":
    # Test recognition
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vietocr_recognizer.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    # Test VietOCR
    print("Testing VietOCR recognizer...")
    recognizer = VietOCRRecognizer(device='cpu')
    text = recognizer.recognize(image)
    
    print(f"Recognized text: {text}")
