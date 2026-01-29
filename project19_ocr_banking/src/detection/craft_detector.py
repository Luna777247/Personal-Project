"""
Text Detection using CRAFT (Character Region Awareness For Text detection)
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
import logging
from pathlib import Path

try:
    from craft_text_detector import Craft
except ImportError:
    Craft = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CRAFTDetector:
    """CRAFT text detector for document images"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 text_threshold: float = 0.7,
                 link_threshold: float = 0.4,
                 low_text: float = 0.4,
                 cuda: bool = True,
                 poly: bool = False):
        """
        Initialize CRAFT detector
        
        Args:
            model_path: Path to pretrained model
            text_threshold: Text confidence threshold
            link_threshold: Link confidence threshold
            low_text: Low text threshold
            cuda: Use GPU if available
            poly: Return polygon boxes instead of rectangles
        """
        if Craft is None:
            raise ImportError("craft-text-detector not installed. Run: pip install craft-text-detector")
        
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.poly = poly
        
        # Initialize CRAFT
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Initializing CRAFT detector on {self.device}")
        
        try:
            self.detector = Craft(
                output_dir=None,
                crop_type="poly" if poly else "box",
                cuda=cuda,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text
            )
            logger.info("CRAFT detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CRAFT: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect text regions in image
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2, x3, y3, x4, y4] for polygons
            or [x_min, y_min, x_max, y_max] for rectangles
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect text
            prediction_result = self.detector.detect_text(image)
            
            boxes = []
            if prediction_result and 'boxes' in prediction_result:
                boxes = prediction_result['boxes']
            
            logger.info(f"Detected {len(boxes)} text regions")
            return boxes
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []
    
    def detect_with_confidence(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect text regions with confidence scores
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (boxes, confidence_scores)
        """
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            prediction_result = self.detector.detect_text(image)
            
            boxes = []
            scores = []
            
            if prediction_result:
                if 'boxes' in prediction_result:
                    boxes = prediction_result['boxes']
                if 'scores' in prediction_result:
                    scores = prediction_result['scores']
                else:
                    scores = [1.0] * len(boxes)  # Default confidence
            
            return boxes, scores
            
        except Exception as e:
            logger.error(f"Text detection with confidence failed: {e}")
            return [], []
    
    def visualize(self, image: np.ndarray, boxes: List[np.ndarray]) -> np.ndarray:
        """
        Visualize detected text regions
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            
        Returns:
            Image with drawn boxes
        """
        vis_image = image.copy()
        
        for box in boxes:
            box = np.array(box, dtype=np.int32)
            
            if len(box) == 4:  # Rectangle [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            elif len(box) == 8:  # Polygon [x1, y1, x2, y2, x3, y3, x4, y4]
                pts = box.reshape((-1, 2))
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 2)
        
        return vis_image


class DBNetDetector:
    """DBNet (Differentiable Binarization) text detector"""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 threshold: float = 0.3,
                 box_threshold: float = 0.5,
                 max_candidates: int = 1000,
                 unclip_ratio: float = 1.5,
                 use_dilate: bool = False):
        """
        Initialize DBNet detector
        
        Args:
            model_path: Path to pretrained model
            threshold: Binary threshold
            box_threshold: Box confidence threshold
            max_candidates: Maximum number of box candidates
            unclip_ratio: Unclip ratio for expanding boxes
            use_dilate: Use dilation for post-processing
        """
        self.threshold = threshold
        self.box_threshold = box_threshold
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.use_dilate = use_dilate
        
        logger.info("DBNet detector initialized (placeholder)")
        logger.warning("DBNet implementation requires custom setup. Using PaddleOCR's DBNet as alternative.")
    
    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect text regions using DBNet
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes
        """
        logger.info("DBNet detection - using PaddleOCR implementation")
        
        try:
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='vi',
                det=True,
                rec=False,
                show_log=False
            )
            
            # Detect only (no recognition)
            result = ocr.ocr(image, det=True, rec=False, cls=False)
            
            boxes = []
            if result and result[0]:
                for line in result[0]:
                    box = np.array(line).reshape(-1).tolist()
                    boxes.append(box)
            
            logger.info(f"DBNet detected {len(boxes)} text regions")
            return boxes
            
        except Exception as e:
            logger.error(f"DBNet detection failed: {e}")
            return []


def create_detector(detector_type: str = "craft", **kwargs):
    """
    Factory function to create text detector
    
    Args:
        detector_type: Type of detector ('craft', 'dbnet')
        **kwargs: Additional arguments for detector
        
    Returns:
        Text detector instance
    """
    if detector_type.lower() == "craft":
        return CRAFTDetector(**kwargs)
    elif detector_type.lower() == "dbnet":
        return DBNetDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


if __name__ == "__main__":
    # Test detection
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python craft_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    # Test CRAFT
    print("Testing CRAFT detector...")
    detector = CRAFTDetector(cuda=True)
    boxes = detector.detect(image)
    
    print(f"Detected {len(boxes)} text regions")
    
    # Visualize
    vis_image = detector.visualize(image, boxes)
    output_path = "output_craft.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"Visualization saved to {output_path}")
