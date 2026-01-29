"""
Text Detection Base Class and Utilities
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTextDetector(ABC):
    """Abstract base class for text detectors"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect text regions in image"""
        pass
    
    @abstractmethod
    def detect_with_confidence(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Detect text regions with confidence scores"""
        pass
    
    def sort_boxes(self, boxes: List[np.ndarray], method: str = 'top-to-bottom') -> List[np.ndarray]:
        """
        Sort detected boxes
        
        Args:
            boxes: List of bounding boxes
            method: Sorting method ('top-to-bottom', 'left-to-right')
            
        Returns:
            Sorted boxes
        """
        if not boxes:
            return boxes
        
        if method == 'top-to-bottom':
            # Sort by y coordinate
            boxes = sorted(boxes, key=lambda x: self._get_box_center(x)[1])
        elif method == 'left-to-right':
            # Sort by x coordinate
            boxes = sorted(boxes, key=lambda x: self._get_box_center(x)[0])
        
        return boxes
    
    def group_boxes_by_line(self, boxes: List[np.ndarray], threshold: float = 20) -> List[List[np.ndarray]]:
        """
        Group boxes into lines
        
        Args:
            boxes: List of bounding boxes
            threshold: Y-coordinate threshold for grouping
            
        Returns:
            List of box groups (lines)
        """
        if not boxes:
            return []
        
        # Sort by y coordinate
        sorted_boxes = self.sort_boxes(boxes, method='top-to-bottom')
        
        lines = []
        current_line = [sorted_boxes[0]]
        current_y = self._get_box_center(sorted_boxes[0])[1]
        
        for box in sorted_boxes[1:]:
            box_y = self._get_box_center(box)[1]
            
            if abs(box_y - current_y) <= threshold:
                current_line.append(box)
            else:
                # Sort current line by x coordinate
                current_line = sorted(current_line, key=lambda x: self._get_box_center(x)[0])
                lines.append(current_line)
                current_line = [box]
                current_y = box_y
        
        # Add last line
        if current_line:
            current_line = sorted(current_line, key=lambda x: self._get_box_center(x)[0])
            lines.append(current_line)
        
        return lines
    
    def merge_boxes(self, boxes: List[np.ndarray], horizontal: bool = True, threshold: float = 20) -> List[np.ndarray]:
        """
        Merge nearby boxes
        
        Args:
            boxes: List of bounding boxes
            horizontal: Merge horizontally if True, vertically if False
            threshold: Distance threshold for merging
            
        Returns:
            Merged boxes
        """
        if not boxes or len(boxes) < 2:
            return boxes
        
        merged = []
        sorted_boxes = self.sort_boxes(boxes, method='left-to-right' if horizontal else 'top-to-bottom')
        
        current_box = sorted_boxes[0]
        
        for box in sorted_boxes[1:]:
            if self._should_merge(current_box, box, horizontal, threshold):
                current_box = self._merge_two_boxes(current_box, box)
            else:
                merged.append(current_box)
                current_box = box
        
        merged.append(current_box)
        return merged
    
    @staticmethod
    def _get_box_center(box: np.ndarray) -> Tuple[float, float]:
        """Get center point of box"""
        if len(box) == 4:  # [x_min, y_min, x_max, y_max]
            return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        elif len(box) == 8:  # [x1, y1, x2, y2, x3, y3, x4, y4]
            x = (box[0] + box[2] + box[4] + box[6]) / 4
            y = (box[1] + box[3] + box[5] + box[7]) / 4
            return (x, y)
        else:
            raise ValueError(f"Invalid box format: {box}")
    
    @staticmethod
    def _get_box_bounds(box: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding rectangle of box"""
        if len(box) == 4:
            return tuple(map(int, box))
        elif len(box) == 8:
            xs = [box[0], box[2], box[4], box[6]]
            ys = [box[1], box[3], box[5], box[7]]
            return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        else:
            raise ValueError(f"Invalid box format: {box}")
    
    def _should_merge(self, box1: np.ndarray, box2: np.ndarray, horizontal: bool, threshold: float) -> bool:
        """Check if two boxes should be merged"""
        bounds1 = self._get_box_bounds(box1)
        bounds2 = self._get_box_bounds(box2)
        
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        if horizontal:
            # Check vertical overlap and horizontal distance
            v_overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
            h_dist = abs(x2_min - x1_max)
            return v_overlap > 0 and h_dist <= threshold
        else:
            # Check horizontal overlap and vertical distance
            h_overlap = min(x1_max, x2_max) - max(x1_min, x2_min)
            v_dist = abs(y2_min - y1_max)
            return h_overlap > 0 and v_dist <= threshold
    
    def _merge_two_boxes(self, box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """Merge two boxes into one"""
        bounds1 = self._get_box_bounds(box1)
        bounds2 = self._get_box_bounds(box2)
        
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        merged = np.array([
            min(x1_min, x2_min),
            min(y1_min, y2_min),
            max(x1_max, x2_max),
            max(y1_max, y2_max)
        ])
        
        return merged
    
    def crop_text_regions(self, image: np.ndarray, boxes: List[np.ndarray], 
                         padding: int = 5) -> List[np.ndarray]:
        """
        Crop text regions from image
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            padding: Padding around boxes
            
        Returns:
            List of cropped images
        """
        crops = []
        h, w = image.shape[:2]
        
        for box in boxes:
            x_min, y_min, x_max, y_max = self._get_box_bounds(box)
            
            # Add padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            # Crop
            crop = image[y_min:y_max, x_min:x_max]
            crops.append(crop)
        
        return crops


def filter_boxes_by_size(boxes: List[np.ndarray], 
                         min_width: int = 10,
                         min_height: int = 10,
                         max_width: Optional[int] = None,
                         max_height: Optional[int] = None) -> List[np.ndarray]:
    """
    Filter boxes by size
    
    Args:
        boxes: List of bounding boxes
        min_width: Minimum width
        min_height: Minimum height
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Filtered boxes
    """
    filtered = []
    
    for box in boxes:
        if len(box) == 4:
            w = box[2] - box[0]
            h = box[3] - box[1]
        elif len(box) == 8:
            xs = [box[0], box[2], box[4], box[6]]
            ys = [box[1], box[3], box[5], box[7]]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
        else:
            continue
        
        if w < min_width or h < min_height:
            continue
        
        if max_width and w > max_width:
            continue
        
        if max_height and h > max_height:
            continue
        
        filtered.append(box)
    
    return filtered


def filter_boxes_by_confidence(boxes: List[np.ndarray], 
                               scores: List[float],
                               threshold: float = 0.5) -> Tuple[List[np.ndarray], List[float]]:
    """
    Filter boxes by confidence score
    
    Args:
        boxes: List of bounding boxes
        scores: Confidence scores
        threshold: Minimum confidence threshold
        
    Returns:
        Filtered boxes and scores
    """
    filtered_boxes = []
    filtered_scores = []
    
    for box, score in zip(boxes, scores):
        if score >= threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
    
    return filtered_boxes, filtered_scores


if __name__ == "__main__":
    # Test utilities
    boxes = [
        np.array([10, 10, 100, 30]),
        np.array([110, 10, 200, 30]),
        np.array([10, 50, 100, 70]),
    ]
    
    print("Original boxes:", boxes)
    
    # Test sorting
    detector = BaseTextDetector()
    sorted_boxes = detector.sort_boxes(boxes, method='top-to-bottom')
    print("Sorted boxes:", sorted_boxes)
    
    # Test grouping
    lines = detector.group_boxes_by_line(boxes, threshold=30)
    print(f"Grouped into {len(lines)} lines")
