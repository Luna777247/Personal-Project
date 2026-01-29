"""
Face Detection Module using RetinaFace and InsightFace
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available. Install with: pip install insightface")

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detection using InsightFace (includes RetinaFace)
    """
    
    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        ctx_id: int = 0,  # GPU device ID, -1 for CPU
    ):
        """
        Initialize face detector
        
        Args:
            model_pack: Model pack name (buffalo_l, buffalo_m, buffalo_s)
            det_size: Detection input size
            det_thresh: Detection confidence threshold
            ctx_id: Device ID (0 for GPU, -1 for CPU)
        """
        self.model_pack = model_pack
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.ctx_id = ctx_id
        
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is required. Install with: pip install insightface")
        
        # Initialize FaceAnalysis app
        self.app = FaceAnalysis(
            name=model_pack,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        
        logger.info(f"FaceDetector initialized with {model_pack}")
    
    def detect_faces(
        self,
        image: np.ndarray,
        max_num: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR format)
            max_num: Maximum number of faces to detect (0 for all)
        
        Returns:
            List of face detections with bbox, kps, det_score, embedding
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Detect faces
        faces = self.app.get(image_rgb, max_num=max_num)
        
        # Convert to dict format
        results = []
        for face in faces:
            result = {
                'bbox': face.bbox.astype(int).tolist(),  # [x1, y1, x2, y2]
                'kps': face.kps.astype(int).tolist() if hasattr(face, 'kps') else None,  # 5 keypoints
                'det_score': float(face.det_score),
                'landmark': face.kps.tolist() if hasattr(face, 'kps') else None,
                'embedding': face.embedding if hasattr(face, 'embedding') else None,
                'age': int(face.age) if hasattr(face, 'age') else None,
                'gender': int(face.gender) if hasattr(face, 'gender') else None,
            }
            results.append(result)
        
        logger.debug(f"Detected {len(results)} faces")
        return results
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Get the largest face in image
        
        Args:
            image: Input image
        
        Returns:
            Largest face detection or None
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Find largest face by bbox area
        largest_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        return largest_face
    
    def draw_faces(
        self,
        image: np.ndarray,
        faces: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on image
        
        Args:
            image: Input image
            faces: List of face detections
            color: Box color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with drawn faces
        """
        result = image.copy()
        
        for face in faces:
            bbox = face['bbox']
            det_score = face['det_score']
            
            # Draw bounding box
            cv2.rectangle(
                result,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                thickness
            )
            
            # Draw confidence score
            label = f"{det_score:.2f}"
            cv2.putText(
                result,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )
            
            # Draw landmarks if available
            if face['landmark'] is not None:
                landmarks = np.array(face['landmark'], dtype=np.int32)
                for point in landmarks:
                    cv2.circle(result, tuple(point), 2, (0, 0, 255), -1)
        
        return result
    
    def align_face(
        self,
        image: np.ndarray,
        face: Dict[str, Any],
        output_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """
        Align face using landmarks
        
        Args:
            image: Input image
            face: Face detection with landmarks
            output_size: Output face size
        
        Returns:
            Aligned face image or None
        """
        if face['landmark'] is None:
            logger.warning("No landmarks available for alignment")
            return None
        
        # Get facial landmarks
        landmarks = np.array(face['landmark'], dtype=np.float32)
        
        # Standard landmarks for alignment (5 points)
        # Left eye, right eye, nose, left mouth, right mouth
        src = landmarks[:5]
        
        # Reference landmarks for output size
        dst = np.array([
            [30.2946, 51.6963],  # Left eye
            [65.5318, 51.5014],  # Right eye
            [48.0252, 71.7366],  # Nose
            [33.5493, 92.3655],  # Left mouth
            [62.7299, 92.2041]   # Right mouth
        ], dtype=np.float32)
        
        # Scale reference landmarks to output size
        if output_size != (112, 112):
            scale = output_size[0] / 112.0
            dst = dst * scale
        
        # Estimate similarity transform
        tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        
        # Apply transformation
        aligned_face = cv2.warpAffine(
            image,
            tform,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return aligned_face
    
    def extract_face(
        self,
        image: np.ndarray,
        bbox: List[int],
        margin: float = 0.2
    ) -> np.ndarray:
        """
        Extract face region with margin
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            margin: Margin ratio to add around bbox
        
        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Add margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(image.shape[1], x2 + margin_w)
        y2 = min(image.shape[0], y2 + margin_h)
        
        face_crop = image[y1:y2, x1:x2]
        return face_crop


class RetinaFaceDetector:
    """
    Standalone RetinaFace detector (using ONNX model)
    Note: InsightFace already includes RetinaFace, this is for standalone usage
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        """
        Initialize RetinaFace detector with ONNX model
        
        Args:
            model_path: Path to ONNX model
            input_size: Input image size
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        """
        import onnxruntime as ort
        
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load ONNX model
        if Path(model_path).exists():
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"RetinaFace model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for RetinaFace"""
        # Resize
        img = cv2.resize(image, self.input_size)
        
        # Normalize
        img = (img - 127.5) / 128.0
        
        # Transpose to CHW format
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using RetinaFace ONNX model
        
        Args:
            image: Input image (BGR)
        
        Returns:
            List of face detections
        """
        # Preprocess
        input_blob = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: input_blob})
        
        # Postprocess (simplified - actual implementation would be more complex)
        # This is a placeholder - full RetinaFace postprocessing is complex
        faces = []
        logger.warning("RetinaFace ONNX postprocessing not fully implemented. Use InsightFace instead.")
        
        return faces


def test_face_detector():
    """Test face detector"""
    # Create test image with face
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detector
    detector = FaceDetector(model_pack="buffalo_l", ctx_id=-1)  # CPU mode
    
    # Detect faces
    faces = detector.detect_faces(test_image)
    print(f"Detected {len(faces)} faces")
    
    # Draw faces
    if faces:
        result = detector.draw_faces(test_image, faces)
        cv2.imwrite("test_detection.jpg", result)
        print("Detection result saved to test_detection.jpg")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_face_detector()
