"""
Liveness Detection Module
Detects if the face is from a real person or a spoofed image/video
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from scipy.spatial import distance as dist

logger = logging.getLogger(__name__)


class BlinkDetector:
    """
    Blink detection using Eye Aspect Ratio (EAR)
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.21,
        consecutive_frames: int = 3,
        min_blinks: int = 1,
        max_blinks: int = 10
    ):
        """
        Initialize blink detector
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold (below = closed)
            consecutive_frames: Number of consecutive frames for blink
            min_blinks: Minimum blinks required
            max_blinks: Maximum blinks allowed
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.min_blinks = min_blinks
        self.max_blinks = max_blinks
        
        self.blink_counter = 0
        self.frame_counter = 0
        
        # Eye landmark indices (for dlib 68-point landmarks)
        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 42
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 48
        
        logger.info("BlinkDetector initialized")
    
    def eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio
        
        Args:
            eye_landmarks: Eye landmark points (6 points)
        
        Returns:
            EAR value
        """
        # Compute vertical distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute horizontal distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(
        self,
        landmarks: np.ndarray
    ) -> Tuple[bool, float, int]:
        """
        Detect blink in single frame
        
        Args:
            landmarks: Facial landmarks (68 points)
        
        Returns:
            (is_blinking, ear_value, total_blinks)
        """
        # Extract eye landmarks
        left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
        right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
        
        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        
        # Check if eye is closed
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            # Eye was closed for sufficient frames
            if self.frame_counter >= self.consecutive_frames:
                self.blink_counter += 1
            self.frame_counter = 0
        
        is_blinking = ear < self.ear_threshold
        
        return is_blinking, ear, self.blink_counter
    
    def verify_liveness(self) -> Tuple[bool, Dict]:
        """
        Verify liveness based on blink count
        
        Returns:
            (is_live, details)
        """
        is_live = self.min_blinks <= self.blink_counter <= self.max_blinks
        
        details = {
            "blink_count": self.blink_counter,
            "min_required": self.min_blinks,
            "max_allowed": self.max_blinks,
            "passed": is_live
        }
        
        logger.debug(f"Blink verification: {details}")
        return is_live, details
    
    def reset(self):
        """Reset blink counter"""
        self.blink_counter = 0
        self.frame_counter = 0


class HeadMovementDetector:
    """
    Head movement detection using pose estimation
    """
    
    def __init__(
        self,
        yaw_threshold: float = 15.0,
        pitch_threshold: float = 10.0,
        roll_threshold: float = 10.0,
        min_movements: int = 2
    ):
        """
        Initialize head movement detector
        
        Args:
            yaw_threshold: Yaw angle threshold (degrees)
            pitch_threshold: Pitch angle threshold
            roll_threshold: Roll angle threshold
            min_movements: Minimum significant movements required
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.min_movements = min_movements
        
        self.initial_pose = None
        self.movement_count = 0
        self.pose_history = []
        
        logger.info("HeadMovementDetector initialized")
    
    def estimate_head_pose(
        self,
        landmarks: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll) from landmarks
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image (height, width)
        
        Returns:
            (yaw, pitch, roll) in degrees
        """
        height, width = image_shape[:2]
        
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ])
        
        # Camera matrix
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4, 1))
        
        # 2D image points from landmarks
        # Using indices: 30(nose), 8(chin), 36(left eye), 45(right eye), 48(left mouth), 54(right mouth)
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left mouth corner
            landmarks[54]   # Right mouth corner
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        # Simplified conversion
        pitch = np.arcsin(-rotation_matrix[2][0]) * 180 / np.pi
        yaw = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * 180 / np.pi
        roll = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * 180 / np.pi
        
        return yaw, pitch, roll
    
    def detect_movement(
        self,
        landmarks: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[bool, Dict]:
        """
        Detect head movement
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image shape
        
        Returns:
            (has_moved, movement_info)
        """
        # Estimate current pose
        yaw, pitch, roll = self.estimate_head_pose(landmarks, image_shape)
        
        current_pose = {"yaw": yaw, "pitch": pitch, "roll": roll}
        self.pose_history.append(current_pose)
        
        # Set initial pose
        if self.initial_pose is None:
            self.initial_pose = current_pose
            return False, current_pose
        
        # Calculate differences
        yaw_diff = abs(yaw - self.initial_pose["yaw"])
        pitch_diff = abs(pitch - self.initial_pose["pitch"])
        roll_diff = abs(roll - self.initial_pose["roll"])
        
        # Check if movement exceeds threshold
        has_moved = (
            yaw_diff > self.yaw_threshold or
            pitch_diff > self.pitch_threshold or
            roll_diff > self.roll_threshold
        )
        
        if has_moved:
            self.movement_count += 1
        
        movement_info = {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "yaw_diff": yaw_diff,
            "pitch_diff": pitch_diff,
            "roll_diff": roll_diff,
            "has_moved": has_moved,
            "movement_count": self.movement_count
        }
        
        return has_moved, movement_info
    
    def verify_liveness(self) -> Tuple[bool, Dict]:
        """
        Verify liveness based on head movements
        
        Returns:
            (is_live, details)
        """
        is_live = self.movement_count >= self.min_movements
        
        details = {
            "movement_count": self.movement_count,
            "min_required": self.min_movements,
            "passed": is_live
        }
        
        logger.debug(f"Head movement verification: {details}")
        return is_live, details
    
    def reset(self):
        """Reset movement detector"""
        self.initial_pose = None
        self.movement_count = 0
        self.pose_history = []


class TextureAnalyzer:
    """
    Texture analysis for anti-spoofing
    Detects printed photos, screen displays using texture features
    """
    
    def __init__(
        self,
        lbp_threshold: float = 0.5,
        blur_threshold: float = 100.0,
        color_diversity_threshold: float = 10.0
    ):
        """
        Initialize texture analyzer
        
        Args:
            lbp_threshold: LBP uniformity threshold
            blur_threshold: Laplacian variance threshold
            color_diversity_threshold: Color diversity threshold
        """
        self.lbp_threshold = lbp_threshold
        self.blur_threshold = blur_threshold
        self.color_diversity_threshold = color_diversity_threshold
        
        logger.info("TextureAnalyzer initialized")
    
    def compute_lbp(self, image: np.ndarray, radius: int = 1) -> np.ndarray:
        """
        Compute Local Binary Pattern
        
        Args:
            image: Grayscale image
            radius: LBP radius
        
        Returns:
            LBP histogram
        """
        from skimage.feature import local_binary_pattern
        
        # Compute LBP
        lbp = local_binary_pattern(image, 8 * radius, radius, method='uniform')
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        # Normalize
        hist = hist.astype(float) / hist.sum()
        
        return hist
    
    def detect_blur(self, image: np.ndarray) -> float:
        """
        Detect blur using Laplacian variance
        
        Args:
            image: Grayscale image
        
        Returns:
            Blur score (higher = sharper)
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def analyze_color_diversity(self, image: np.ndarray) -> float:
        """
        Analyze color diversity (spoofed images often have limited colors)
        
        Args:
            image: Color image (BGR)
        
        Returns:
            Color diversity score
        """
        # Convert to different color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate standard deviation of each channel
        std_h = np.std(hsv[:, :, 0])
        std_s = np.std(hsv[:, :, 1])
        std_v = np.std(hsv[:, :, 2])
        
        # Average diversity
        diversity = (std_h + std_s + std_v) / 3.0
        
        return diversity
    
    def analyze_fourier(self, image: np.ndarray) -> float:
        """
        Analyze frequency domain (printed/screen images have different spectrum)
        
        Args:
            image: Grayscale image
        
        Returns:
            Fourier feature score
        """
        # Apply FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Calculate high frequency energy
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask for high frequencies
        mask = np.ones((rows, cols), dtype=np.uint8)
        r = 30
        cv2.circle(mask, (ccol, crow), r, 0, -1)
        
        # Calculate energy ratio
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        
        ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        return ratio
    
    def analyze_texture(
        self,
        face_image: np.ndarray
    ) -> Tuple[bool, Dict]:
        """
        Analyze texture features for liveness detection
        
        Args:
            face_image: Face crop (BGR)
        
        Returns:
            (is_live, analysis_details)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Compute features
        blur_score = self.detect_blur(gray)
        color_diversity = self.analyze_color_diversity(face_image)
        fourier_score = self.analyze_fourier(gray)
        
        # Compute LBP
        lbp_hist = self.compute_lbp(gray)
        lbp_uniformity = np.max(lbp_hist)  # High uniformity = likely spoofed
        
        # Determine if live
        is_blur_ok = blur_score > self.blur_threshold
        is_color_ok = color_diversity > self.color_diversity_threshold
        is_lbp_ok = lbp_uniformity < self.lbp_threshold
        is_fourier_ok = fourier_score > 0.3
        
        # Combine checks (at least 3 out of 4 should pass)
        passed_checks = sum([is_blur_ok, is_color_ok, is_lbp_ok, is_fourier_ok])
        is_live = passed_checks >= 3
        
        details = {
            "blur_score": float(blur_score),
            "blur_passed": is_blur_ok,
            "color_diversity": float(color_diversity),
            "color_passed": is_color_ok,
            "lbp_uniformity": float(lbp_uniformity),
            "lbp_passed": is_lbp_ok,
            "fourier_score": float(fourier_score),
            "fourier_passed": is_fourier_ok,
            "passed_checks": passed_checks,
            "is_live": is_live
        }
        
        logger.debug(f"Texture analysis: {details}")
        return is_live, details


class LivenessDetector:
    """
    Combined liveness detection system
    """
    
    def __init__(
        self,
        enable_blink: bool = True,
        enable_head_movement: bool = True,
        enable_texture: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize liveness detector
        
        Args:
            enable_blink: Enable blink detection
            enable_head_movement: Enable head movement detection
            enable_texture: Enable texture analysis
            config: Configuration dict
        """
        self.enable_blink = enable_blink
        self.enable_head_movement = enable_head_movement
        self.enable_texture = enable_texture
        
        config = config or {}
        
        # Initialize detectors
        if enable_blink:
            blink_cfg = config.get("blink", {})
            self.blink_detector = BlinkDetector(**blink_cfg)
        
        if enable_head_movement:
            head_cfg = config.get("head_movement", {})
            self.head_detector = HeadMovementDetector(**head_cfg)
        
        if enable_texture:
            texture_cfg = config.get("texture", {})
            self.texture_analyzer = TextureAnalyzer(**texture_cfg)
        
        logger.info("LivenessDetector initialized")
    
    def detect(
        self,
        image: np.ndarray,
        face_bbox: Optional[List[int]] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[bool, Dict]:
        """
        Detect liveness
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            landmarks: Facial landmarks (68 points)
        
        Returns:
            (is_live, details)
        """
        results = {}
        checks_passed = 0
        total_checks = 0
        
        # Blink detection
        if self.enable_blink and landmarks is not None:
            total_checks += 1
            is_blinking, ear, blink_count = self.blink_detector.detect_blink(landmarks)
            is_live_blink, blink_details = self.blink_detector.verify_liveness()
            results["blink"] = blink_details
            if is_live_blink:
                checks_passed += 1
        
        # Head movement detection
        if self.enable_head_movement and landmarks is not None:
            total_checks += 1
            has_moved, movement_info = self.head_detector.detect_movement(landmarks, image.shape)
            is_live_head, head_details = self.head_detector.verify_liveness()
            results["head_movement"] = head_details
            if is_live_head:
                checks_passed += 1
        
        # Texture analysis
        if self.enable_texture:
            total_checks += 1
            # Extract face region
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                face_crop = image[y1:y2, x1:x2]
            else:
                face_crop = image
            
            is_live_texture, texture_details = self.texture_analyzer.analyze_texture(face_crop)
            results["texture"] = texture_details
            if is_live_texture:
                checks_passed += 1
        
        # Overall liveness (majority voting)
        is_live = checks_passed >= (total_checks // 2 + 1)
        
        results["overall"] = {
            "is_live": is_live,
            "checks_passed": checks_passed,
            "total_checks": total_checks,
            "confidence": checks_passed / total_checks if total_checks > 0 else 0
        }
        
        logger.info(f"Liveness detection: {is_live} ({checks_passed}/{total_checks} checks passed)")
        return is_live, results
    
    def reset(self):
        """Reset all detectors"""
        if self.enable_blink:
            self.blink_detector.reset()
        if self.enable_head_movement:
            self.head_detector.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test texture analyzer
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    analyzer = TextureAnalyzer()
    is_live, details = analyzer.analyze_texture(test_image)
    print(f"Liveness: {is_live}")
    print(f"Details: {details}")
