"""
Demo script for Face Recognition eKYC
"""
import sys
import cv2
import numpy as np
import base64
import requests
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detection import FaceDetector
from embedding import FaceEmbedder
from liveness import LivenessDetector, TextureAnalyzer
from matching import FaceMatcher, CCCDProcessor


def demo_face_detection():
    """Demo face detection"""
    print("\n=== Face Detection Demo ===")
    
    # Load test image (you would load actual image here)
    image = cv2.imread("test_images/test_selfie.jpg")
    if image is None:
        print("Creating sample image...")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detector
    detector = FaceDetector(model_pack="buffalo_l", ctx_id=-1)
    
    # Detect faces
    faces = detector.detect_faces(image)
    
    print(f"Detected {len(faces)} face(s)")
    
    for i, face in enumerate(faces):
        print(f"\nFace {i+1}:")
        print(f"  Bbox: {face['bbox']}")
        print(f"  Confidence: {face['det_score']:.4f}")
        if face.get('age'):
            print(f"  Age: {face['age']}")
        if face.get('gender') is not None:
            gender = "Male" if face['gender'] == 1 else "Female"
            print(f"  Gender: {gender}")
    
    # Draw faces
    if faces:
        result = detector.draw_faces(image, faces)
        cv2.imwrite("demo_detection.jpg", result)
        print("\nResult saved to demo_detection.jpg")


def demo_face_matching():
    """Demo face matching"""
    print("\n=== Face Matching Demo ===")
    
    # Load images (or create samples)
    selfie = cv2.imread("test_images/test_selfie.jpg")
    cccd = cv2.imread("test_images/test_cccd.jpg")
    
    if selfie is None or cccd is None:
        print("Creating sample images...")
        selfie = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cccd = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # Initialize matcher
    matcher = FaceMatcher(ctx_id=-1)
    
    # Match faces
    is_match, similarity, details = matcher.match_faces(selfie, cccd)
    
    print(f"\nMatching Result:")
    print(f"  Is Match: {is_match}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Threshold: {matcher.similarity_threshold}")
    
    if details:
        print(f"\nDetails:")
        if 'error' not in details:
            print(f"  CCCD face confidence: {details['cccd_face']['confidence']:.4f}")
            print(f"  Selfie face confidence: {details['selfie_face']['confidence']:.4f}")


def demo_liveness_detection():
    """Demo liveness detection"""
    print("\n=== Liveness Detection Demo ===")
    
    # Load image
    image = cv2.imread("test_images/test_selfie.jpg")
    if image is None:
        print("Creating sample image...")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Analyze texture only (other methods need video)
    analyzer = TextureAnalyzer()
    
    # Extract face region
    from detection import FaceDetector
    detector = FaceDetector(ctx_id=-1)
    face = detector.get_largest_face(image)
    
    if face:
        bbox = face['bbox']
        face_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        # Analyze texture
        is_live, details = analyzer.analyze_texture(face_crop)
        
        print(f"\nLiveness Result:")
        print(f"  Is Live: {is_live}")
        print(f"  Blur Score: {details['blur_score']:.2f} (threshold: {analyzer.blur_threshold})")
        print(f"  Color Diversity: {details['color_diversity']:.2f} (threshold: {analyzer.color_diversity_threshold})")
        print(f"  LBP Uniformity: {details['lbp_uniformity']:.4f} (threshold: {analyzer.lbp_threshold})")
        print(f"  Fourier Score: {details['fourier_score']:.4f}")
        print(f"  Passed Checks: {details['passed_checks']}/4")
    else:
        print("No face detected for liveness check")


def demo_api_call():
    """Demo API call"""
    print("\n=== API Call Demo ===")
    
    # Check if API is running
    api_url = "http://localhost:8000"
    
    try:
        # Health check
        response = requests.get(f"{api_url}/health")
        print(f"API Status: {response.json()['status']}")
        
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Face detection
        print("\nTesting /detect endpoint...")
        response = requests.post(
            f"{api_url}/detect",
            json={
                "image": image_b64,
                "max_faces": 1,
                "return_landmarks": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Success: {result['success']}")
            print(f"  Faces detected: {result['num_faces']}")
        else:
            print(f"  Error: {response.status_code}")
        
    except requests.exceptions.ConnectionError:
        print("API not running. Start with: uvicorn api.main:app --reload")


def demo_complete_verification():
    """Demo complete verification workflow"""
    print("\n=== Complete Verification Demo ===")
    
    # Load images
    selfie = cv2.imread("test_images/test_selfie.jpg")
    cccd = cv2.imread("test_images/test_cccd.jpg")
    
    if selfie is None or cccd is None:
        print("Creating sample images...")
        selfie = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cccd = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # Initialize components
    matcher = FaceMatcher(ctx_id=-1)
    liveness_detector = LivenessDetector(
        enable_blink=False,  # Needs video
        enable_head_movement=False,  # Needs video
        enable_texture=True
    )
    
    # Complete verification
    result = matcher.verify_identity(
        selfie,
        cccd,
        liveness_check=True,
        liveness_detector=liveness_detector
    )
    
    print(f"\nVerification Result:")
    print(f"  Verified: {result['verified']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    if result['face_match']:
        print(f"\nFace Match:")
        print(f"  Similarity: {result['face_match'].get('similarity', 'N/A')}")
        print(f"  Is Match: {result['face_match'].get('is_match', 'N/A')}")
    
    if result['liveness']:
        liveness_overall = result['liveness'].get('overall', {})
        print(f"\nLiveness Check:")
        print(f"  Is Live: {liveness_overall.get('is_live', 'N/A')}")
        print(f"  Confidence: {liveness_overall.get('confidence', 0):.2f}")
    
    if result['errors']:
        print(f"\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    print("Face Recognition eKYC Demo")
    print("=" * 50)
    
    # Run demos
    try:
        demo_face_detection()
    except Exception as e:
        print(f"Face detection demo error: {e}")
    
    try:
        demo_face_matching()
    except Exception as e:
        print(f"Face matching demo error: {e}")
    
    try:
        demo_liveness_detection()
    except Exception as e:
        print(f"Liveness detection demo error: {e}")
    
    try:
        demo_complete_verification()
    except Exception as e:
        print(f"Complete verification demo error: {e}")
    
    try:
        demo_api_call()
    except Exception as e:
        print(f"API call demo error: {e}")
    
    print("\n" + "=" * 50)
    print("Demo complete!")
