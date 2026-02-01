"""
Preprocess sample images for better OCR accuracy
"""
import cv2
import numpy as np
from pathlib import Path
import argparse


def preprocess_image(image_path, output_path=None, operations=None):
    """
    Preprocess image for OCR
    
    Args:
        image_path: Path to input image
        output_path: Path to save output (optional)
        operations: List of operations to apply
    """
    if operations is None:
        operations = ['denoise', 'contrast', 'threshold']
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f"Processing: {image_path}")
    print(f"Original size: {img.shape}")
    
    processed = img.copy()
    
    # Apply operations
    if 'grayscale' in operations:
        print("  - Converting to grayscale")
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    if 'denoise' in operations:
        print("  - Applying denoising")
        if len(processed.shape) == 3:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        else:
            processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
    
    if 'contrast' in operations:
        print("  - Enhancing contrast")
        if len(processed.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        else:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    
    if 'threshold' in operations:
        print("  - Applying adaptive threshold")
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            processed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    
    if 'deskew' in operations:
        print("  - Deskewing image")
        processed = deskew_image(processed)
    
    # Save output
    if output_path:
        cv2.imwrite(str(output_path), processed)
        print(f"Saved to: {output_path}")
    
    return processed


def deskew_image(image):
    """Deskew image using Hough transform"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None and len(lines) > 0:
        # Calculate average angle
        angles = []
        for line in lines[:10]:  # Use first 10 lines
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        
        avg_angle = np.median(angles)
        
        # Rotate image
        if abs(avg_angle) > 0.5:  # Only rotate if angle is significant
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    
    return image


def batch_preprocess(input_dir, output_dir, operations=None):
    """
    Batch preprocess images in directory
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        operations: List of operations to apply
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for img_file in image_files:
        output_file = output_path / f"processed_{img_file.name}"
        
        try:
            preprocess_image(img_file, output_file, operations)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"\nProcessed {len(image_files)} images")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for OCR")
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('-o', '--output', help='Output image or directory')
    parser.add_argument('-ops', '--operations', nargs='+', 
                       choices=['grayscale', 'denoise', 'contrast', 'threshold', 'deskew'],
                       default=['denoise', 'contrast'],
                       help='Operations to apply')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='Batch process directory')
    
    args = parser.parse_args()
    
    if args.batch:
        output_dir = args.output or str(Path(args.input) / 'preprocessed')
        batch_preprocess(args.input, output_dir, args.operations)
    else:
        output_path = args.output or f"preprocessed_{Path(args.input).name}"
        result = preprocess_image(args.input, output_path, args.operations)
        
        print("\nPreprocessing complete!")
