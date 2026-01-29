#!/usr/bin/env python3
"""
YOLOv8 Object Detection Demo
============================

Quick demonstration of YOLOv8 object detection capabilities.

This script shows how to:
- Load a pretrained YOLOv8 model
- Run inference on sample images
- Display detection results
- Export model for deployment

Author: AI Assistant
Date: 2025
"""

import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Import our custom modules
from src.yolov8_detector import YOLOv8Detector
from src.data_preprocessing import create_sample_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_images():
    """Create sample images for demonstration"""
    logger.info("Creating demo images...")

    # Create demo directory
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)

    # Create sample dataset first
    data_yaml = create_sample_dataset("data/sample_dataset")

    # Get images from the sample dataset
    sample_images = list(Path("data/sample_dataset/test/images").glob("*.jpg"))

    if sample_images:
        # Copy a few images to demo directory
        for i, img_path in enumerate(sample_images[:3]):  # Take first 3 images
            demo_img_path = demo_dir / f"demo_{i}.jpg"
            import shutil
            shutil.copy(str(img_path), str(demo_img_path))
            logger.info(f"Created demo image: {demo_img_path}")

    return list(demo_dir.glob("*.jpg"))

def run_demo():
    """Run the YOLOv8 detection demo"""
    print("=" * 60)
    print("🚀 YOLOv8 Object Detection Demo")
    print("=" * 60)

    try:
        # Initialize detector with pretrained model
        logger.info("Initializing YOLOv8 detector...")
        detector = YOLOv8Detector(model_size='yolov8n', device='auto')

        # Load pretrained model
        logger.info("Loading pretrained YOLOv8 model...")
        detector.load_model()  # Uses pretrained yolov8n.pt

        print("✅ Model loaded successfully!")
        print(f"   Model: YOLOv8 Nano")
        print(f"   Classes: {len(detector.class_names)} object categories")
        print(f"   Device: {detector.device}")

        # Create or find demo images
        demo_images = create_demo_images()

        if not demo_images:
            logger.error("No demo images available. Please check the sample dataset creation.")
            return

        print(f"\\n📸 Found {len(demo_images)} demo images")

        # Run inference on demo images
        print("\\n🔍 Running object detection...")

        for i, img_path in enumerate(demo_images[:3], 1):  # Process up to 3 images
            print(f"\\n   Processing image {i}/{min(3, len(demo_images))}: {img_path.name}")

            # Run inference
            results = detector.predict(
                str(img_path),
                conf=0.25,
                iou=0.6,
                save=False,
                verbose=False
            )

            # Process and display results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    detections = len(boxes)
                    print(f"      📍 Detected {detections} objects")

                    # Show top detections
                    for j, box in enumerate(boxes[:5]):  # Show top 5
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_name = detector.class_names[class_id] if detector.class_names else f"class_{class_id}"

                        print(".2f")

                    # Visualize result
                    annotated_image = result.plot()
                    plt.figure(figsize=(12, 8))
                    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.title(f"Detection Results - {img_path.name}")
                    plt.tight_layout()
                    plt.show()

                    break  # Only show first result
                else:
                    print("      📍 No objects detected")

        # Performance benchmark
        print("\\n⚡ Running performance benchmark...")
        benchmark_results = detector.benchmark_performance(
            test_images_path=str(Path(demo_images[0]).parent),
            num_runs=10
        )

        print("   Benchmark Results:")
        for batch_size, results in benchmark_results.items():
            print(".2f")
            print(".2f")

        # Model export demonstration
        print("\\n📦 Model export demonstration...")
        print("   (Note: Export may take a moment for large models)")

        try:
            # Export to ONNX (smaller demo)
            onnx_path = detector.export_model(
                format='onnx',
                opset=11,
                simplify=True,
                verbose=False
            )
            print(f"   ✅ ONNX model exported: {onnx_path}")
        except Exception as e:
            print(f"   ⚠️  ONNX export failed: {e}")

        print("\\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print("\\nKey Features Demonstrated:")
        print("  ✅ Model loading and initialization")
        print("  ✅ Real-time object detection")
        print("  ✅ Multiple object class recognition")
        print("  ✅ Confidence scoring")
        print("  ✅ Performance benchmarking")
        print("  ✅ Model export capabilities")
        print("\\nNext Steps:")
        print("  1. Try with your own images: detector.predict('path/to/your/image.jpg')")
        print("  2. Train custom model: detector.train('data/data.yaml')")
        print("  3. Start API server: python src/api.py")
        print("  4. Run interactive notebook: jupyter notebook notebooks/yolov8_interactive.ipynb")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\\n❌ Demo failed: {e}")
        print("\\nTroubleshooting:")
        print("  1. Check if all dependencies are installed: pip install -r requirements.txt")
        print("  2. Ensure you have sufficient disk space for model downloads")
        print("  3. Try running with CPU if GPU memory is insufficient")
        return False

    return True

def cleanup_demo():
    """Clean up demo files"""
    import shutil

    demo_dir = Path("demo_images")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
        logger.info("Cleaned up demo images")

if __name__ == "__main__":
    try:
        success = run_demo()

        if success:
            print("\\n🧹 Cleaning up demo files...")
            cleanup_demo()
            print("   Demo cleanup completed")

    except KeyboardInterrupt:
        print("\\n\\n⏹️  Demo interrupted by user")
        cleanup_demo()

    except Exception as e:
        print(f"\\n💥 Unexpected error: {e}")
        logger.exception("Unexpected error in demo")

    print("\\n👋 Thank you for trying YOLOv8 Object Detection!")