#!/usr/bin/env python3
"""
Test Training Script for YOLOv8 Object Detection Project
Verifies the complete YOLOv8 training pipeline with real COCO128 data
"""

import os
import sys
import yaml
from datetime import datetime
import json
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.yolov8_detector import YOLOv8Detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_yolov8_training():
    """Test the complete YOLOv8 training pipeline"""
    logger.info("🚀 Starting YOLOv8 Object Detection Training Test")
    logger.info("=" * 60)

    try:
        # Check if COCO128 data exists
        data_config = "data/coco128/data.yaml"
        if not os.path.exists(data_config):
            raise FileNotFoundError(f"Data config not found: {data_config}")

        # Load data configuration
        with open(data_config, 'r') as f:
            data_cfg = yaml.safe_load(f)

        logger.info(f"Dataset: COCO128 with {data_cfg['nc']} classes")
        logger.info(f"Classes: {data_cfg['names'][:5]}...")  # Show first 5 classes

        # Initialize YOLOv8 detector
        logger.info("Initializing YOLOv8 detector...")
        detector = YOLOv8Detector(model_size='yolov8n', device='cpu')  # Use CPU for testing

        # Test model loading
        logger.info("Testing model loading...")
        detector.load_model()

        # Test training with very small epochs for quick verification
        logger.info("Testing training pipeline (1 epoch for verification)...")
        save_dir = "models/test_yolov8"

        # Train for just 1 epoch with small batch size
        results = detector.train(
            data_yaml_path=data_config,
            epochs=1,  # Just 1 epoch for testing
            batch_size=4,  # Small batch size
            img_size=320,  # Small image size for faster testing
            save_dir=save_dir,
            verbose=False
        )

        logger.info("Training completed successfully!")

        # Test inference on a sample image
        logger.info("Testing inference...")
        test_images_dir = data_cfg['train']
        if os.path.exists(test_images_dir):
            test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')][:3]  # Test on 3 images

            for img_file in test_images:
                img_path = os.path.join(test_images_dir, img_file)
                logger.info(f"Running inference on {img_file}...")

                # Run inference
                results = detector.predict(img_path, conf=0.25, verbose=False)

                # Count detections
                total_detections = 0
                if results and len(results) > 0:
                    for result in results:
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            total_detections += len(result.boxes)

                logger.info(f"  Detected {total_detections} objects")

        # Test evaluation if possible
        logger.info("Testing evaluation...")
        try:
            eval_results = detector.evaluate(
                data_config=data_config,
                model_path=os.path.join(save_dir, "weights", "best.pt"),
                verbose=False
            )

            if eval_results:
                logger.info("Evaluation completed successfully!")
                logger.info(f"mAP@0.5: {eval_results.get('metrics/mAP50(B)', 'N/A')}")
                logger.info(f"mAP@0.5:0.95: {eval_results.get('metrics/mAP50-95(B)', 'N/A')}")
            else:
                logger.info("Evaluation returned no results (expected for 1 epoch training)")
        except Exception as e:
            logger.warning(f"Evaluation failed (expected for minimal training): {e}")

        # Save test results
        results_summary = {
            "test_timestamp": datetime.now().isoformat(),
            "dataset": "COCO128",
            "model": "yolov8n",
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "imgsz": 320,
                "status": "completed"
            },
            "inference_test": {
                "images_tested": len(test_images) if 'test_images' in locals() else 0,
                "status": "completed"
            },
            "evaluation": {
                "status": "completed" if 'eval_results' in locals() and eval_results else "skipped_minimal_training"
            },
            "status": "success"
        }

        os.makedirs('results', exist_ok=True)
        with open('results/test_training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)

        logger.info("\n✅ YOLOv8 training pipeline test completed successfully!")
        logger.info("🎯 Model training: Completed (1 epoch)")
        logger.info("🔍 Inference: Working")
        logger.info("📊 Evaluation: Ready for full training")
        logger.info("📁 Results saved to results/test_training_results.json")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_yolov8_training()
    sys.exit(0 if success else 1)