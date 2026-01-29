"""
YOLOv8 Object Detection API
===========================

FastAPI service for real-time YOLOv8 object detection inference.

Features:
- RESTful API for image and video processing
- Batch inference support
- Model management (load, unload, switch)
- Performance monitoring
- Docker-ready deployment

Author: AI Assistant
Date: 2025
"""

import os
import cv2
import numpy as np
import base64
import io
from PIL import Image
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model instance
model_instance = None
model_config = {
    'model_path': None,
    'device': 'cpu',
    'conf_threshold': 0.25,
    'iou_threshold': 0.6,
    'max_batch_size': 8
}

# Performance tracking
performance_stats = {
    'total_requests': 0,
    'total_inference_time': 0,
    'avg_inference_time': 0,
    'model_load_time': 0
}

class DetectionRequest(BaseModel):
    """Request model for detection"""
    image: str = Field(..., description="Base64 encoded image")
    conf_threshold: Optional[float] = Field(0.25, description="Confidence threshold")
    iou_threshold: Optional[float] = Field(0.6, description="IoU threshold")
    max_detections: Optional[int] = Field(100, description="Maximum number of detections")

class DetectionResponse(BaseModel):
    """Response model for detection results"""
    detections: List[Dict[str, Any]] = Field(..., description="List of detected objects")
    image_shape: List[int] = Field(..., description="Original image shape [height, width]")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class BatchDetectionRequest(BaseModel):
    """Request model for batch detection"""
    images: List[str] = Field(..., description="List of base64 encoded images")
    conf_threshold: Optional[float] = Field(0.25, description="Confidence threshold")
    iou_threshold: Optional[float] = Field(0.6, description="IoU threshold")
    max_detections: Optional[int] = Field(100, description="Maximum detections per image")

class ModelLoadRequest(BaseModel):
    """Request model for loading a model"""
    model_path: str = Field(..., description="Path to model weights")
    device: Optional[str] = Field('auto', description="Device to load model on")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting YOLOv8 Detection API")
    yield
    # Shutdown
    logger.info("Shutting down YOLOv8 Detection API")

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Real-time object detection service using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

def load_model(model_path: str, device: str = 'auto') -> YOLO:
    """
    Load YOLOv8 model

    Args:
        model_path (str): Path to model weights
        device (str): Device to load on

    Returns:
        YOLO: Loaded model instance
    """
    global model_instance, model_config

    start_time = time.time()

    try:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = YOLO(model_path)
        model.to(device)

        model_instance = model
        model_config.update({
            'model_path': model_path,
            'device': device
        })

        load_time = time.time() - start_time
        performance_stats['model_load_time'] = load_time

        logger.info(".2f")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to numpy array

    Args:
        base64_string (str): Base64 encoded image

    Returns:
        np.ndarray: Decoded image
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode numpy array to base64 string

    Args:
        image (np.ndarray): Image array

    Returns:
        str: Base64 encoded image
    """
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return ""

def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw bounding boxes and labels on image

    Args:
        image (np.ndarray): Input image
        detections (List[Dict]): Detection results

    Returns:
        np.ndarray: Image with detections drawn
    """
    result_image = image.copy()

    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        class_name = detection['class_name']
        confidence = detection['confidence']

        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_image

def process_detections(results, class_names: List[str], max_detections: int = 100) -> List[Dict[str, Any]]:
    """
    Process YOLO results into standardized format

    Args:
        results: YOLO results object
        class_names (List[str]): List of class names
        max_detections (int): Maximum number of detections to return

    Returns:
        List[Dict]: Processed detections
    """
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                if i >= max_detections:
                    break

                # Get bounding box coordinates
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox

                # Get class and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())

                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                }

                detections.append(detection)

    return detections

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "YOLOv8 Object Detection API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": model_config['device'],
        "performance": performance_stats
    }

@app.post("/load_model")
async def load_model_endpoint(request: ModelLoadRequest):
    """Load YOLOv8 model"""
    try:
        model = load_model(request.model_path, request.device)
        return {
            "message": "Model loaded successfully",
            "model_path": request.model_path,
            "device": request.device,
            "load_time": performance_stats['model_load_time']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_objects(request: DetectionRequest):
    """Detect objects in image"""
    global performance_stats

    if model_instance is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Use /load_model first.")

    start_time = time.time()

    try:
        # Decode image
        image = decode_base64_image(request.image)
        height, width = image.shape[:2]

        # Run inference
        results = model_instance(
            image,
            conf=request.conf_threshold or model_config['conf_threshold'],
            iou=request.iou_threshold or model_config['iou_threshold'],
            verbose=False
        )

        # Process results
        class_names = model_instance.names if hasattr(model_instance, 'names') else []
        detections = process_detections(results, class_names, request.max_detections)

        processing_time = time.time() - start_time

        # Update performance stats
        performance_stats['total_requests'] += 1
        performance_stats['total_inference_time'] += processing_time
        performance_stats['avg_inference_time'] = performance_stats['total_inference_time'] / performance_stats['total_requests']

        response = DetectionResponse(
            detections=detections,
            image_shape=[height, width, 3],
            processing_time=processing_time,
            model_info={
                'model_path': model_config['model_path'],
                'device': model_config['device'],
                'conf_threshold': request.conf_threshold or model_config['conf_threshold'],
                'iou_threshold': request.iou_threshold or model_config['iou_threshold']
            }
        )

        return response

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect_batch")
async def detect_objects_batch(request: BatchDetectionRequest):
    """Detect objects in batch of images"""
    global performance_stats

    if model_instance is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Use /load_model first.")

    if len(request.images) > model_config['max_batch_size']:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.images)} exceeds maximum {model_config['max_batch_size']}"
        )

    start_time = time.time()

    try:
        batch_results = []

        for base64_image in request.images:
            # Decode image
            image = decode_base64_image(base64_image)

            # Run inference
            results = model_instance(
                image,
                conf=request.conf_threshold or model_config['conf_threshold'],
                iou=request.iou_threshold or model_config['iou_threshold'],
                verbose=False
            )

            # Process results
            class_names = model_instance.names if hasattr(model_instance, 'names') else []
            detections = process_detections(results, class_names, request.max_detections)

            batch_results.append({
                'detections': detections,
                'image_shape': list(image.shape)
            })

        processing_time = time.time() - start_time

        # Update performance stats
        performance_stats['total_requests'] += len(request.images)
        performance_stats['total_inference_time'] += processing_time
        performance_stats['avg_inference_time'] = performance_stats['total_inference_time'] / performance_stats['total_requests']

        return {
            'batch_results': batch_results,
            'total_images': len(request.images),
            'processing_time': processing_time,
            'avg_time_per_image': processing_time / len(request.images)
        }

    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@app.post("/detect_with_image")
async def detect_with_visualization(request: DetectionRequest):
    """Detect objects and return image with bounding boxes"""
    global performance_stats

    if model_instance is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Use /load_model first.")

    start_time = time.time()

    try:
        # Decode image
        image = decode_base64_image(request.image)

        # Run inference
        results = model_instance(
            image,
            conf=request.conf_threshold or model_config['conf_threshold'],
            iou=request.iou_threshold or model_config['iou_threshold'],
            verbose=False
        )

        # Process results
        class_names = model_instance.names if hasattr(model_instance, 'names') else []
        detections = process_detections(results, class_names, request.max_detections)

        # Draw detections on image
        result_image = draw_detections(image, detections)

        # Encode result image
        result_base64 = encode_image_to_base64(result_image)

        processing_time = time.time() - start_time

        # Update performance stats
        performance_stats['total_requests'] += 1
        performance_stats['total_inference_time'] += processing_time
        performance_stats['avg_inference_time'] = performance_stats['total_inference_time'] / performance_stats['total_requests']

        return {
            'detections': detections,
            'result_image': result_base64,
            'image_shape': list(image.shape),
            'processing_time': processing_time,
            'model_info': {
                'model_path': model_config['model_path'],
                'device': model_config['device']
            }
        }

    except Exception as e:
        logger.error(f"Detection with visualization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/upload_detect")
async def upload_and_detect(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
    max_detections: int = 100
):
    """Upload image file and detect objects"""
    global performance_stats

    if model_instance is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Use /load_model first.")

    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    start_time = time.time()

    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run inference
        results = model_instance(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        # Process results
        class_names = model_instance.names if hasattr(model_instance, 'names') else []
        detections = process_detections(results, class_names, max_detections)

        processing_time = time.time() - start_time

        # Update performance stats
        performance_stats['total_requests'] += 1
        performance_stats['total_inference_time'] += processing_time
        performance_stats['avg_inference_time'] = performance_stats['total_inference_time'] / performance_stats['total_requests']

        return DetectionResponse(
            detections=detections,
            image_shape=list(image.shape),
            processing_time=processing_time,
            model_info={
                'model_path': model_config['model_path'],
                'device': model_config['device'],
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            }
        )

    except Exception as e:
        logger.error(f"Upload detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/performance")
async def get_performance_stats():
    """Get performance statistics"""
    return {
        'performance_stats': performance_stats,
        'model_config': model_config
    }

@app.post("/reset_stats")
async def reset_performance_stats():
    """Reset performance statistics"""
    global performance_stats
    performance_stats = {
        'total_requests': 0,
        'total_inference_time': 0,
        'avg_inference_time': 0,
        'model_load_time': performance_stats['model_load_time']  # Keep model load time
    }
    return {"message": "Performance statistics reset"}

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )