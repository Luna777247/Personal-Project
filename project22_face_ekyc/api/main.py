"""
FastAPI application for Face Recognition eKYC
"""
import os
import sys
import time
import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.models import (
    FaceDetectRequest, FaceDetectResponse,
    FaceMatchRequest, FaceMatchResponse,
    FaceVerifyRequest, FaceVerifyResponse,
    LivenessCheckRequest, LivenessCheckResponse,
    HealthResponse, ErrorResponse,
    VerificationStatus, LivenessStatus, FaceInfo
)

from detection import FaceDetector
from embedding import FaceEmbedder
from liveness import LivenessDetector
from matching import FaceMatcher, CCCDProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition eKYC",
    description="Face verification API for electronic Know Your Customer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api'].get('allowed_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
start_time = time.time()
models_loaded = False
face_detector = None
face_embedder = None
liveness_detector = None
face_matcher = None
cccd_processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global models_loaded, face_detector, face_embedder, liveness_detector, face_matcher, cccd_processor
    
    try:
        logger.info("Loading models...")
        
        # Get GPU setting
        use_gpu = config['performance'].get('use_gpu', True)
        ctx_id = config['performance'].get('gpu_device', 0) if use_gpu else -1
        
        # Initialize detector
        face_detector = FaceDetector(
            model_pack=config['models']['insightface']['model_pack'],
            det_size=tuple(config['models']['insightface']['det_size']),
            det_thresh=config['models']['insightface']['det_thresh'],
            ctx_id=ctx_id
        )
        
        # Initialize embedder
        face_embedder = FaceEmbedder(
            model_pack=config['models']['insightface']['model_pack'],
            ctx_id=ctx_id
        )
        
        # Initialize liveness detector
        liveness_detector = LivenessDetector(
            enable_blink=config['liveness']['blink'].get('enable', True) if 'blink' in config['liveness'] else True,
            enable_head_movement=config['liveness']['head_movement'].get('enable', True) if 'head_movement' in config['liveness'] else True,
            enable_texture=config['liveness']['texture'].get('enable', True) if 'texture' in config['liveness'] else True,
            config=config['liveness']
        )
        
        # Initialize face matcher
        face_matcher = FaceMatcher(
            similarity_threshold=config['matching']['similarity_threshold'],
            distance_threshold=config['matching']['distance_threshold'],
            metric=config['matching']['metric'],
            config=config
        )
        
        # Initialize CCCD processor
        cccd_processor = CCCDProcessor()
        
        models_loaded = True
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        models_loaded = False


def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_b64)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition eKYC API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        version="1.0.0",
        models_loaded=models_loaded,
        uptime_seconds=uptime
    )


@app.post("/detect", response_model=FaceDetectResponse)
async def detect_faces(request: FaceDetectRequest):
    """
    Detect faces in image
    
    - **image**: Base64 encoded image
    - **max_faces**: Maximum number of faces to detect
    - **return_landmarks**: Whether to return facial landmarks
    """
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    try:
        # Decode image
        image = decode_image(request.image)
        
        # Detect faces
        faces = face_detector.detect_faces(
            image,
            max_num=request.max_faces if request.max_faces > 0 else 0
        )
        
        # Format response
        face_infos = []
        for face in faces:
            face_info = FaceInfo(
                bbox=face['bbox'],
                confidence=face['det_score'],
                landmarks=face['landmark'] if request.return_landmarks else None,
                age=face.get('age'),
                gender=face.get('gender')
            )
            face_infos.append(face_info)
        
        return FaceDetectResponse(
            success=True,
            num_faces=len(faces),
            faces=face_infos,
            message=f"Detected {len(faces)} face(s)"
        )
    
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/match", response_model=FaceMatchResponse)
async def match_faces(request: FaceMatchRequest):
    """
    Match two face images
    
    - **selfie_image**: Base64 encoded selfie image
    - **cccd_image**: Base64 encoded CCCD (ID card) image
    - **threshold**: Optional similarity threshold
    """
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    try:
        # Decode images
        selfie = decode_image(request.selfie_image)
        cccd = decode_image(request.cccd_image)
        
        # Override threshold if provided
        threshold = request.threshold if request.threshold is not None else config['matching']['similarity_threshold']
        face_matcher.similarity_threshold = threshold
        
        # Match faces
        is_match, similarity, details = face_matcher.match_faces(
            selfie,
            cccd,
            return_details=True
        )
        
        return FaceMatchResponse(
            success=True,
            is_match=is_match,
            similarity=similarity,
            threshold=threshold,
            metric=config['matching']['metric'],
            details=details,
            message="Faces match" if is_match else "Faces do not match"
        )
    
    except Exception as e:
        logger.error(f"Face matching error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/liveness", response_model=LivenessCheckResponse)
async def check_liveness(request: LivenessCheckRequest):
    """
    Check liveness of face image
    
    - **image**: Base64 encoded face image
    - **enable_blink**: Enable blink detection
    - **enable_head_movement**: Enable head movement detection
    - **enable_texture**: Enable texture analysis
    """
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    try:
        # Decode image
        image = decode_image(request.image)
        
        # Detect face
        face = face_detector.get_largest_face(image)
        if face is None:
            return LivenessCheckResponse(
                success=False,
                is_live=False,
                status=LivenessStatus.UNCERTAIN,
                confidence=0.0,
                details={},
                message="No face detected"
            )
        
        # Configure liveness detector
        liveness_detector.enable_blink = request.enable_blink
        liveness_detector.enable_head_movement = request.enable_head_movement
        liveness_detector.enable_texture = request.enable_texture
        
        # Check liveness
        is_live, details = liveness_detector.detect(
            image,
            face_bbox=face['bbox']
        )
        
        confidence = details['overall']['confidence']
        
        return LivenessCheckResponse(
            success=True,
            is_live=is_live,
            status=LivenessStatus.LIVE if is_live else LivenessStatus.SPOOFED,
            confidence=confidence,
            details=details,
            message="Face is live" if is_live else "Face appears to be spoofed"
        )
    
    except Exception as e:
        logger.error(f"Liveness check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/verify", response_model=FaceVerifyResponse)
async def verify_identity(request: FaceVerifyRequest):
    """
    Complete identity verification (face matching + liveness)
    
    - **selfie_image**: Base64 encoded selfie image
    - **cccd_image**: Base64 encoded CCCD image
    - **enable_liveness**: Enable liveness detection
    - **threshold**: Optional matching threshold
    """
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )
    
    try:
        # Decode images
        selfie = decode_image(request.selfie_image)
        cccd = decode_image(request.cccd_image)
        
        # Override threshold if provided
        if request.threshold is not None:
            face_matcher.similarity_threshold = request.threshold
        
        # Verify identity
        result = face_matcher.verify_identity(
            selfie,
            cccd,
            liveness_check=request.enable_liveness,
            liveness_detector=liveness_detector if request.enable_liveness else None
        )
        
        # Determine status
        if result['verified']:
            status_value = VerificationStatus.VERIFIED
        elif result['errors']:
            status_value = VerificationStatus.REJECTED
        else:
            status_value = VerificationStatus.PENDING
        
        return FaceVerifyResponse(
            success=True,
            verified=result['verified'],
            status=status_value,
            confidence=result['confidence'],
            face_match=result['face_match'],
            liveness=result['liveness'],
            errors=result['errors'],
            message="Identity verified" if result['verified'] else "Verification failed"
        )
    
    except Exception as e:
        logger.error(f"Identity verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred"
        ).dict()
    )


def main():
    """Run the API server"""
    import uvicorn
    
    host = config['api'].get('host', '0.0.0.0')
    port = config['api'].get('port', 8000)
    workers = config['api'].get('workers', 1)
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )


if __name__ == "__main__":
    main()
