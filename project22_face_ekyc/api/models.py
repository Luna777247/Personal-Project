"""
Pydantic models for API request/response
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum


class VerificationStatus(str, Enum):
    """Verification status enum"""
    VERIFIED = "verified"
    REJECTED = "rejected"
    PENDING = "pending"
    ERROR = "error"


class LivenessStatus(str, Enum):
    """Liveness detection status"""
    LIVE = "live"
    SPOOFED = "spoofed"
    UNCERTAIN = "uncertain"


# Request models

class FaceDetectRequest(BaseModel):
    """Face detection request"""
    image: str = Field(..., description="Base64 encoded image")
    max_faces: int = Field(default=1, ge=0, le=10, description="Maximum number of faces to detect")
    return_landmarks: bool = Field(default=True, description="Whether to return facial landmarks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image": "base64_encoded_image_string",
                "max_faces": 1,
                "return_landmarks": True
            }
        }


class FaceMatchRequest(BaseModel):
    """Face matching request"""
    selfie_image: str = Field(..., description="Base64 encoded selfie image")
    cccd_image: str = Field(..., description="Base64 encoded CCCD (ID card) image")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Similarity threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "selfie_image": "base64_encoded_selfie",
                "cccd_image": "base64_encoded_cccd",
                "threshold": 0.6
            }
        }


class FaceVerifyRequest(BaseModel):
    """Complete face verification request"""
    selfie_image: str = Field(..., description="Base64 encoded selfie image")
    cccd_image: str = Field(..., description="Base64 encoded CCCD image")
    enable_liveness: bool = Field(default=True, description="Enable liveness detection")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Matching threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "selfie_image": "base64_encoded_selfie",
                "cccd_image": "base64_encoded_cccd",
                "enable_liveness": True,
                "threshold": 0.6
            }
        }


class LivenessCheckRequest(BaseModel):
    """Liveness detection request"""
    image: str = Field(..., description="Base64 encoded face image")
    enable_blink: bool = Field(default=True, description="Enable blink detection")
    enable_head_movement: bool = Field(default=True, description="Enable head movement detection")
    enable_texture: bool = Field(default=True, description="Enable texture analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image": "base64_encoded_image",
                "enable_blink": True,
                "enable_head_movement": True,
                "enable_texture": True
            }
        }


# Response models

class FaceInfo(BaseModel):
    """Face detection information"""
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score")
    landmarks: Optional[List[List[float]]] = Field(default=None, description="Facial landmarks")
    age: Optional[int] = Field(default=None, description="Estimated age")
    gender: Optional[int] = Field(default=None, description="Gender (0: female, 1: male)")


class FaceDetectResponse(BaseModel):
    """Face detection response"""
    success: bool = Field(..., description="Whether detection was successful")
    num_faces: int = Field(..., description="Number of faces detected")
    faces: List[FaceInfo] = Field(..., description="List of detected faces")
    message: Optional[str] = Field(default=None, description="Additional message")


class FaceMatchResponse(BaseModel):
    """Face matching response"""
    success: bool = Field(..., description="Whether matching was successful")
    is_match: bool = Field(..., description="Whether faces match")
    similarity: float = Field(..., description="Similarity score")
    threshold: float = Field(..., description="Threshold used")
    metric: str = Field(..., description="Similarity metric used")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    message: Optional[str] = Field(default=None, description="Additional message")


class LivenessCheckResponse(BaseModel):
    """Liveness detection response"""
    success: bool = Field(..., description="Whether check was successful")
    is_live: bool = Field(..., description="Whether face is live")
    status: LivenessStatus = Field(..., description="Liveness status")
    confidence: float = Field(..., description="Confidence score")
    details: Dict[str, Any] = Field(..., description="Detailed results")
    message: Optional[str] = Field(default=None, description="Additional message")


class FaceVerifyResponse(BaseModel):
    """Complete face verification response"""
    success: bool = Field(..., description="Whether verification was successful")
    verified: bool = Field(..., description="Whether identity is verified")
    status: VerificationStatus = Field(..., description="Verification status")
    confidence: float = Field(..., description="Overall confidence score")
    face_match: Optional[Dict[str, Any]] = Field(default=None, description="Face matching results")
    liveness: Optional[Dict[str, Any]] = Field(default=None, description="Liveness detection results")
    errors: List[str] = Field(default_factory=list, description="List of errors")
    message: Optional[str] = Field(default=None, description="Additional message")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    uptime_seconds: float = Field(..., description="Uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
