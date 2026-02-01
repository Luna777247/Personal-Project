"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class OCRRequest(BaseModel):
    """Request model for OCR"""
    document_type: Optional[str] = Field(None, description="Document type hint")


class TextRegion(BaseModel):
    """Text region with bounding box"""
    box: List[float] = Field(..., description="Bounding box coordinates")
    text: str = Field(..., description="Recognized text")
    index: int = Field(..., description="Region index")


class OCRResponse(BaseModel):
    """Response model for OCR"""
    source: str = Field(..., description="Source identifier")
    status: str = Field(..., description="Processing status")
    document_type: str = Field(..., description="Detected document type")
    raw_text: str = Field(..., description="All recognized text")
    text_regions: List[TextRegion] = Field(..., description="Individual text regions")
    extracted_fields: Dict[str, Any] = Field(..., description="Extracted structured fields")
    confidence: float = Field(..., description="Overall confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchOCRResponse(BaseModel):
    """Response model for batch OCR"""
    total: int = Field(..., description="Total number of files")
    processed: int = Field(..., description="Number of successfully processed files")
    results: List[OCRResponse] = Field(..., description="Individual results")


class DocumentType(BaseModel):
    """Document type information"""
    code: str = Field(..., description="Document type code")
    name: str = Field(..., description="Document name (Vietnamese)")
    name_en: str = Field(..., description="Document name (English)")
    fields: List[str] = Field(..., description="Available fields")


class DocumentTypesResponse(BaseModel):
    """Response model for document types"""
    supported_types: List[DocumentType]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Timestamp")
    pipeline_loaded: bool = Field(..., description="Pipeline initialization status")


class StatsResponse(BaseModel):
    """Statistics response"""
    total_uploads: int = Field(..., description="Total uploaded files")
    upload_directory: str = Field(..., description="Upload directory path")
    max_file_size_mb: float = Field(..., description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(..., description="Allowed file extensions")
