"""
FastAPI Service for OCR Banking
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline import OCRPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OCR Banking API",
    description="API for recognizing and extracting information from banking documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR pipeline
ocr_pipeline: Optional[OCRPipeline] = None

# Configuration
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


@app.on_event("startup")
async def startup_event():
    """Initialize OCR pipeline on startup"""
    global ocr_pipeline
    
    try:
        logger.info("Initializing OCR pipeline...")
        
        # Read config from environment or use defaults
        detector_type = os.getenv('DETECTOR_TYPE', 'craft')
        recognizer_type = os.getenv('RECOGNIZER_TYPE', 'vietocr')
        device = os.getenv('DEVICE', 'cpu')
        
        ocr_pipeline = OCRPipeline(
            detector_type=detector_type,
            recognizer_type=recognizer_type,
            device=device,
            use_postprocessing=True
        )
        
        logger.info("OCR pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OCR pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down OCR service...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "OCR Banking API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ocr": "/api/ocr",
            "batch_ocr": "/api/ocr/batch",
            "document_types": "/api/document-types"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_loaded": ocr_pipeline is not None
    }


@app.post("/api/ocr")
async def ocr_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None)
):
    """
    Perform OCR on uploaded document
    
    Args:
        file: Uploaded image file
        document_type: Optional document type hint
        
    Returns:
        OCR results with extracted fields
    """
    if ocr_pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not initialized")
    
    try:
        # Validate file
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"
            )
        
        # Read file
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)} MB"
            )
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
        cv2.imwrite(str(save_path), image)
        
        logger.info(f"Processing image: {file.filename}")
        
        # Process with OCR pipeline
        result = ocr_pipeline.process_image_array(image, source=file.filename)
        
        # Add metadata
        result['metadata'] = {
            'filename': file.filename,
            'uploaded_at': datetime.utcnow().isoformat(),
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'saved_path': str(save_path)
        }
        
        logger.info(f"OCR completed: {file.filename}, type: {result['document_type']}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr/batch")
async def batch_ocr(files: List[UploadFile] = File(...)):
    """
    Perform OCR on multiple documents
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of OCR results
    """
    if ocr_pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not initialized")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Process each file
            result = await ocr_document(file)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return JSONResponse(content={
        "total": len(files),
        "processed": len(results),
        "results": results
    })


@app.get("/api/document-types")
async def get_document_types():
    """Get supported document types"""
    return {
        "supported_types": [
            {
                "code": "cccd",
                "name": "Căn cước công dân",
                "name_en": "Citizen ID Card",
                "fields": [
                    "id_number",
                    "full_name",
                    "date_of_birth",
                    "gender",
                    "nationality",
                    "place_of_residence"
                ]
            },
            {
                "code": "cmnd",
                "name": "Chứng minh nhân dân",
                "name_en": "Identity Card",
                "fields": [
                    "id_number",
                    "full_name",
                    "date_of_birth",
                    "gender",
                    "place_of_origin"
                ]
            },
            {
                "code": "bank_statement",
                "name": "Sao kê ngân hàng",
                "name_en": "Bank Statement",
                "fields": [
                    "account_number",
                    "account_holder",
                    "opening_balance",
                    "closing_balance",
                    "transactions"
                ]
            },
            {
                "code": "loan_document",
                "name": "Hợp đồng vay",
                "name_en": "Loan Agreement",
                "fields": [
                    "contract_number",
                    "borrower_name",
                    "loan_amount",
                    "interest_rate",
                    "term"
                ]
            }
        ]
    }


@app.get("/api/stats")
async def get_stats():
    """Get service statistics"""
    upload_count = len(list(UPLOAD_DIR.glob("*")))
    
    return {
        "total_uploads": upload_count,
        "upload_directory": str(UPLOAD_DIR),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "allowed_extensions": list(ALLOWED_EXTENSIONS)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
