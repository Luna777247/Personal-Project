"""
FastAPI Inference Service
REST API for model inference using Triton backend
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from loguru import logger

from src.inference.triton_client import TritonGRPCClient


# Pydantic models
class InferenceRequest(BaseModel):
    """Inference request model"""
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field("", description="Version of the model (empty for latest)")
    inputs: Dict[str, List[List[float]]] = Field(..., description="Input data")
    outputs: Optional[List[str]] = Field(None, description="Output names to return")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "fraud_detector",
                "model_version": "1",
                "inputs": {
                    "float_input": [[0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.6, 0.7, 0.3]]
                }
            }
        }


class InferenceResponse(BaseModel):
    """Inference response model"""
    model_name: str
    model_version: str
    outputs: Dict[str, List[Any]]
    inference_time_ms: float
    timestamp: str


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    versions: List[str]
    platform: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    ready: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    triton_server: str
    models_loaded: int
    uptime_seconds: float


# Prometheus metrics
inference_counter = Counter(
    'model_inference_total',
    'Total number of inference requests',
    ['model', 'version', 'status']
)

inference_latency = Histogram(
    'model_inference_latency_seconds',
    'Inference latency in seconds',
    ['model', 'version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

active_requests = Gauge(
    'model_active_requests',
    'Number of active inference requests',
    ['model', 'version']
)

model_ready_gauge = Gauge(
    'model_ready',
    'Model readiness status (1=ready, 0=not ready)',
    ['model', 'version']
)


# FastAPI app
app = FastAPI(
    title="AI Ops Inference API",
    description="Production inference API for ML models via Triton",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
triton_client: Optional[TritonGRPCClient] = None
start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize Triton client on startup"""
    global triton_client
    
    try:
        triton_url = "triton:8001"  # Docker service name
        logger.info(f"Connecting to Triton: {triton_url}")
        
        triton_client = TritonGRPCClient(url=triton_url, verbose=False)
        
        # Get server info
        server_info = triton_client.get_server_metadata()
        logger.info(f"✓ Connected to Triton {server_info['version']}")
        
    except Exception as e:
        logger.error(f"Failed to connect to Triton: {e}")
        logger.warning("Starting without Triton connection")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global triton_client
    
    if triton_client:
        triton_client.close()
        logger.info("Triton client closed")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Ops Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if not triton_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client not initialized"
        )
    
    try:
        # Check server status
        is_live = triton_client.client.is_server_live()
        is_ready = triton_client.client.is_server_ready()
        
        if not (is_live and is_ready):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Triton server not ready"
            )
        
        # Count loaded models (simplified)
        models_loaded = 1  # TODO: Implement actual model counting
        
        return HealthResponse(
            status="healthy",
            triton_server="connected",
            models_loaded=models_loaded,
            uptime_seconds=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.get("/models", response_model=List[str], tags=["Models"])
async def list_models():
    """List available models"""
    if not triton_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client not initialized"
        )
    
    try:
        # TODO: Implement actual model listing from Triton
        return ["fraud_detector", "credit_scorer"]
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str, model_version: str = ""):
    """Get model information"""
    if not triton_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client not initialized"
        )
    
    try:
        # Check if model is ready
        is_ready = triton_client.is_model_ready(model_name, model_version)
        
        if not is_ready:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found or not ready"
            )
        
        # Get model metadata
        metadata = triton_client.get_model_metadata(model_name, model_version)
        
        return ModelInfo(
            name=metadata["name"],
            versions=metadata["versions"],
            platform=metadata["platform"],
            inputs=metadata["inputs"],
            outputs=metadata["outputs"],
            ready=is_ready
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def infer(request: InferenceRequest):
    """Perform model inference"""
    if not triton_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client not initialized"
        )
    
    model_name = request.model_name
    model_version = request.model_version
    
    # Update active requests gauge
    active_requests.labels(model=model_name, version=model_version).inc()
    
    try:
        # Check if model is ready
        if not triton_client.is_model_ready(model_name, model_version):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found or not ready"
            )
        
        # Prepare inputs
        triton_inputs = {}
        for input_name, input_data in request.inputs.items():
            triton_inputs[input_name] = np.array(input_data, dtype=np.float32)
        
        # Perform inference with timing
        start_time_infer = time.time()
        
        with inference_latency.labels(model=model_name, version=model_version).time():
            results = triton_client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=request.outputs,
                model_version=model_version
            )
        
        inference_time = (time.time() - start_time_infer) * 1000  # ms
        
        # Convert numpy arrays to lists for JSON serialization
        output_dict = {}
        for output_name, output_data in results.items():
            output_dict[output_name] = output_data.tolist()
        
        # Update success counter
        inference_counter.labels(
            model=model_name,
            version=model_version,
            status="success"
        ).inc()
        
        return InferenceResponse(
            model_name=model_name,
            model_version=model_version or "latest",
            outputs=output_dict,
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        inference_counter.labels(
            model=model_name,
            version=model_version,
            status="error"
        ).inc()
        raise
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        inference_counter.labels(
            model=model_name,
            version=model_version,
            status="error"
        ).inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    finally:
        # Decrement active requests
        active_requests.labels(model=model_name, version=model_version).dec()


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/statistics/{model_name}", tags=["Monitoring"])
async def get_model_statistics(model_name: str, model_version: str = ""):
    """Get model inference statistics"""
    if not triton_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton client not initialized"
        )
    
    try:
        stats = triton_client.get_inference_statistics(model_name, model_version)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return Response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
