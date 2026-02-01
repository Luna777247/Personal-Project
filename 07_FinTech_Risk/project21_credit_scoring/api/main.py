"""
FastAPI Credit Scoring Service
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from typing import Dict
import logging
from pathlib import Path
import os

from api.models import (
    CreditScoreRequest,
    CreditScoreResponse,
    ExplainResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    RiskLevel,
    FeatureContribution
)
from api.predictor import CreditScorePredictor

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Credit Scoring API",
    description="ML-powered credit scoring service with SHAP explanations",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
predictor: CreditScorePredictor = None
start_time = time.time()
metrics = {
    "total_predictions": 0,
    "total_response_time": 0.0,
    "total_risk_score": 0.0,
    "approvals": 0
}


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    global predictor
    
    try:
        model_path = os.getenv("MODEL_PATH", "models/xgboost_model.json")
        logger.info(f"Loading model from {model_path}")
        
        # Note: In production, load feature engineer and scaler as well
        predictor = CreditScorePredictor(model_path)
        
        logger.info("API ready")
    
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        # Continue without model for testing
        logger.warning("Running without loaded model")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "Credit Scoring API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_version="1.0.0",
        uptime_seconds=time.time() - start_time
    )


@app.post("/predict", response_model=CreditScoreResponse, tags=["Prediction"])
async def predict(request: CreditScoreRequest):
    """
    Credit score prediction
    
    Returns credit score, risk level, and approval decision
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        # Convert request to dict
        data = request.dict()
        
        # Make prediction
        score, probability = predictor.predict(data)
        
        # Determine risk level
        risk_level = predictor.determine_risk_level(probability)
        
        # Determine approval
        approval = predictor.determine_approval(probability)
        
        # Get key factors
        key_factors = predictor.get_key_factors(data)
        
        # Update metrics
        response_time = (time.time() - start) * 1000
        metrics["total_predictions"] += 1
        metrics["total_response_time"] += response_time
        metrics["total_risk_score"] += probability
        if approval == "Approved":
            metrics["approvals"] += 1
        
        return CreditScoreResponse(
            score=float(score),
            probability=float(probability),
            risk_level=RiskLevel(risk_level),
            approval_decision=approval,
            key_factors=key_factors
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["Prediction"])
async def explain(request: CreditScoreRequest):
    """
    Credit score prediction with SHAP explanations
    
    Returns prediction with feature contributions
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to dict
        data = request.dict()
        
        # Make prediction
        score, probability = predictor.predict(data)
        
        # Determine risk level
        risk_level = predictor.determine_risk_level(probability)
        
        # Determine approval
        approval = predictor.determine_approval(probability)
        
        # TODO: Compute SHAP values (requires SHAP explainer)
        # For now, return mock feature contributions
        top_features = [
            FeatureContribution(
                feature="debt_to_income_ratio",
                value=data['total_debt'] / data['income'],
                shap_value=0.15
            ),
            FeatureContribution(
                feature="num_late_payments",
                value=float(data['num_late_payments']),
                shap_value=0.12
            ),
            FeatureContribution(
                feature="credit_history_length",
                value=float(data['credit_history_length']),
                shap_value=-0.08
            )
        ]
        
        return ExplainResponse(
            score=float(score),
            probability=float(probability),
            risk_level=RiskLevel(risk_level),
            approval_decision=approval,
            base_value=0.2,
            top_features=top_features
        )
    
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictRequest):
    """
    Batch credit score prediction
    
    Process multiple applications at once
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        summary = {"approved": 0, "review": 0, "denied": 0}
        
        for application in request.applications:
            data = application.dict()
            
            score, probability = predictor.predict(data)
            risk_level = predictor.determine_risk_level(probability)
            approval = predictor.determine_approval(probability)
            key_factors = predictor.get_key_factors(data)
            
            results.append(CreditScoreResponse(
                score=float(score),
                probability=float(probability),
                risk_level=RiskLevel(risk_level),
                approval_decision=approval,
                key_factors=key_factors
            ))
            
            # Update summary
            if approval == "Approved":
                summary["approved"] += 1
            elif approval == "Review":
                summary["review"] += 1
            else:
                summary["denied"] += 1
        
        return BatchPredictResponse(
            results=results,
            summary=summary
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["General"])
async def get_metrics():
    """
    Get API usage metrics
    """
    if metrics["total_predictions"] == 0:
        return MetricsResponse(
            total_predictions=0,
            avg_response_time_ms=0.0,
            approval_rate=0.0,
            avg_risk_score=0.0
        )
    
    return MetricsResponse(
        total_predictions=metrics["total_predictions"],
        avg_response_time_ms=metrics["total_response_time"] / metrics["total_predictions"],
        approval_rate=metrics["approvals"] / metrics["total_predictions"],
        avg_risk_score=metrics["total_risk_score"] / metrics["total_predictions"]
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
