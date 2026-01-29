"""
FastAPI Service for Fraud Detection
Real-time fraud detection API with MLflow model serving
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.pyfunc
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection service using ML models",
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

# Prometheus metrics
PREDICTIONS_COUNTER = Counter('fraud_predictions_total', 'Total number of predictions', ['model', 'prediction'])
PREDICTION_TIME = Histogram('fraud_prediction_duration_seconds', 'Time spent processing prediction')
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])

# Global variables for models and feature engineer
model = None
feature_engineer = None
model_version = None
fraud_threshold = float(os.getenv('FRAUD_THRESHOLD', 0.5))


# Pydantic models
class Transaction(BaseModel):
    """Single transaction for fraud detection"""
    transaction_id: str
    customer_id: int
    merchant_id: int
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    timestamp: str = Field(..., description="Transaction timestamp (ISO format)")
    merchant_category: str
    card_type: str
    transaction_type: str
    country_code: str
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_00001234",
                "customer_id": 1234,
                "merchant_id": 567,
                "amount": 125.50,
                "timestamp": "2024-12-11T14:30:00",
                "merchant_category": "retail",
                "card_type": "credit",
                "transaction_type": "online",
                "country_code": "US",
                "device_id": "DEV_5678",
                "ip_address": "192.168.1.1"
            }
        }


class BatchTransactions(BaseModel):
    """Batch of transactions for fraud detection"""
    transactions: List[Transaction]


class PredictionResponse(BaseModel):
    """Fraud prediction response"""
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    risk_level: str
    timestamp: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    fraud_rate: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, feature_engineer, model_version
    
    try:
        # Load feature engineer
        from src.feature_engineering import TransactionFeatureEngineer
        feature_engineer = TransactionFeatureEngineer()
        
        # Try to load model from MLflow
        model_path = os.getenv('MODEL_PATH', 'models/production/')
        
        if os.path.exists(os.path.join(model_path, 'model.pkl')):
            model = joblib.load(os.path.join(model_path, 'model.pkl'))
            model_version = "production"
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No model found. Using dummy model for testing.")
            model = None
            model_version = "none"
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        model_version = "error"


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    API_REQUESTS.labels(endpoint='root', status='success').inc()
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    API_REQUESTS.labels(endpoint='health', status='success').inc()
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
@PREDICTION_TIME.time()
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud for a single transaction
    
    Args:
        transaction: Transaction data
        
    Returns:
        Prediction response with fraud probability and risk level
    """
    try:
        API_REQUESTS.labels(endpoint='predict', status='success').inc()
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Create features
        df_features = feature_engineer.create_all_features(df, fit=False)
        
        # Select numeric features
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'customer_id', 'merchant_id']]
        X = df_features[numeric_cols].fillna(0)
        
        # Predict
        fraud_proba = model.predict_proba(X)[0]
        is_fraud = int(fraud_proba >= fraud_threshold)
        
        # Determine risk level
        if fraud_proba >= 0.8:
            risk_level = "high"
        elif fraud_proba >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Update metrics
        PREDICTIONS_COUNTER.labels(model=model_version, prediction=is_fraud).inc()
        
        # Log prediction
        logger.info(f"Transaction {transaction.transaction_id}: fraud_proba={fraud_proba:.4f}, is_fraud={is_fraud}")
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=float(fraud_proba),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            model_version=model_version
        )
        
    except Exception as e:
        API_REQUESTS.labels(endpoint='predict', status='error').inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
@PREDICTION_TIME.time()
async def predict_fraud_batch(batch: BatchTransactions):
    """
    Predict fraud for a batch of transactions
    
    Args:
        batch: Batch of transactions
        
    Returns:
        Batch prediction response
    """
    try:
        API_REQUESTS.labels(endpoint='predict_batch', status='success').inc()
        
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in batch.transactions])
        
        # Create features
        df_features = feature_engineer.create_all_features(df, fit=False)
        
        # Select numeric features
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'customer_id', 'merchant_id']]
        X = df_features[numeric_cols].fillna(0)
        
        # Predict
        fraud_probas = model.predict_proba(X)
        is_frauds = (fraud_probas >= fraud_threshold).astype(int)
        
        # Create responses
        predictions = []
        for i, transaction in enumerate(batch.transactions):
            fraud_proba = float(fraud_probas[i])
            is_fraud = int(is_frauds[i])
            
            if fraud_proba >= 0.8:
                risk_level = "high"
            elif fraud_proba >= 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            predictions.append(PredictionResponse(
                transaction_id=transaction.transaction_id,
                is_fraud=is_fraud,
                fraud_probability=fraud_proba,
                risk_level=risk_level,
                timestamp=datetime.now().isoformat(),
                model_version=model_version
            ))
            
            # Update metrics
            PREDICTIONS_COUNTER.labels(model=model_version, prediction=is_fraud).inc()
        
        fraud_count = sum(is_frauds)
        fraud_rate = fraud_count / len(batch.transactions)
        
        logger.info(f"Batch prediction: {len(batch.transactions)} transactions, {fraud_count} frauds ({fraud_rate:.2%})")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(batch.transactions),
            fraud_count=int(fraud_count),
            fraud_rate=float(fraud_rate)
        )
        
    except Exception as e:
        API_REQUESTS.labels(endpoint='predict_batch', status='error').inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)
