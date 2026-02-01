from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import mlflow.sklearn
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Mini - Model Serving API",
    description="Production-ready ML model serving with MLflow tracking",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Input features for prediction")

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: Optional[float] = None
    inference_time: float
    model_version: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_inference_time: float
    average_inference_time: float
    model_version: str

class ModelManager:
    def __init__(self, model_path: str = "models/best_model"):
        self.model_path = model_path
        self.model = None
        self.model_version = "1.0.0"
        self.load_model()

    def load_model(self):
        """Load MLflow model"""
        try:
            self.model = mlflow.sklearn.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")

            # Try to get model version from MLflow
            try:
                import mlflow
                client = mlflow.tracking.MlflowClient()
                # This is a simplified version - in production you'd track versions properly
                self.model_version = "1.0.0"
            except:
                pass

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make single prediction"""
        start_time = time.time()

        try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])

            # Make prediction
            prediction = self.model.predict(df)[0]

            # Get prediction probabilities if available
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(df)[0]
                confidence = float(np.max(proba))

            inference_time = time.time() - start_time

            return {
                "prediction": prediction,
                "confidence": confidence,
                "inference_time": round(inference_time, 4),
                "model_version": self.model_version,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

    def predict_batch(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch predictions"""
        start_time = time.time()

        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)

            # Make predictions
            predictions = self.model.predict(df)

            # Get prediction probabilities if available
            confidences = None
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(df)
                confidences = [float(np.max(proba)) for proba in probas]

            total_inference_time = time.time() - start_time
            avg_inference_time = total_inference_time / len(data)

            # Format responses
            prediction_responses = []
            for i, pred in enumerate(predictions):
                response = {
                    "prediction": pred,
                    "confidence": confidences[i] if confidences else None,
                    "inference_time": round(avg_inference_time, 4),
                    "model_version": self.model_version,
                    "timestamp": datetime.utcnow().isoformat()
                }
                prediction_responses.append(response)

            return {
                "predictions": prediction_responses,
                "total_inference_time": round(total_inference_time, 4),
                "average_inference_time": round(avg_inference_time, 4),
                "model_version": self.model_version
            }

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail="Batch prediction failed")

# Initialize model manager
model_manager = ModelManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MLOps Mini - Model Serving API",
        "version": "1.0.0",
        "model_loaded": model_manager.model is not None,
        "endpoints": {
            "POST /predict": "Single prediction",
            "POST /predict_batch": "Batch predictions",
            "GET /health": "Health check",
            "GET /model_info": "Model information",
            "POST /feedback": "Log prediction feedback"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_manager.model is not None,
        "model_version": model_manager.model_version
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    logger.info(f"Received prediction request with {len(request.features)} features")
    result = model_manager.predict_single(request.features)
    return PredictionResponse(**result)

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    logger.info(f"Received batch prediction request with {len(request.data)} samples")

    if len(request.data) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 samples allowed per request")

    result = model_manager.predict_batch(request.data)
    return BatchPredictionResponse(**result)

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if model_manager.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Get basic model information
    model_info = {
        "model_version": model_manager.model_version,
        "model_type": type(model_manager.model).__name__,
        "has_predict_proba": hasattr(model_manager.model, 'predict_proba')
    }

    # Try to get feature information if available
    if hasattr(model_manager.model, 'n_features_in_'):
        model_info["n_features"] = model_manager.model.n_features_in_

    if hasattr(model_manager.model, 'feature_names_in_'):
        model_info["feature_names"] = model_manager.model.feature_names_in_.tolist()

    return model_info

@app.post("/feedback")
async def log_feedback(prediction_id: str, actual_value: Any, feedback: Optional[str] = None):
    """Log prediction feedback for model monitoring"""
    # In a production system, you'd store this in a database
    feedback_data = {
        "prediction_id": prediction_id,
        "actual_value": actual_value,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save to file (in production, use a proper database)
    feedback_file = "feedback_log.jsonl"
    with open(feedback_file, 'a') as f:
        f.write(json.dumps(feedback_data) + '\n')

    logger.info(f"Feedback logged for prediction {prediction_id}")
    return {"message": "Feedback logged successfully"}

@app.get("/performance")
async def get_performance_metrics():
    """Get model performance metrics (simplified version)"""
    # In production, you'd calculate these from logged predictions and feedback
    return {
        "model_version": model_manager.model_version,
        "metrics": {
            "accuracy": 0.85,  # Would be calculated from feedback
            "precision": 0.83,
            "recall": 0.87,
            "f1_score": 0.85
        },
        "total_predictions": 1000,  # Would be tracked
        "last_updated": datetime.utcnow().isoformat()
    }

# Background task for model retraining (simplified)
def retrain_model():
    """Background task to retrain model with new data"""
    logger.info("Starting model retraining...")
    # In production, this would trigger the full ML pipeline
    time.sleep(5)  # Simulate training time
    logger.info("Model retraining completed")

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    background_tasks.add_task(retrain_model)
    return {"message": "Model retraining started in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )