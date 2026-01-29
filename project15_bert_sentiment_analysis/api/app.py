from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="API for sentiment analysis using fine-tuned BERT model",
    version="1.0.0"
)

class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    inference_time: float

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_inference_time: float
    average_inference_time: float

class ModelManager:
    def __init__(self, model_path: str = "models/bert_sentiment"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.load_model()

    def load_model(self):
        """Load the fine-tuned BERT model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")

    def predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment for a single text"""
        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            inference_time = time.time() - start_time

            return {
                "text": text,
                "sentiment": self.label_map[predicted_class],
                "confidence": round(confidence, 4),
                "inference_time": round(inference_time, 4)
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

    def predict_batch(self, texts: List[str]) -> Dict:
        """Predict sentiment for multiple texts"""
        start_time = time.time()
        results = []

        try:
            for text in texts:
                result = self.predict_sentiment(text)
                results.append(result)

            total_time = time.time() - start_time
            avg_time = total_time / len(texts)

            return {
                "results": results,
                "total_inference_time": round(total_time, 4),
                "average_inference_time": round(avg_time, 4)
            }

        except Exception as e:
            logger.error(f"Error during batch prediction: {e}")
            raise HTTPException(status_code=500, detail="Batch prediction failed")

# Initialize model manager
model_manager = ModelManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BERT Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Single text sentiment analysis",
            "POST /predict_batch": "Batch sentiment analysis",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device)
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for a single text"""
    logger.info(f"Received prediction request for text: {request.text[:50]}...")
    result = model_manager.predict_sentiment(request.text)
    return SentimentResponse(**result)

@app.post("/predict_batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(request: BatchSentimentRequest):
    """Predict sentiment for multiple texts"""
    logger.info(f"Received batch prediction request with {len(request.texts)} texts")

    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts allowed per request")

    result = model_manager.predict_batch(request.texts)
    return BatchSentimentResponse(**result)

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    # In a production environment, you'd track these metrics
    return {
        "model_info": {
            "name": "BERT Sentiment Classifier",
            "labels": list(model_manager.label_map.values()),
            "max_sequence_length": 128
        },
        "performance": {
            "device": str(model_manager.device),
            "typical_inference_time": "0.05-0.15 seconds per text"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )