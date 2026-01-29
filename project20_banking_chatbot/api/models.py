"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    user_id: Optional[str] = Field(None, description="User ID")
    provider: Optional[str] = Field("openai", description="LLM provider (openai/ollama)")
    model: Optional[str] = Field(None, description="Specific model name")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Lãi suất tiết kiệm MB Bank kỳ hạn 6 tháng là bao nhiêu?",
                "session_id": "user123_20240101",
                "user_id": "user123",
                "provider": "openai",
                "top_k": 5
            }
        }


class RetrievedDocument(BaseModel):
    """Retrieved document model"""
    id: str
    content: str
    score: float
    metadata: Optional[Dict] = {}


class ChatResponse(BaseModel):
    """Chat response model"""
    query: str
    response: str
    session_id: str
    retrieved_docs: List[RetrievedDocument]
    provider: str
    timing: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Lãi suất tiết kiệm là bao nhiêu?",
                "response": "Lãi suất tiết kiệm MB Bank kỳ hạn 6 tháng hiện tại là 6.0%/năm...",
                "session_id": "user123_20240101",
                "retrieved_docs": [
                    {
                        "id": "doc_001",
                        "content": "Lãi suất tiết kiệm...",
                        "score": 0.92,
                        "metadata": {"source": "mbbank.com.vn"}
                    }
                ],
                "provider": "openai",
                "timing": {
                    "retrieval": 0.15,
                    "llm": 1.23,
                    "total": 1.38
                },
                "timestamp": "2024-01-01T10:00:00"
            }
        }


class FeedbackRequest(BaseModel):
    """Feedback request model"""
    session_id: str = Field(..., description="Session ID")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Bot response")
    rating: int = Field(..., description="Rating (1-5)", ge=1, le=5)
    comment: Optional[str] = Field(None, description="Optional feedback comment")
    user_id: Optional[str] = Field(None, description="User ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user123_20240101",
                "query": "Lãi suất tiết kiệm?",
                "response": "Lãi suất 6.0%/năm",
                "rating": 5,
                "comment": "Rất hữu ích!",
                "user_id": "user123"
            }
        }


class FeedbackResponse(BaseModel):
    """Feedback response model"""
    success: bool
    message: str
    feedback_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str]


class ConversationHistoryRequest(BaseModel):
    """Conversation history request"""
    session_id: str = Field(..., description="Session ID")
    limit: Optional[int] = Field(10, description="Number of turns to return", ge=1, le=50)


class ConversationTurn(BaseModel):
    """Single conversation turn"""
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = {}


class ConversationHistoryResponse(BaseModel):
    """Conversation history response"""
    session_id: str
    turns: List[ConversationTurn]
    total_turns: int


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
