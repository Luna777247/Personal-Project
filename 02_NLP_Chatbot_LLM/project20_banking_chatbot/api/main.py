"""
FastAPI main application
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sys
from pathlib import Path
import logging
from typing import Dict
import time
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.models import (
    ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse,
    HealthResponse, ConversationHistoryRequest, ConversationHistoryResponse,
    ErrorResponse, RetrievedDocument, ConversationTurn
)
from api.session import session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MB Bank Chatbot API",
    description="RAG-based chatbot for MB Bank products and services",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (will be initialized on startup)
rag_pipeline = None
feedback_store = []


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global rag_pipeline
    
    try:
        logger.info("Initializing RAG pipeline...")
        
        # Import here to avoid circular imports
        from rag import EmbeddingGenerator, ChromaVectorStore, Retriever
        from llm import LLMRouter
        
        # Initialize components
        embedder = EmbeddingGenerator(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )
        
        vector_store = ChromaVectorStore(
            collection_name="mb_banking_docs",
            persist_directory="data/embeddings/chroma_db"
        )
        
        retriever = Retriever(embedder, vector_store, top_k=5, score_threshold=0.7)
        
        llm_router = LLMRouter(
            default_provider="openai",
            openai_config={
                "model": "gpt-4-turbo-preview",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            ollama_config={
                "base_url": "http://localhost:11434",
                "model": "qwen2.5:7b",
                "temperature": 0.7
            }
        )
        
        from llm import RAGPipeline
        rag_pipeline = RAGPipeline(retriever, llm_router, max_context_length=2000)
        
        logger.info("RAG pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Continue startup even if initialization fails


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "MB Bank Chatbot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {
        "rag_pipeline": "ready" if rag_pipeline else "not_initialized",
        "session_manager": "ready",
        "feedback_store": "ready"
    }
    
    return HealthResponse(
        status="healthy" if rag_pipeline else "degraded",
        version="1.0.0",
        services=services_status
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat endpoint with RAG
    
    Args:
        request: Chat request
        
    Returns:
        Chat response with retrieved documents
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)
        elif not session_manager.get_session(session_id):
            session_manager.sessions[session_id] = {
                'session_id': session_id,
                'user_id': request.user_id,
                'conversation': [],
                'created_at': time.time(),
                'last_activity': time.time(),
                'metadata': {}
            }
        
        # Get conversation history
        conversation_history = session_manager.get_conversation_history(session_id)
        
        # Query RAG pipeline
        result = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            conversation_history=conversation_history,
            provider=request.provider
        )
        
        # Update session
        session_manager.update_session(session_id, request.query, result['response'])
        
        # Format retrieved documents
        retrieved_docs = [
            RetrievedDocument(
                id=doc.get('id', 'unknown'),
                content=doc.get('content', ''),
                score=doc.get('score', 0.0),
                metadata=doc.get('metadata', {})
            )
            for doc in result['retrieved_docs']
        ]
        
        # Prepare response
        response = ChatResponse(
            query=request.query,
            response=result['response'],
            session_id=session_id,
            retrieved_docs=retrieved_docs,
            provider=result['provider'],
            timing=result['timing']
        )
        
        # Background task: cleanup old sessions
        background_tasks.add_task(session_manager.cleanup_old_sessions)
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint
    
    Args:
        request: Chat request
        
    Returns:
        Streaming response
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    async def generate():
        try:
            # Get session
            session_id = request.session_id or session_manager.create_session(request.user_id)
            conversation_history = session_manager.get_conversation_history(session_id)
            
            # Retrieve documents
            retrieved_docs = rag_pipeline.retriever.retrieve(
                request.query,
                top_k=request.top_k
            )
            
            context = rag_pipeline.retriever.build_context(
                retrieved_docs,
                max_length=rag_pipeline.max_context_length
            )
            
            # Stream response
            response_text = ""
            stream = rag_pipeline.llm_router.openai_client.stream_response(
                rag_pipeline.llm_router.prompt_manager.build_messages(
                    query=request.query,
                    context=context,
                    conversation_history=conversation_history
                )
            )
            
            for chunk in stream:
                response_text += chunk
                yield chunk
            
            # Update session
            session_manager.update_session(session_id, request.query, response_text)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a response
    
    Args:
        request: Feedback request
        
    Returns:
        Feedback response
    """
    try:
        feedback_id = str(uuid.uuid4())
        
        feedback_data = {
            'feedback_id': feedback_id,
            'session_id': request.session_id,
            'query': request.query,
            'response': request.response,
            'rating': request.rating,
            'comment': request.comment,
            'user_id': request.user_id,
            'timestamp': time.time()
        }
        
        feedback_store.append(feedback_data)
        
        logger.info(f"Feedback received: {feedback_id} - Rating: {request.rating}")
        
        return FeedbackResponse(
            success=True,
            message="Cảm ơn bạn đã đánh giá!",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, limit: int = 10):
    """
    Get conversation history for a session
    
    Args:
        session_id: Session ID
        limit: Maximum turns to return
        
    Returns:
        Conversation history
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation = session_manager.get_conversation_history(session_id, limit)
    
    turns = [
        ConversationTurn(
            role=turn['role'],
            content=turn['content'],
            timestamp=turn['timestamp']
        )
        for turn in conversation
    ]
    
    return ConversationHistoryResponse(
        session_id=session_id,
        turns=turns,
        total_turns=len(conversation)
    )


@app.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    """
    Delete a conversation session
    
    Args:
        session_id: Session ID
        
    Returns:
        Success message
    """
    session_manager.delete_session(session_id)
    
    return {"message": "Session deleted successfully"}


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    session_stats = session_manager.get_stats()
    
    return {
        "sessions": session_stats,
        "feedback_count": len(feedback_store),
        "avg_rating": sum(f['rating'] for f in feedback_store) / len(feedback_store) if feedback_store else 0
    }


@app.get("/providers")
async def get_providers():
    """Get available LLM providers"""
    if not rag_pipeline:
        return {"providers": [], "models": {}}
    
    providers = rag_pipeline.llm_router.get_available_providers()
    models = {
        provider: rag_pipeline.llm_router.get_available_models(provider)
        for provider in providers
    }
    
    return {
        "providers": providers,
        "models": models,
        "default_provider": rag_pipeline.llm_router.default_provider
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
