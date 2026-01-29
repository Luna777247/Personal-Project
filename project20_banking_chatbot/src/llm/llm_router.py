"""
LLM router to switch between OpenAI and Ollama
"""
from typing import List, Dict, Optional, Union
import logging
from .openai_client import OpenAIClient
from .ollama_client import OllamaClient, OllamaModelManager
from .prompt_manager import PromptManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMRouter:
    """Route requests to appropriate LLM provider"""
    
    def __init__(
        self,
        default_provider: str = "openai",
        openai_config: Optional[Dict] = None,
        ollama_config: Optional[Dict] = None
    ):
        """
        Initialize LLM router
        
        Args:
            default_provider: Default provider (openai/ollama)
            openai_config: OpenAI configuration
            ollama_config: Ollama configuration
        """
        self.default_provider = default_provider
        
        # Initialize clients
        self.openai_client = None
        self.ollama_client = None
        self.ollama_manager = None
        
        # Initialize OpenAI
        if openai_config:
            try:
                self.openai_client = OpenAIClient(**openai_config)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Ollama
        if ollama_config:
            try:
                self.ollama_client = OllamaClient(**ollama_config)
                self.ollama_manager = OllamaModelManager(
                    base_url=ollama_config.get('base_url', 'http://localhost:11434')
                )
                logger.info("Ollama client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {e}")
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager()
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response using specified provider
        
        Args:
            messages: List of messages
            provider: Provider to use (overrides default)
            model: Specific model name
            stream: Stream response
            
        Returns:
            Generated response
        """
        provider = provider or self.default_provider
        
        if provider == "openai":
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized")
            return self.openai_client.generate_response(messages, stream=stream)
        
        elif provider == "ollama":
            if not self.ollama_client:
                raise ValueError("Ollama client not initialized")
            
            # Use specific model if provided
            if model:
                client = self.ollama_manager.get_model(model)
            else:
                client = self.ollama_client
            
            return client.generate_response(messages, stream=stream)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_type: Optional[str] = None
    ) -> str:
        """
        Generate response with RAG context
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous conversation
            provider: LLM provider
            model: Specific model
            prompt_type: Prompt type (auto-detected if None)
            
        Returns:
            Generated response
        """
        # Detect intent if prompt type not specified
        if not prompt_type:
            prompt_type = self.prompt_manager.detect_intent(query)
        
        # Build messages
        messages = self.prompt_manager.build_messages(
            query=query,
            context=context,
            prompt_type=prompt_type,
            conversation_history=conversation_history
        )
        
        # Generate response
        response = self.generate(messages, provider, model)
        
        return response
    
    def chat(
        self,
        query: str,
        context: str,
        session_state: Dict,
        provider: Optional[str] = None
    ) -> Dict:
        """
        Chat with context and session management
        
        Args:
            query: User query
            context: Retrieved context
            session_state: Session state containing history
            provider: LLM provider
            
        Returns:
            Response dictionary with metadata
        """
        # Get conversation history
        conversation_history = session_state.get('conversation', [])
        
        # Generate response
        response = self.generate_with_context(
            query=query,
            context=context,
            conversation_history=conversation_history,
            provider=provider
        )
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": response})
        
        session_state['conversation'] = conversation_history
        
        return {
            "response": response,
            "conversation": conversation_history,
            "provider": provider or self.default_provider
        }
    
    def switch_provider(self, provider: str):
        """
        Switch default provider
        
        Args:
            provider: Provider name
        """
        if provider not in ['openai', 'ollama']:
            raise ValueError(f"Invalid provider: {provider}")
        
        self.default_provider = provider
        logger.info(f"Switched to provider: {provider}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        providers = []
        
        if self.openai_client:
            providers.append('openai')
        if self.ollama_client:
            providers.append('ollama')
        
        return providers
    
    def get_available_models(self, provider: str) -> List[str]:
        """
        Get available models for provider
        
        Args:
            provider: Provider name
            
        Returns:
            List of model names
        """
        if provider == 'openai':
            return ['gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo']
        
        elif provider == 'ollama':
            if self.ollama_client:
                return self.ollama_client.list_models()
            return []
        
        return []


class RAGPipeline:
    """Complete RAG pipeline with LLM"""
    
    def __init__(
        self,
        retriever,
        llm_router: LLMRouter,
        max_context_length: int = 2000
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Retriever instance
            llm_router: LLM router
            max_context_length: Maximum context length
        """
        self.retriever = retriever
        self.llm_router = llm_router
        self.max_context_length = max_context_length
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        conversation_history: Optional[List[Dict]] = None,
        provider: Optional[str] = None
    ) -> Dict:
        """
        End-to-end RAG query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            conversation_history: Previous conversation
            provider: LLM provider
            
        Returns:
            Response dictionary
        """
        import time
        
        start_time = time.time()
        
        # Retrieve documents
        retrieval_start = time.time()
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        # Build context
        context = self.retriever.build_context(
            retrieved_docs,
            max_length=self.max_context_length
        )
        
        # Generate response
        llm_start = time.time()
        response = self.llm_router.generate_with_context(
            query=query,
            context=context,
            conversation_history=conversation_history,
            provider=provider
        )
        llm_time = time.time() - llm_start
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "response": response,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "provider": provider or self.llm_router.default_provider,
            "timing": {
                "retrieval": retrieval_time,
                "llm": llm_time,
                "total": total_time
            }
        }


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize router
    router = LLMRouter(
        default_provider="ollama",
        openai_config={
            "api_key": os.getenv('OPENAI_API_KEY'),
            "model": "gpt-4-turbo-preview"
        },
        ollama_config={
            "base_url": "http://localhost:11434",
            "model": "qwen2.5:7b"
        }
    )
    
    # Test generation
    query = "Lãi suất tiết kiệm MB Bank là bao nhiêu?"
    context = "Lãi suất tiết kiệm MB Bank kỳ hạn 6 tháng là 6.0%/năm."
    
    response = router.generate_with_context(query, context)
    
    print(f"Query: {query}")
    print(f"Response: {response}")
