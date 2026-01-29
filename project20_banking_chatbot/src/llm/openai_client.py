"""
OpenAI GPT integration
"""
from openai import OpenAI
from typing import List, Dict, Optional, Generator
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI GPT client for chatbot"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response from OpenAI
        
        Args:
            messages: List of message dictionaries
            temperature: Override temperature
            max_tokens: Override max tokens
            stream: Stream response
            
        Returns:
            Generated response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream
            )
            
            if stream:
                return response  # Return generator
            else:
                return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def generate_response_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate response with RAG context
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: System prompt
            conversation_history: Previous conversation
            
        Returns:
            Generated response
        """
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-5:]:  # Last 5 turns
                messages.append({
                    "role": turn.get('role', 'user'),
                    "content": turn.get('content', '')
                })
        
        # Add context and query
        user_message = f"""Dựa trên thông tin sau đây về sản phẩm/dịch vụ của MB Bank:

{context}

Câu hỏi: {query}

Hãy trả lời câu hỏi một cách chính xác và thân thiện dựa trên thông tin được cung cấp."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = self.generate_response(messages)
        
        return response
    
    def stream_response(
        self,
        messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """
        Stream response from OpenAI
        
        Args:
            messages: List of messages
            
        Yields:
            Response chunks
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"Failed to stream response: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def check_context_length(self, messages: List[Dict]) -> bool:
        """
        Check if context fits within model limits
        
        Args:
            messages: Message list
            
        Returns:
            True if within limits
        """
        total_tokens = sum(self.count_tokens(msg['content']) for msg in messages)
        
        # GPT-4 context limit is typically 8192 or 128k tokens
        max_context = 8000  # Conservative limit
        
        return total_tokens <= max_context


class OpenAIEmbeddings:
    """OpenAI embeddings (alternative to Sentence Transformers)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002"
    ):
        """
        Initialize OpenAI embeddings
        
        Args:
            api_key: OpenAI API key
            model: Embedding model
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized OpenAI embeddings with model: {model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed text using OpenAI
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            return response.data[0].embedding
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of embeddings
        """
        embeddings = []
        
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        
        return embeddings


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize client
    client = OpenAIClient()
    
    # Test generation
    messages = [
        {"role": "system", "content": "Bạn là trợ lý ảo của MB Bank."},
        {"role": "user", "content": "Lãi suất tiết kiệm MB Bank là bao nhiêu?"}
    ]
    
    response = client.generate_response(messages)
    print(f"Response: {response}")
