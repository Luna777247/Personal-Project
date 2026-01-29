"""
Ollama client for local LLMs (Qwen2.5, Llama3.1)
"""
import requests
from typing import List, Dict, Optional, Generator
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama client for local LLMs"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        num_ctx: int = 4096
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL
            model: Model name (qwen2.5:7b, llama3.1:8b, mistral:7b)
            temperature: Temperature for generation
            num_ctx: Context window size
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        
        # Check if Ollama is available
        self._check_connection()
        
        logger.info(f"Initialized Ollama client with model: {model}")
    
    def _check_connection(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Connected to Ollama server")
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response using Ollama
        
        Args:
            messages: List of message dictionaries
            temperature: Override temperature
            stream: Stream response
            
        Returns:
            Generated response
        """
        # Convert messages to Ollama format
        prompt = self._format_messages(messages)
        
        # Prepare request
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature or self.temperature,
                "num_ctx": self.num_ctx
            }
        }
        
        try:
            if stream:
                return self._stream_generate(payload)
            else:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get('response', '')
        
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Ollama prompt"""
        formatted = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        # Add final prompt for assistant response
        formatted.append("Assistant:")
        
        return "\n\n".join(formatted)
    
    def _stream_generate(self, payload: Dict) -> Generator[str, None, None]:
        """Stream response from Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
        
        except Exception as e:
            logger.error(f"Failed to stream response: {e}")
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
            for turn in conversation_history[-5:]:
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
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat using Ollama's chat endpoint
        
        Args:
            messages: List of messages
            
        Returns:
            Response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '')
        
        except Exception as e:
            logger.error(f"Failed to chat: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            return models
        
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str):
        """
        Pull model from Ollama registry
        
        Args:
            model_name: Model name to pull
        """
        payload = {"name": model_name, "stream": True}
        
        try:
            logger.info(f"Pulling model: {model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        logger.info(f"Pull status: {data['status']}")
            
            logger.info(f"Successfully pulled model: {model_name}")
        
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            raise


class OllamaModelManager:
    """Manage multiple Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize model manager
        
        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self.models = {
            'qwen2.5': OllamaClient(base_url, model="qwen2.5:7b"),
            'llama3.1': OllamaClient(base_url, model="llama3.1:8b"),
            'mistral': OllamaClient(base_url, model="mistral:7b")
        }
    
    def get_model(self, model_name: str) -> OllamaClient:
        """
        Get model client
        
        Args:
            model_name: Model name (qwen2.5, llama3.1, mistral)
            
        Returns:
            OllamaClient instance
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name]
    
    def generate_with_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Generate response with specific model
        
        Args:
            model_name: Model name
            messages: Messages
            
        Returns:
            Response
        """
        client = self.get_model(model_name)
        return client.generate_response(messages)


if __name__ == "__main__":
    # Example usage
    client = OllamaClient(model="qwen2.5:7b")
    
    # List available models
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Test generation
    messages = [
        {"role": "system", "content": "Bạn là trợ lý ảo của MB Bank."},
        {"role": "user", "content": "Xin chào, MB Bank có những sản phẩm tiết kiệm nào?"}
    ]
    
    response = client.generate_response(messages)
    print(f"\nResponse: {response}")
