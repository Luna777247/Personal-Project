"""
LLM Configuration for Disaster Information Extraction

This module contains configuration settings for various Large Language Models
used in disaster information extraction from Vietnamese news articles.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for a specific LLM model"""
    provider: str
    model_name: str
    api_key_env: str
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.1
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    cost_per_1k_tokens: float = 0.0  # USD


# Available LLM configurations
LLM_CONFIGS = {
    # OpenAI Models
    "gpt-5.1-codex-max": LLMConfig(
        provider="openai",
        model_name="gpt-5.1-codex-max",
        api_key_env="OPENAI_API_KEY",
        max_tokens=8000,
        temperature=0.1,
        cost_per_1k_tokens=0.05
    ),
    "gpt-4": LLMConfig(
        provider="openai",
        model_name="gpt-4",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4000,
        temperature=0.1,
        cost_per_1k_tokens=0.03
    ),
    "gpt-4-turbo": LLMConfig(
        provider="openai",
        model_name="gpt-4-turbo-preview",
        api_key_env="OPENAI_API_KEY",
        max_tokens=4000,
        temperature=0.1,
        cost_per_1k_tokens=0.01
    ),
    "gpt-3.5-turbo": LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",
        max_tokens=2000,
        temperature=0.1,
        cost_per_1k_tokens=0.002
    ),

    # Anthropic Claude Models
    "claude-3-opus": LLMConfig(
        provider="anthropic",
        model_name="claude-3-opus-20240229",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4000,
        temperature=0.1,
        cost_per_1k_tokens=0.015
    ),
    "claude-3-sonnet": LLMConfig(
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4000,
        temperature=0.1,
        cost_per_1k_tokens=0.003
    ),
    "claude-3-haiku": LLMConfig(
        provider="anthropic",
        model_name="claude-3-haiku-20240307",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=4000,
        temperature=0.1,
        cost_per_1k_tokens=0.00025
    ),

    # Groq Models (Fast Llama)
    "llama3-70b": LLMConfig(
        provider="groq",
        model_name="llama3-70b-8192",
        api_key_env="GROQ_API_KEY",
        max_tokens=8000,
        temperature=0.1,
        cost_per_1k_tokens=0.00059
    ),
    "llama3-8b": LLMConfig(
        provider="groq",
        model_name="llama3-8b-8192",
        api_key_env="GROQ_API_KEY",
        max_tokens=8000,
        temperature=0.1,
        cost_per_1k_tokens=0.00005
    ),
    "mixtral-8x7b": LLMConfig(
        provider="groq",
        model_name="mixtral-8x7b-32768",
        api_key_env="GROQ_API_KEY",
        max_tokens=32000,
        temperature=0.1,
        cost_per_1k_tokens=0.00027
    )
}

# Default model selection
DEFAULT_MODEL = "gpt-5.1-codex-max"  # Preview model enabled globally
FALLBACK_MODELS = ["gpt-4-turbo", "llama3-8b", "claude-3-haiku"]  # Robust fallbacks

# Extraction settings
EXTRACTION_SETTINGS = {
    # Model selection
    "default_model": DEFAULT_MODEL,
    "fallback_models": FALLBACK_MODELS,
    "auto_fallback": True,

    # API settings
    "max_concurrent_requests": 5,
    "rate_limit_per_minute": 60,
    "cache_enabled": True,
    "cache_ttl_hours": 24,

    # Response processing
    "max_response_length": 4000,
    "json_validation_enabled": True,
    "confidence_threshold": 0.7,
    "hallucination_check": True,

    # Cost management
    "max_cost_per_day": 10.0,  # USD
    "cost_tracking_enabled": True,
    "budget_alert_threshold": 0.8,

    # Logging and monitoring
    "log_level": "INFO",
    "log_file": str(Path(__file__).parent.parent / "data" / "llm_extraction.log"),
    "enable_console_logging": True,
    "metrics_enabled": True,

    # Error handling
    "max_retries": 3,
    "retry_backoff_factor": 2.0,
    "timeout_seconds": 60,

    # Vietnamese-specific settings
    "language": "vietnamese",
    "encoding": "utf-8",
    "normalize_unicode": True
}

# Provider-specific settings
PROVIDER_SETTINGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "supported_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-5.1-codex-max"],
        "rate_limits": {
            "requests_per_minute": 60,
            "tokens_per_minute": 90000
        }
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "supported_models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "rate_limits": {
            "requests_per_minute": 50,
            "tokens_per_minute": 40000
        }
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "supported_models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        "rate_limits": {
            "requests_per_minute": 30,
            "tokens_per_minute": 6000
        }
    }
}

# Environment variable validation
def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are set"""
    required_keys = set()
    for config in LLM_CONFIGS.values():
        required_keys.add(config.api_key_env)

    validation_results = {}
    for key in required_keys:
        validation_results[key] = bool(os.getenv(key))

    return validation_results

def get_available_models() -> List[str]:
    """Get list of models with valid API keys"""
    validation = validate_api_keys()
    available_models = []

    for model_name, config in LLM_CONFIGS.items():
        if validation.get(config.api_key_env, False):
            available_models.append(model_name)

    return available_models

def estimate_cost(text_length: int, model_name: str) -> float:
    """Estimate cost for processing text with given model"""
    if model_name not in LLM_CONFIGS:
        return 0.0

    config = LLM_CONFIGS[model_name]
    # Rough estimation: 1 token ≈ 4 characters for Vietnamese
    estimated_tokens = text_length // 4
    cost_per_1k = config.cost_per_1k_tokens

    return (estimated_tokens / 1000) * cost_per_1k