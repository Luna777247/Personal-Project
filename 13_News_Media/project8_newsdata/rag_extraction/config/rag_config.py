"""
RAG Configuration for Disaster Information Extraction

This module contains configuration settings for the RAG-based extraction system,
including vector database settings, embedding models, and chunking parameters.
"""

import os
from typing import Dict, Any, List
from pathlib import Path

# Vector Database Configurations
VECTOR_DB_CONFIGS = {
    "chroma": {
        "host": "localhost",
        "port": 8000,
        "collection_name": "disaster_news",
        "persist_directory": "./data/chroma_db",
        "metadata": {
            "description": "Disaster news articles collection",
            "created_by": "rag_extraction_system"
        }
    },
    "qdrant": {
        "host": "localhost",
        "port": 6333,
        "collection_name": "disaster_news",
        "vector_size": 768,  # Depends on embedding model
        "distance": "Cosine",
        "metadata": {
            "description": "Disaster news articles collection",
            "created_by": "rag_extraction_system"
        }
    },
    "milvus": {
        "host": "localhost",
        "port": 19530,
        "collection_name": "disaster_news",
        "dimension": 768,  # Depends on embedding model
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "metadata": {
            "description": "Disaster news articles collection",
            "created_by": "rag_extraction_system"
        }
    }
}

# Embedding Model Configurations
EMBEDDING_CONFIGS = {
    "sentence-transformers": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_type": "sentence-transformers",
        "dimension": 384,
        "max_seq_length": 512,
        "device": "auto",  # auto, cpu, cuda
        "normalize_embeddings": True,
        "description": "Multilingual model good for Vietnamese text"
    },
    "openai": {
        "model_name": "text-embedding-3-small",
        "model_type": "openai",
        "dimension": 1536,
        "max_tokens": 8191,
        "cost_per_1k_tokens": 0.00002,
        "description": "OpenAI's latest embedding model"
    },
    "bge": {
        "model_name": "BAAI/bge-large-zh-v1.5",
        "model_type": "sentence-transformers",
        "dimension": 1024,
        "max_seq_length": 512,
        "device": "auto",
        "normalize_embeddings": True,
        "description": "BGE model with good multilingual support"
    }
}

# Text Chunking Configurations
CHUNKING_CONFIGS = {
    "recursive": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", ". ", " ", ""],
        "length_function": "character",
        "description": "Recursive text splitter with overlap"
    },
    "semantic": {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "separators": ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        "length_function": "character",
        "description": "Semantic-aware chunking"
    },
    "vietnamese": {
        "chunk_size": 900,
        "chunk_overlap": 150,
        "separators": ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        "length_function": "character",
        "description": "Optimized for Vietnamese text structure"
    }
}

# RAG Pipeline Configurations
RAG_CONFIGS = {
    "default": {
        "vector_db": "chroma",
        "embedding_model": "sentence-transformers",
        "chunking_strategy": "vietnamese",
        "top_k": 5,  # Number of chunks to retrieve
        "similarity_threshold": 0.7,
        "max_context_length": 4000,  # Max tokens for LLM context
        "rerank_results": True,
        "diversity_factor": 0.5  # For diverse chunk selection
    },
    "high_precision": {
        "vector_db": "qdrant",
        "embedding_model": "openai",
        "chunking_strategy": "recursive",
        "top_k": 3,
        "similarity_threshold": 0.8,
        "max_context_length": 6000,
        "rerank_results": True,
        "diversity_factor": 0.3
    },
    "large_scale": {
        "vector_db": "milvus",
        "embedding_model": "bge",
        "chunking_strategy": "recursive",
        "top_k": 10,
        "similarity_threshold": 0.6,
        "max_context_length": 8000,
        "rerank_results": False,
        "diversity_factor": 0.7
    }
}

# Disaster Query Templates
DISASTER_QUERY_TEMPLATES = {
    "general": "thông tin về thiên tai {disaster_type} tại {location}",
    "specific": "thiệt hại và số liệu về {disaster_type} ở {location} vào {time}",
    "impact": "tác động và hậu quả của {disaster_type} tại {location}",
    "response": "công tác cứu hộ và ứng phó với {disaster_type} ở {location}",
    "forecast": "dự báo và cảnh báo về {disaster_type} tại {location}"
}

# Default Settings
DEFAULT_VECTOR_DB = "chroma"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers"
DEFAULT_CHUNKING_STRATEGY = "vietnamese"
DEFAULT_RAG_CONFIG = "default"

# Performance Settings
MAX_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 32
SEARCH_TIMEOUT = 30  # seconds
INDEX_UPDATE_INTERVAL = 3600  # seconds

# Cache Settings
CACHE_TTL_SECONDS = 3600
MAX_CACHE_SIZE = 10000

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_vector_db_config(db_type: str = DEFAULT_VECTOR_DB) -> Dict[str, Any]:
    """Get vector database configuration"""
    return VECTOR_DB_CONFIGS.get(db_type, VECTOR_DB_CONFIGS[DEFAULT_VECTOR_DB])

def get_embedding_config(model_type: str = DEFAULT_EMBEDDING_MODEL) -> Dict[str, Any]:
    """Get embedding model configuration"""
    return EMBEDDING_CONFIGS.get(model_type, EMBEDDING_CONFIGS[DEFAULT_EMBEDDING_MODEL])

def get_chunking_config(strategy: str = DEFAULT_CHUNKING_STRATEGY) -> Dict[str, Any]:
    """Get chunking configuration"""
    return CHUNKING_CONFIGS.get(strategy, CHUNKING_CONFIGS[DEFAULT_CHUNKING_STRATEGY])

def get_rag_config(config_name: str = DEFAULT_RAG_CONFIG) -> Dict[str, Any]:
    """Get RAG pipeline configuration"""
    return RAG_CONFIGS.get(config_name, RAG_CONFIGS[DEFAULT_RAG_CONFIG])

def validate_configurations():
    """Validate all configurations"""
    # Check if required directories exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Validate vector DB configs
    for db_type, config in VECTOR_DB_CONFIGS.items():
        if "collection_name" not in config:
            raise ValueError(f"Missing collection_name in {db_type} config")

    # Validate embedding configs
    for model_type, config in EMBEDDING_CONFIGS.items():
        if "dimension" not in config:
            raise ValueError(f"Missing dimension in {model_type} embedding config")

    print("✅ All RAG configurations validated successfully")

if __name__ == "__main__":
    validate_configurations()