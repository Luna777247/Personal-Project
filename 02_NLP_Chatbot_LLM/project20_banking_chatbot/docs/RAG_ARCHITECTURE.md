# RAG Architecture - Banking Chatbot

## Overview

This document explains the Retrieval-Augmented Generation (RAG) architecture used in the Banking Chatbot system.

## System Architecture

```
User Query
    │
    ▼
[Embedding Generator] → Query Vector (768 dims)
    │
    ▼
[Vector Store Search] → Top-K Similar Documents
    │
    ▼
[Retriever + Filter] → Relevant Context
    │
    ▼
[Prompt Builder] → Formatted Prompt
    │
    ▼
[LLM Router] → Generated Response
```

## Components

### 1. Embedding Generator

**Model**: `paraphrase-multilingual-mpnet-base-v2`
- **Dimensions**: 768
- **Languages**: 50+ including Vietnamese
- **Architecture**: MPNet (BERT + autoregressive)

**Why This Model?**
- Multilingual support with Vietnamese optimization
- State-of-art semantic similarity performance
- Balance between quality (768d) and speed
- Open source and free to use

### 2. Text Chunker

**Strategy**: Recursive chunking with overlap
- **Chunk Size**: 500 characters
- **Overlap**: 50 characters
- **Preserves**: Sentence boundaries

**Purpose**: Split documents into optimal sizes for:
- Embedding model limits (512 tokens)
- Precise retrieval
- Context preservation

### 3. Vector Store (ChromaDB)

**Features**:
- Persistent storage (SQLite + binary files)
- HNSW indexing (O(log N) search)
- Metadata filtering
- Cosine similarity search

**Storage Structure**:
```
chroma_db/
├── chroma.sqlite3      # Metadata
└── {collection}/       # Vectors
    ├── data_level0.bin
    └── index_metadata.pkl
```

### 4. Retriever

**Process**:
1. Generate query embedding
2. Search top-K (default: 10)
3. Filter by threshold (> 0.7)
4. Return top-5 documents

**Similarity Scoring**:
- Metric: Cosine similarity
- Range: [0, 1]
- Threshold: 0.7 (configurable)

### 5. LLM Router

**Supported Providers**:

**OpenAI**:
- `gpt-4o-mini`: Fast, cost-effective
- `gpt-4o`: High quality

**Ollama** (Local):
- `qwen2.5:latest`: Chinese/English, good for Vietnamese
- `llama3.1:latest`: General purpose

**Features**:
- Provider fallback
- Streaming support
- Token limit management
- Error handling

## Data Flow

### Indexing Phase

```
Raw Documents
    ↓
[Text Chunker] → Chunks (500 chars)
    ↓
[Embedding Generator] → Vectors (768d)
    ↓
[Vector Store] → Stored with metadata
```

### Query Phase

```
User Query
    ↓
[Embedding Generator] → Query Vector
    ↓
[Vector Search] → Similar Documents (top-10)
    ↓
[Threshold Filter] → Relevant Docs (score > 0.7)
    ↓
[Context Builder] → Formatted Context
    ↓
[Prompt Template] → Complete Prompt
    ↓
[LLM] → Generated Response
```

## Prompt Engineering

### Template Structure

```
System: You are an MB Bank assistant. Answer based on provided context.

Context:
{retrieved_documents}

Conversation History:
{previous_turns}

User: {current_query}