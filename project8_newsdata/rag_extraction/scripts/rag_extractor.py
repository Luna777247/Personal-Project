"""
RAG-Based Disaster Information Extractor

This module implements the main RAG (Retrieval-Augmented Generation) system
for disaster information extraction using vector databases and LLM.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Vector DB imports
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from pymilvus import connections, Collection
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from langchain.embeddings import OpenAIEmbeddings
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

# Text processing
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Local imports
from config.rag_config import (
    get_vector_db_config, get_embedding_config, get_chunking_config, get_rag_config,
    DEFAULT_VECTOR_DB, DEFAULT_EMBEDDING_MODEL, DEFAULT_CHUNKING_STRATEGY
)
from config.prompts import create_extraction_prompt, get_prompt
from scripts.llm_extractor import LLMExtractor, LLMExtractionResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGDisasterExtractor:
    """RAG-based disaster information extractor using vector databases"""

    def __init__(self,
                 vector_db_type: str = DEFAULT_VECTOR_DB,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 chunking_strategy: str = DEFAULT_CHUNKING_STRATEGY,
                 rag_config: str = "default"):
        """
        Initialize RAG extractor

        Args:
            vector_db_type: Type of vector database ('chroma', 'qdrant', 'milvus')
            embedding_model: Embedding model type
            chunking_strategy: Text chunking strategy
            rag_config: RAG pipeline configuration
        """
        self.vector_db_type = vector_db_type
        self.embedding_model_type = embedding_model
        self.chunking_strategy = chunking_strategy
        self.rag_config = get_rag_config(rag_config)

        # Initialize components
        self.vector_db = None
        self.embedding_model = None
        self.text_splitter = None
        self.llm_extractor = None

        # Metrics
        self.metrics = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "cache_hits": 0,
            "processing_time": 0.0
        }

        # Initialize system
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            # Initialize vector database
            self._initialize_vector_db()

            # Initialize embedding model
            self._initialize_embedding_model()

            # Initialize text splitter
            self._initialize_text_splitter()

            # Initialize LLM extractor
            self._initialize_llm_extractor()

            logger.info("✅ RAG system initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            raise

    def _initialize_vector_db(self):
        """Initialize vector database"""
        config = get_vector_db_config(self.vector_db_type)

        if self.vector_db_type == "chroma" and CHROMA_AVAILABLE:
            persist_dir = Path(config["persist_directory"])
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.vector_db = chromadb.PersistentClient(path=str(persist_dir))
            self.collection = self.vector_db.get_or_create_collection(
                name=config["collection_name"],
                metadata=config.get("metadata", {})
            )
            logger.info(f"✅ ChromaDB initialized at {persist_dir}")

        elif self.vector_db_type == "qdrant" and QDRANT_AVAILABLE:
            self.vector_db = QdrantClient(
                host=config["host"],
                port=config["port"]
            )
            # Create collection if not exists
            try:
                self.vector_db.get_collection(config["collection_name"])
            except:
                self.vector_db.create_collection(
                    collection_name=config["collection_name"],
                    vectors_config={
                        "size": config["vector_size"],
                        "distance": config["distance"]
                    }
                )
            logger.info(f"✅ Qdrant initialized at {config['host']}:{config['port']}")

        elif self.vector_db_type == "milvus" and MILVUS_AVAILABLE:
            connections.connect(
                alias="default",
                host=config["host"],
                port=config["port"]
            )
            # Collection will be created when adding documents
            logger.info(f"✅ Milvus initialized at {config['host']}:{config['port']}")

        else:
            raise ValueError(f"Vector database {self.vector_db_type} not available or not supported")

    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        config = get_embedding_config(self.embedding_model_type)

        if self.embedding_model_type == "sentence-transformers" and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(
                config["model_name"],
                device=config.get("device", "auto")
            )
            logger.info(f"✅ SentenceTransformers model loaded: {config['model_name']}")

        elif self.embedding_model_type == "openai" and LANGCHAIN_OPENAI_AVAILABLE:
            self.embedding_model = OpenAIEmbeddings(
                model=config["model_name"],
                openai_api_key=self._get_openai_key()
            )
            logger.info(f"✅ OpenAI embeddings initialized: {config['model_name']}")

        else:
            raise ValueError(f"Embedding model {self.embedding_model_type} not available")

    def _initialize_text_splitter(self):
        """Initialize text splitter"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain required for text splitting")

        config = get_chunking_config(self.chunking_strategy)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=config["separators"],
            length_function=len
        )
        logger.info(f"✅ Text splitter initialized with chunk_size={config['chunk_size']}")

    def _initialize_llm_extractor(self):
        """Initialize LLM extractor"""
        try:
            self.llm_extractor = LLMExtractor()
            logger.info("✅ LLM extractor initialized")
        except ValueError:
            logger.warning("⚠️  LLM extractor not available (no API keys)")
            self.llm_extractor = None

    def _get_openai_key(self) -> str:
        """Get OpenAI API key"""
        import os
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found")
        return key

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector database

        Args:
            documents: List of document dicts with 'id', 'content', 'metadata'

        Returns:
            Success status
        """
        try:
            start_time = time.time()

            all_chunks = []
            all_embeddings = []
            all_metadata = []
            all_ids = []

            for doc in documents:
                # Chunk the document
                chunks = self._chunk_document(doc["content"])

                # Generate embeddings
                embeddings = self._generate_embeddings(chunks)

                # Prepare metadata
                for i, chunk in enumerate(chunks):
                    metadata = {
                        **doc.get("metadata", {}),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "document_id": doc["id"],
                        "chunk_text": chunk[:500]  # Preview
                    }

                    all_chunks.append(chunk)
                    all_embeddings.append(embeddings[i])
                    all_metadata.append(metadata)
                    all_ids.append(f"{doc['id']}_chunk_{i}")

            # Add to vector database
            self._add_to_vector_db(all_chunks, all_embeddings, all_metadata, all_ids)

            # Update metrics
            self.metrics["total_documents"] += len(documents)
            self.metrics["total_chunks"] += len(all_chunks)
            self.metrics["processing_time"] += time.time() - start_time

            logger.info(f"✅ Added {len(documents)} documents ({len(all_chunks)} chunks)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False

    def _chunk_document(self, content: str) -> List[str]:
        """Chunk document into smaller pieces"""
        if self.text_splitter:
            return self.text_splitter.split_text(content)
        else:
            # Fallback: simple chunking
            chunk_size = 1000
            return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        if hasattr(self.embedding_model, 'encode'):
            # SentenceTransformers
            return self.embedding_model.encode(texts, normalize_embeddings=True).tolist()
        elif hasattr(self.embedding_model, 'embed_documents'):
            # LangChain embeddings
            return self.embedding_model.embed_documents(texts)
        else:
            raise ValueError("Unsupported embedding model interface")

    def _add_to_vector_db(self, chunks: List[str], embeddings: List[List[float]],
                         metadata: List[Dict], ids: List[str]):
        """Add data to vector database"""
        if self.vector_db_type == "chroma":
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata,
                ids=ids
            )

        elif self.vector_db_type == "qdrant":
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={"text": chunk, **meta}
                )
                for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata))
            ]

            config = get_vector_db_config(self.vector_db_type)
            self.vector_db.upsert(
                collection_name=config["collection_name"],
                points=points
            )

        elif self.vector_db_type == "milvus":
            # Milvus implementation would go here
            pass

    def search_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant document chunks with scores
        """
        if top_k is None:
            top_k = self.rag_config["top_k"]

        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]

            # Search vector database
            results = self._search_vector_db(query_embedding, top_k)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.get("text", result.get("document", "")),
                    "metadata": result.get("metadata", result.get("payload", {})),
                    "score": result.get("score", result.get("distance", 0.0))
                })

            self.metrics["total_queries"] += 1
            logger.info(f"✅ Found {len(formatted_results)} relevant chunks for query")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    def _search_vector_db(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Search vector database"""
        if self.vector_db_type == "chroma":
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            return [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": 1.0 - dist  # Convert distance to similarity
                }
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]

        elif self.vector_db_type == "qdrant":
            config = get_vector_db_config(self.vector_db_type)
            results = self.vector_db.search(
                collection_name=config["collection_name"],
                query_vector=query_embedding,
                limit=top_k
            )

            return [
                {
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload,
                    "score": result.score
                }
                for result in results
            ]

        elif self.vector_db_type == "milvus":
            # Milvus search implementation
            pass

        return []

    def extract_disaster_info(self, query: str, model: str = None) -> Optional[LLMExtractionResult]:
        """
        Extract disaster information using RAG pipeline

        Args:
            query: Search query about disaster
            model: LLM model to use for extraction

        Returns:
            Extraction result or None if failed
        """
        try:
            start_time = time.time()

            # Step 1: Search for relevant chunks
            relevant_chunks = self.search_documents(query)

            if not relevant_chunks:
                logger.warning("⚠️  No relevant chunks found")
                return None

            # Step 2: Prepare context from chunks
            context_text = self._prepare_context(relevant_chunks)

            # Step 3: Extract information using LLM
            if self.llm_extractor:
                extraction_prompt = create_extraction_prompt(query, context_text)
                result = self.llm_extractor.extract_disaster_info_from_prompt(
                    extraction_prompt, model=model
                )

                processing_time = time.time() - start_time
                result.processing_time = processing_time

                logger.info(f"✅ RAG extraction completed in {processing_time:.2f}s")
                return result
            else:
                logger.error("❌ LLM extractor not available")
                return None

        except Exception as e:
            logger.error(f"❌ RAG extraction failed: {e}")
            return None

    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from relevant chunks"""
        # Sort by score and limit context length
        sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        max_length = self.rag_config["max_context_length"]

        context_parts = []
        current_length = 0

        for chunk in sorted_chunks:
            chunk_text = chunk["text"]
            chunk_length = len(chunk_text)

            if current_length + chunk_length > max_length:
                break

            context_parts.append(f"[Đoạn {len(context_parts)+1}]:\n{chunk_text}")
            current_length += chunk_length

        return "\n\n".join(context_parts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return {
            **self.metrics,
            "vector_db_type": self.vector_db_type,
            "embedding_model": self.embedding_model_type,
            "chunking_strategy": self.chunking_strategy
        }

    def clear_database(self) -> bool:
        """Clear all data from vector database"""
        try:
            if self.vector_db_type == "chroma":
                config = get_vector_db_config(self.vector_db_type)
                collection_name = config["collection_name"]

                # Delete collection and recreate
                try:
                    self.vector_db.delete_collection(collection_name)
                except:
                    pass

                self.collection = self.vector_db.create_collection(
                    name=collection_name,
                    metadata=config.get("metadata", {})
                )

            elif self.vector_db_type == "qdrant":
                config = get_vector_db_config(self.vector_db_type)
                self.vector_db.delete_collection(config["collection_name"])
                # Recreate collection
                self.vector_db.create_collection(
                    collection_name=config["collection_name"],
                    vectors_config={
                        "size": config["vector_size"],
                        "distance": config["distance"]
                    }
                )

            # Reset metrics
            self.metrics = {k: 0 if isinstance(v, (int, float)) else v for k, v in self.metrics.items()}

            logger.info("✅ Database cleared successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
            return False


# Convenience functions
def create_rag_extractor(vector_db: str = "chroma", embedding: str = "sentence-transformers") -> RAGDisasterExtractor:
    """Create RAG extractor with default settings"""
    return RAGDisasterExtractor(
        vector_db_type=vector_db,
        embedding_model=embedding
    )


def batch_add_documents(extractor: RAGDisasterExtractor, documents: List[Dict], batch_size: int = 10) -> bool:
    """Add documents in batches for better performance"""
    success = True
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        if not extractor.add_documents(batch):
            success = False
    return success


if __name__ == "__main__":
    # Test the RAG system
    try:
        extractor = create_rag_extractor()

        # Test documents
        test_docs = [
            {
                "id": "test_1",
                "content": "Bão số 12 gây thiệt hại nặng nề tại Quảng Nam. Theo báo cáo, cơn bão làm 15 người chết, 27 người bị thương.",
                "metadata": {"source": "test", "date": "2023-11-15"}
            }
        ]

        # Add documents
        extractor.add_documents(test_docs)

        # Test search and extraction
        result = extractor.extract_disaster_info("thông tin bão tại Quảng Nam")
        if result:
            print("Extracted info:", result.extracted_info)

        print("✅ RAG system test completed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()