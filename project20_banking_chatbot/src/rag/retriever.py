"""
Retriever for RAG system
"""
from typing import List, Dict, Optional
import logging
from .embedder import EmbeddingGenerator
from .vector_store import ChromaVectorStore, FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant documents for RAG"""
    
    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store,
        top_k: int = 5,
        score_threshold: float = 0.7
    ):
        """
        Initialize retriever
        
        Args:
            embedder: Embedding generator
            vector_store: Vector store instance
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            top_k: Number of results (overrides default)
            filter_metadata: Metadata filters
            
        Returns:
            List of retrieved documents
        """
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Search vector store
        k = top_k or self.top_k
        results = self.vector_store.query(
            query_embedding,
            top_k=k,
            filter_metadata=filter_metadata
        )
        
        # Filter by score threshold
        filtered_results = []
        for result in results:
            score = self._get_score(result)
            
            if score >= self.score_threshold:
                result['score'] = score
                filtered_results.append(result)
        
        logger.info(f"Retrieved {len(filtered_results)} documents for query: {query}")
        
        return filtered_results
    
    def _get_score(self, result: Dict) -> float:
        """Extract score from result"""
        # ChromaDB uses distance, FAISS uses score
        if 'distance' in result:
            # Convert distance to similarity (for cosine distance)
            return 1.0 - result['distance']
        elif 'score' in result:
            return result['score']
        else:
            return 0.0
    
    def retrieve_with_context(
        self,
        query: str,
        conversation_history: List[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve with conversation context
        
        Args:
            query: Current query
            conversation_history: Previous conversation turns
            top_k: Number of results
            
        Returns:
            Retrieved documents
        """
        # Augment query with conversation context
        if conversation_history:
            context_texts = [turn['content'] for turn in conversation_history[-3:]]
            augmented_query = f"{' '.join(context_texts)} {query}"
        else:
            augmented_query = query
        
        return self.retrieve(augmented_query, top_k)
    
    def build_context(
        self,
        retrieved_docs: List[Dict],
        max_length: int = 2000,
        include_metadata: bool = True
    ) -> str:
        """
        Build context string from retrieved documents
        
        Args:
            retrieved_docs: Retrieved documents
            max_length: Maximum context length
            include_metadata: Include metadata in context
            
        Returns:
            Context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Format document
            doc_text = doc['content']
            
            if include_metadata:
                metadata = doc.get('metadata', {})
                title = metadata.get('title', '') or doc.get('title', '')
                
                if title:
                    doc_text = f"[{title}]\n{doc_text}"
            
            # Check length
            if current_length + len(doc_text) > max_length:
                # Truncate if needed
                remaining = max_length - current_length
                if remaining > 100:  # Only add if meaningful
                    doc_text = doc_text[:remaining] + "..."
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        context = "\n\n".join(context_parts)
        
        logger.info(f"Built context with {len(retrieved_docs)} documents, {len(context)} characters")
        
        return context


class HybridRetriever(Retriever):
    """Hybrid retriever with keyword + vector search"""
    
    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store,
        top_k: int = 5,
        score_threshold: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever
        
        Args:
            embedder: Embedding generator
            vector_store: Vector store
            top_k: Number of results
            score_threshold: Score threshold
            keyword_weight: Weight for keyword search (vs vector search)
        """
        super().__init__(embedder, vector_store, top_k, score_threshold)
        self.keyword_weight = keyword_weight
        self.vector_weight = 1.0 - keyword_weight
    
    def keyword_search(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """Simple keyword search"""
        query_terms = set(query.lower().split())
        
        scores = []
        for doc in documents:
            content = doc['content'].lower()
            
            # Count matching terms
            matches = sum(1 for term in query_terms if term in content)
            score = matches / len(query_terms) if query_terms else 0
            
            scores.append((doc, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [{'content': doc['content'], 'metadata': doc.get('metadata', {}), 'score': score}
                for doc, score in scores[:top_k]]
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Hybrid retrieval combining vector and keyword search"""
        k = top_k or self.top_k
        
        # Vector search
        vector_results = super().retrieve(query, top_k=k, filter_metadata=filter_metadata)
        
        # Combine scores
        combined = {}
        
        for result in vector_results:
            doc_id = result['id']
            vector_score = result['score']
            
            combined[doc_id] = {
                'content': result['content'],
                'metadata': result.get('metadata', {}),
                'score': self.vector_weight * vector_score
            }
        
        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results[:k]


def create_retriever(
    embedder: EmbeddingGenerator,
    vector_store,
    retriever_type: str = "basic",
    config: Optional[Dict] = None
) -> Retriever:
    """
    Create retriever instance
    
    Args:
        embedder: Embedding generator
        vector_store: Vector store
        retriever_type: Type of retriever (basic/hybrid)
        config: Configuration dictionary
        
    Returns:
        Retriever instance
    """
    config = config or {}
    
    top_k = config.get('top_k', 5)
    score_threshold = config.get('score_threshold', 0.7)
    
    if retriever_type == "basic":
        return Retriever(embedder, vector_store, top_k, score_threshold)
    
    elif retriever_type == "hybrid":
        keyword_weight = config.get('keyword_weight', 0.3)
        return HybridRetriever(
            embedder, vector_store, top_k, score_threshold, keyword_weight
        )
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


if __name__ == "__main__":
    # Example usage
    from .embedder import EmbeddingGenerator
    from .vector_store import ChromaVectorStore
    
    # Initialize components
    embedder = EmbeddingGenerator()
    vector_store = ChromaVectorStore()
    
    # Create retriever
    retriever = Retriever(embedder, vector_store)
    
    # Test retrieval
    query = "Lãi suất tiết kiệm MB Bank là bao nhiêu?"
    results = retriever.retrieve(query)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(results)} documents:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:200]}...")
