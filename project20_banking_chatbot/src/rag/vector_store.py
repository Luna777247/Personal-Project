"""
Vector store using ChromaDB, FAISS, and HNSWLib
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store using ChromaDB"""
    
    def __init__(
        self,
        collection_name: str = "mb_banking_docs",
        persist_directory: str = "data/embeddings/chroma_db",
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of collection
            persist_directory: Directory to persist data
            distance_metric: Distance metric (cosine/l2/ip)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
        logger.info(f"Current document count: {self.collection.count()}")
    
    def add_documents(
        self,
        documents: List[Dict],
        embeddings: Dict[str, np.ndarray]
    ):
        """
        Add documents to vector store
        
        Args:
            documents: List of document dictionaries
            embeddings: Dictionary mapping doc IDs to embeddings
        """
        ids = []
        embedding_list = []
        metadatas = []
        documents_text = []
        
        for doc in documents:
            doc_id = doc['id']
            
            if doc_id not in embeddings:
                continue
            
            ids.append(doc_id)
            embedding_list.append(embeddings[doc_id].tolist())
            
            # Prepare metadata
            metadata = {
                'title': doc.get('title', ''),
                'source_url': doc.get('source_url', ''),
                'doc_type': doc.get('doc_type', ''),
                'word_count': doc.get('word_count', 0),
                'categories': json.dumps(doc.get('metadata', {}).get('categories', []))
            }
            metadatas.append(metadata)
            
            # Document text
            documents_text.append(doc['content'])
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            metadatas=metadatas,
            documents=documents_text
        )
        
        logger.info(f"Added {len(ids)} documents to ChromaDB")
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query vector store
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results
            filter_metadata: Metadata filters
            
        Returns:
            List of results
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        try:
            result = self.collection.get(ids=[doc_id])
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
        
        return None
    
    def delete_collection(self):
        """Delete collection"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory
        }


class FAISSVectorStore:
    """Vector store using FAISS"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_path: str = "data/embeddings/faiss_index"
    ):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Embedding dimension
            index_path: Path to save index
        """
        import faiss
        
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        
        # Create index
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (for normalized vectors = cosine)
        
        # Document mapping
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.documents = {}
        
        logger.info(f"Initialized FAISS index with dimension {embedding_dim}")
    
    def add_documents(
        self,
        documents: List[Dict],
        embeddings: Dict[str, np.ndarray]
    ):
        """Add documents to index"""
        embedding_list = []
        
        for doc in documents:
            doc_id = doc['id']
            
            if doc_id not in embeddings:
                continue
            
            # Get current index
            idx = len(self.id_to_idx)
            
            # Add mapping
            self.id_to_idx[doc_id] = idx
            self.idx_to_id[idx] = doc_id
            
            # Store document
            self.documents[doc_id] = doc
            
            # Add embedding
            embedding_list.append(embeddings[doc_id])
        
        # Add to FAISS index
        embeddings_array = np.array(embedding_list).astype('float32')
        self.index.add(embeddings_array)
        
        logger.info(f"Added {len(embedding_list)} documents to FAISS")
    
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """Query FAISS index"""
        # Search
        query_vector = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        
        # Format results
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No result
                continue
            
            doc_id = self.idx_to_id[idx]
            doc = self.documents[doc_id]
            
            result = {
                'id': doc_id,
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'score': float(distances[0][i])
            }
            results.append(result)
        
        return results
    
    def save(self):
        """Save index and mappings"""
        import faiss
        import pickle
        
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save mappings
        with open(f"{self.index_path}.pkl", 'wb') as f:
            pickle.dump({
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'documents': self.documents
            }, f)
        
        logger.info(f"Saved FAISS index to {self.index_path}")
    
    def load(self):
        """Load index and mappings"""
        import faiss
        import pickle
        
        # Load FAISS index
        self.index = faiss.read_index(f"{self.index_path}.index")
        
        # Load mappings
        with open(f"{self.index_path}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.id_to_idx = data['id_to_idx']
            self.idx_to_id = data['idx_to_id']
            self.documents = data['documents']
        
        logger.info(f"Loaded FAISS index from {self.index_path}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.embedding_dim,
            'index_path': self.index_path
        }


class VectorStoreFactory:
    """Factory to create vector stores"""
    
    @staticmethod
    def create(
        store_type: str,
        config: Dict
    ) -> Union[ChromaVectorStore, FAISSVectorStore]:
        """
        Create vector store
        
        Args:
            store_type: Type of store (chromadb/faiss/hnswlib)
            config: Configuration dictionary
            
        Returns:
            Vector store instance
        """
        if store_type == "chromadb":
            return ChromaVectorStore(
                collection_name=config.get('collection_name', 'mb_banking_docs'),
                persist_directory=config.get('persist_directory', 'data/embeddings/chroma_db'),
                distance_metric=config.get('distance_metric', 'cosine')
            )
        
        elif store_type == "faiss":
            return FAISSVectorStore(
                embedding_dim=config.get('embedding_dim', 768),
                index_path=config.get('index_path', 'data/embeddings/faiss_index')
            )
        
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")


def build_vector_store(
    documents_path: str = "data/processed/documents.json",
    embeddings_path: str = "data/embeddings/doc_embeddings.pkl",
    store_type: str = "chromadb",
    config: Optional[Dict] = None
):
    """
    Build vector store from documents and embeddings
    
    Args:
        documents_path: Path to documents
        embeddings_path: Path to embeddings
        store_type: Type of vector store
        config: Store configuration
    """
    import pickle
    
    # Load documents
    logger.info(f"Loading documents from {documents_path}")
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Create vector store
    config = config or {}
    vector_store = VectorStoreFactory.create(store_type, config)
    
    # Add documents
    vector_store.add_documents(documents, embeddings)
    
    # Save if FAISS
    if isinstance(vector_store, FAISSVectorStore):
        vector_store.save()
    
    logger.info("Vector store built successfully")
    
    return vector_store


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build vector store")
    parser.add_argument('-d', '--documents', default='data/processed/documents.json',
                       help='Documents file')
    parser.add_argument('-e', '--embeddings', default='data/embeddings/doc_embeddings.pkl',
                       help='Embeddings file')
    parser.add_argument('-t', '--type', default='chromadb', choices=['chromadb', 'faiss'],
                       help='Vector store type')
    
    args = parser.parse_args()
    
    build_vector_store(args.documents, args.embeddings, args.type)
