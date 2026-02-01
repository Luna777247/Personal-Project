"""
Embedding generation using Sentence Transformers
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union
import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text documents"""
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of Sentence Transformer model
            device: Device to use (cpu/cuda)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings to unit length
            show_progress: Show progress bar
            
        Returns:
            Embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_documents(
        self,
        documents: List[Dict],
        text_field: str = 'content',
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode documents to embeddings
        
        Args:
            documents: List of document dictionaries
            text_field: Field containing text to encode
            normalize: Normalize embeddings
            
        Returns:
            Dictionary mapping document IDs to embeddings
        """
        logger.info(f"Encoding {len(documents)} documents...")
        
        # Extract texts
        texts = []
        doc_ids = []
        
        for doc in documents:
            if text_field in doc:
                texts.append(doc[text_field])
                doc_ids.append(doc['id'])
        
        # Generate embeddings
        embeddings = self.encode_text(texts, normalize=normalize)
        
        # Create mapping
        embedding_map = {}
        for doc_id, embedding in zip(doc_ids, embeddings):
            embedding_map[doc_id] = embedding
        
        logger.info(f"Generated embeddings for {len(embedding_map)} documents")
        
        return embedding_map
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode query to embedding
        
        Args:
            query: Query text
            normalize: Normalize embedding
            
        Returns:
            Query embedding
        """
        embedding = self.encode_text(query, normalize=normalize, show_progress=False)
        
        return embedding[0]  # Return single embedding
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], file_path: str):
        """
        Save embeddings to file
        
        Args:
            embeddings: Dictionary of embeddings
            file_path: Path to save file
        """
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Saved embeddings to {file_path}")
    
    def load_embeddings(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from file
        
        Args:
            file_path: Path to embeddings file
            
        Returns:
            Dictionary of embeddings
        """
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
        
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine/dot)
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity (assumes normalized embeddings)
            similarity = np.dot(embedding1, embedding2)
        elif metric == 'dot':
            # Dot product
            similarity = np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(similarity)
    
    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: Dict[str, np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar documents to query
        
        Args:
            query_embedding: Query embedding
            doc_embeddings: Document embeddings
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, similarity) tuples
        """
        similarities = []
        
        for doc_id, doc_embedding in doc_embeddings.items():
            similarity = self.compute_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class BatchEmbedder:
    """Batch process documents for embedding generation"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize batch embedder
        
        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.embedder = embedding_generator
    
    def process_documents_in_batches(
        self,
        documents: List[Dict],
        batch_size: int = 100,
        save_every: int = 500,
        output_dir: str = "data/embeddings"
    ) -> Dict[str, np.ndarray]:
        """
        Process documents in batches
        
        Args:
            documents: List of documents
            batch_size: Batch size
            save_every: Save checkpoint every N documents
            output_dir: Directory to save checkpoints
            
        Returns:
            All embeddings
        """
        all_embeddings = {}
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Generate embeddings for batch
            batch_embeddings = self.embedder.encode_documents(batch)
            all_embeddings.update(batch_embeddings)
            
            # Save checkpoint
            if (i + batch_size) % save_every == 0:
                checkpoint_path = f"{output_dir}/checkpoint_{i+batch_size}.pkl"
                self.embedder.save_embeddings(all_embeddings, checkpoint_path)
        
        logger.info(f"Completed processing {len(all_embeddings)} documents")
        
        return all_embeddings


def generate_embeddings_for_documents(
    documents_path: str = "data/processed/documents.json",
    output_path: str = "data/embeddings/doc_embeddings.pkl",
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for processed documents
    
    Args:
        documents_path: Path to processed documents
        output_path: Path to save embeddings
        model_name: Embedding model name
        
    Returns:
        Document embeddings
    """
    import json
    
    # Load documents
    logger.info(f"Loading documents from {documents_path}")
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Initialize embedder
    embedder = EmbeddingGenerator(model_name=model_name)
    
    # Generate embeddings
    embeddings = embedder.encode_documents(documents)
    
    # Save embeddings
    embedder.save_embeddings(embeddings, output_path)
    
    logger.info(f"Generated embeddings for {len(embeddings)} documents")
    logger.info(f"Embedding dimension: {embedder.embedding_dim}")
    
    return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument('-i', '--input', default='data/processed/documents.json',
                       help='Input documents file')
    parser.add_argument('-o', '--output', default='data/embeddings/doc_embeddings.pkl',
                       help='Output embeddings file')
    parser.add_argument('-m', '--model', default='paraphrase-multilingual-mpnet-base-v2',
                       help='Embedding model name')
    
    args = parser.parse_args()
    
    generate_embeddings_for_documents(args.input, args.output, args.model)
