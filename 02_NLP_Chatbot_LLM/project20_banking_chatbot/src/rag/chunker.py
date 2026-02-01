"""
Text chunking for RAG
"""
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text into smaller pieces for RAG"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators to split on
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using recursive splitting
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator
        for separator in self.separators:
            if separator in text:
                chunks = self._split_by_separator(text, separator)
                if chunks:
                    return chunks
        
        # Fallback: split by character
        return self._split_by_chars(text)
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator"""
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split exceeds chunk size
            if len(current_chunk) + len(split) + len(separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap
                    if self.chunk_overlap > 0:
                        overlap = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap + separator + split
                    else:
                        current_chunk = split
                else:
                    # Split is too large, recursively chunk it
                    current_chunk = split
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Split text by character positions"""
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Find good split point (space)
            if end < len(text):
                while end > start and text[end] not in [' ', '\n']:
                    end -= 1
                
                if end == start:  # No space found
                    end = start + self.chunk_size
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            if not content:
                continue
            
            # Chunk content
            chunks = self.chunk_text(content)
            
            # Create document for each chunk
            for i, chunk in enumerate(chunks):
                chunked_doc = {
                    'id': f"{doc['id']}_chunk_{i}",
                    'parent_id': doc['id'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'title': doc.get('title', ''),
                    'content': chunk,
                    'source_url': doc.get('source_url', ''),
                    'doc_type': doc.get('doc_type', ''),
                    'metadata': doc.get('metadata', {}),
                    'char_count': len(chunk),
                    'word_count': len(chunk.split())
                }
                
                chunked_docs.append(chunked_doc)
        
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        
        return chunked_docs


class SentenceChunker:
    """Chunk text by sentences"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize sentence chunker
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        import re
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text by sentences"""
        sentences = self.split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last sentences for overlap
                    overlap_text = ' '.join(current_chunk)
                    if len(overlap_text) > self.chunk_overlap:
                        overlap_text = overlap_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text]
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def chunk_documents_for_rag(
    documents_path: str = "data/processed/documents.json",
    output_path: str = "data/processed/chunked_documents.json",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: str = "recursive"
) -> List[Dict]:
    """
    Chunk documents for RAG
    
    Args:
        documents_path: Path to documents
        output_path: Path to save chunked documents
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        strategy: Chunking strategy (recursive/sentence)
        
    Returns:
        Chunked documents
    """
    import json
    from pathlib import Path
    
    # Load documents
    logger.info(f"Loading documents from {documents_path}")
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Create chunker
    if strategy == "recursive":
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "sentence":
        chunker = SentenceChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Chunk documents
    chunked_docs = chunker.chunk_documents(documents)
    
    # Save chunked documents
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(chunked_docs)} chunks to {output_path}")
    
    # Print statistics
    avg_chunk_size = sum(doc['char_count'] for doc in chunked_docs) / len(chunked_docs)
    logger.info(f"Average chunk size: {avg_chunk_size:.1f} characters")
    
    return chunked_docs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk documents for RAG")
    parser.add_argument('-i', '--input', default='data/processed/documents.json',
                       help='Input documents file')
    parser.add_argument('-o', '--output', default='data/processed/chunked_documents.json',
                       help='Output chunked documents file')
    parser.add_argument('-s', '--size', type=int, default=512,
                       help='Chunk size')
    parser.add_argument('--overlap', type=int, default=50,
                       help='Chunk overlap')
    parser.add_argument('--strategy', default='recursive', choices=['recursive', 'sentence'],
                       help='Chunking strategy')
    
    args = parser.parse_args()
    
    chunk_documents_for_rag(
        args.input,
        args.output,
        args.size,
        args.overlap,
        args.strategy
    )
