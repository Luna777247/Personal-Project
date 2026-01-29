"""
Data processor to clean and structure scraped data
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean scraped data"""
    
    def __init__(self):
        self.processed_documents = []
    
    def load_raw_data(self, file_path: str) -> List[Dict]:
        """Load raw scraped data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} raw documents")
        return data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Vietnamese
        text = re.sub(r'[^\w\s\.\,\!\?\%\-\/\:\;\(\)]', '', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    def extract_documents(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Extract structured documents from raw data
        
        Args:
            raw_data: Raw scraped data
            
        Returns:
            List of processed documents
        """
        documents = []
        
        for page_data in raw_data:
            # Main page document
            if page_data.get('content'):
                doc = self._create_document(
                    title=page_data.get('title', ''),
                    content=page_data.get('content', []),
                    source_url=page_data.get('url', ''),
                    doc_type='page_content',
                    metadata=page_data.get('metadata', {})
                )
                if doc:
                    documents.append(doc)
            
            # Product documents
            for product in page_data.get('products', []):
                doc = self._create_product_document(product, page_data.get('url'))
                if doc:
                    documents.append(doc)
            
            # Interest rate documents
            if page_data.get('interest_rates'):
                doc = self._create_interest_rate_document(
                    page_data.get('interest_rates'),
                    page_data.get('url')
                )
                if doc:
                    documents.append(doc)
        
        logger.info(f"Extracted {len(documents)} documents")
        return documents
    
    def _create_document(
        self,
        title: str,
        content: List[str],
        source_url: str,
        doc_type: str,
        metadata: Dict = None
    ) -> Dict:
        """Create structured document"""
        # Combine and clean content
        combined_content = ' '.join(content)
        cleaned_content = self.clean_text(combined_content)
        
        if not cleaned_content or len(cleaned_content) < 50:
            return None
        
        doc = {
            'id': self._generate_doc_id(title, source_url),
            'title': self.clean_text(title),
            'content': cleaned_content,
            'source_url': source_url,
            'doc_type': doc_type,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'word_count': len(cleaned_content.split()),
            'char_count': len(cleaned_content)
        }
        
        return doc
    
    def _create_product_document(self, product: Dict, source_url: str) -> Dict:
        """Create document from product info"""
        name = product.get('name', '')
        description = product.get('description', '')
        benefits = product.get('benefits', [])
        
        # Combine content
        content_parts = [name, description]
        
        if benefits:
            content_parts.append("Lợi ích:")
            content_parts.extend(benefits)
        
        content = ' '.join(content_parts)
        cleaned_content = self.clean_text(content)
        
        if not cleaned_content or len(cleaned_content) < 30:
            return None
        
        doc = {
            'id': self._generate_doc_id(name, source_url),
            'title': self.clean_text(name),
            'content': cleaned_content,
            'source_url': source_url,
            'doc_type': 'product',
            'created_at': datetime.now().isoformat(),
            'metadata': {
                'product_name': name,
                'has_benefits': len(benefits) > 0,
                'benefit_count': len(benefits)
            },
            'word_count': len(cleaned_content.split()),
            'char_count': len(cleaned_content)
        }
        
        return doc
    
    def _create_interest_rate_document(self, rates: List[Dict], source_url: str) -> Dict:
        """Create document from interest rate data"""
        if not rates:
            return None
        
        # Format rate information
        content_parts = ["Thông tin lãi suất:"]
        
        for rate in rates:
            if 'term' in rate and 'rate' in rate:
                rate_text = f"Kỳ hạn {rate['term']}: {rate['rate']}"
                if 'condition' in rate:
                    rate_text += f" ({rate['condition']})"
                content_parts.append(rate_text)
            elif 'description' in rate:
                content_parts.append(rate['description'])
        
        content = ' '.join(content_parts)
        cleaned_content = self.clean_text(content)
        
        if not cleaned_content:
            return None
        
        doc = {
            'id': self._generate_doc_id('interest_rates', source_url),
            'title': 'Bảng lãi suất',
            'content': cleaned_content,
            'source_url': source_url,
            'doc_type': 'interest_rate',
            'created_at': datetime.now().isoformat(),
            'metadata': {
                'rate_count': len(rates)
            },
            'word_count': len(cleaned_content.split()),
            'char_count': len(cleaned_content)
        }
        
        return doc
    
    def _generate_doc_id(self, title: str, url: str) -> str:
        """Generate unique document ID"""
        import hashlib
        
        id_string = f"{title}_{url}_{datetime.now().date()}"
        doc_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
        
        return doc_id
    
    def deduplicate(self, documents: List[Dict]) -> List[Dict]:
        """Remove duplicate documents"""
        seen_ids = set()
        unique_docs = []
        
        for doc in documents:
            doc_id = doc['id']
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        removed = len(documents) - len(unique_docs)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate documents")
        
        return unique_docs
    
    def enrich_metadata(self, documents: List[Dict]) -> List[Dict]:
        """Add additional metadata"""
        for doc in documents:
            # Add category based on content
            content_lower = doc['content'].lower()
            
            categories = []
            if any(kw in content_lower for kw in ['tiết kiệm', 'gửi tiền', 'lãi suất']):
                categories.append('savings')
            if any(kw in content_lower for kw in ['thẻ', 'card', 'credit', 'debit']):
                categories.append('cards')
            if any(kw in content_lower for kw in ['vay', 'loan', 'credit', 'tín dụng']):
                categories.append('loans')
            if any(kw in content_lower for kw in ['chuyển tiền', 'transfer', 'thanh toán']):
                categories.append('payments')
            
            doc['metadata']['categories'] = categories
            
            # Add language
            doc['metadata']['language'] = 'vi'
            
            # Add search keywords
            keywords = self._extract_keywords(doc['content'])
            doc['metadata']['keywords'] = keywords
        
        return documents
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter out common words
        stop_words = {'là', 'của', 'và', 'có', 'được', 'cho', 'tại', 'từ', 'với', 'để', 'này', 'các'}
        
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        keywords = [word for word, freq in sorted_words[:top_n]]
        
        return keywords
    
    def process_pipeline(self, raw_data_path: str, output_path: str) -> List[Dict]:
        """
        Full processing pipeline
        
        Args:
            raw_data_path: Path to raw data
            output_path: Path to save processed data
            
        Returns:
            Processed documents
        """
        logger.info("Starting data processing pipeline...")
        
        # Load raw data
        raw_data = self.load_raw_data(raw_data_path)
        
        # Extract documents
        documents = self.extract_documents(raw_data)
        
        # Deduplicate
        documents = self.deduplicate(documents)
        
        # Enrich metadata
        documents = self.enrich_metadata(documents)
        
        # Save processed data
        self.save_documents(documents, output_path)
        
        self.processed_documents = documents
        
        logger.info(f"Processing complete: {len(documents)} documents")
        
        return documents
    
    def save_documents(self, documents: List[Dict], output_path: str):
        """Save processed documents"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(documents)} documents to {output_path}")
    
    def generate_statistics(self, documents: List[Dict]) -> Dict:
        """Generate statistics about processed data"""
        stats = {
            'total_documents': len(documents),
            'doc_types': {},
            'total_words': 0,
            'total_chars': 0,
            'avg_words_per_doc': 0,
            'categories': {},
            'sources': set()
        }
        
        for doc in documents:
            # Count by type
            doc_type = doc['doc_type']
            stats['doc_types'][doc_type] = stats['doc_types'].get(doc_type, 0) + 1
            
            # Total words/chars
            stats['total_words'] += doc['word_count']
            stats['total_chars'] += doc['char_count']
            
            # Categories
            for category in doc['metadata'].get('categories', []):
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Sources
            stats['sources'].add(doc['source_url'])
        
        # Calculate averages
        if documents:
            stats['avg_words_per_doc'] = stats['total_words'] / len(documents)
        
        stats['sources'] = list(stats['sources'])
        
        return stats


def process_scraped_data(
    raw_data_path: str = "data/raw/scraped_data.json",
    output_path: str = "data/processed/documents.json"
):
    """
    Main function to process scraped data
    
    Args:
        raw_data_path: Path to raw data
        output_path: Path to save processed data
    """
    processor = DataProcessor()
    
    documents = processor.process_pipeline(raw_data_path, output_path)
    
    # Generate and print statistics
    stats = processor.generate_statistics(documents)
    
    logger.info("=== Processing Statistics ===")
    logger.info(f"Total documents: {stats['total_documents']}")
    logger.info(f"Document types: {stats['doc_types']}")
    logger.info(f"Total words: {stats['total_words']}")
    logger.info(f"Average words per document: {stats['avg_words_per_doc']:.1f}")
    logger.info(f"Categories: {stats['categories']}")
    logger.info(f"Unique sources: {len(stats['sources'])}")
    
    return documents, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process scraped data")
    parser.add_argument('-i', '--input', default='data/raw/scraped_data.json',
                       help='Input raw data file')
    parser.add_argument('-o', '--output', default='data/processed/documents.json',
                       help='Output processed data file')
    
    args = parser.parse_args()
    
    process_scraped_data(args.input, args.output)
