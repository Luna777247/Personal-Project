"""
Conversation logger to MongoDB
"""
from typing import Dict, List, Optional
import logging
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationLogger:
    """Log conversations to MongoDB"""
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "banking_chatbot",
        collection_name: str = "conversations"
    ):
        """
        Initialize conversation logger
        
        Args:
            connection_string: MongoDB connection string
            database_name: Database name
            collection_name: Collection name
        """
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URL', 
            'mongodb://localhost:27017/'
        )
        self.database_name = database_name
        self.collection_name = collection_name
        
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            
            # Create indexes
            self._create_indexes()
            
            logger.info(f"Connected to MongoDB: {database_name}.{collection_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    def _create_indexes(self):
        """Create indexes for efficient querying"""
        if self.collection:
            self.collection.create_index("session_id")
            self.collection.create_index("user_id")
            self.collection.create_index("timestamp")
    
    def log_conversation(
        self,
        session_id: str,
        user_id: Optional[str],
        query: str,
        response: str,
        retrieved_docs: List[Dict],
        provider: str,
        model: Optional[str],
        timing: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Log conversation turn
        
        Args:
            session_id: Session ID
            user_id: User ID
            query: User query
            response: Bot response
            retrieved_docs: Retrieved documents
            provider: LLM provider
            model: Model name
            timing: Timing information
            metadata: Additional metadata
            
        Returns:
            Log ID or None if failed
        """
        if not self.collection:
            logger.warning("MongoDB not connected, skipping log")
            return None
        
        try:
            log_entry = {
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': datetime.now(),
                'query': query,
                'response': response,
                'retrieved_docs': [
                    {
                        'id': doc.get('id'),
                        'score': doc.get('score'),
                        'content_preview': doc.get('content', '')[:200]
                    }
                    for doc in retrieved_docs
                ],
                'provider': provider,
                'model': model,
                'timing': timing,
                'metadata': metadata or {}
            }
            
            result = self.collection.insert_one(log_entry)
            
            logger.info(f"Logged conversation: {result.inserted_id}")
            
            return str(result.inserted_id)
        
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")
            return None
    
    def get_conversations(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get conversation logs
        
        Args:
            session_id: Filter by session ID
            user_id: Filter by user ID
            limit: Maximum results
            
        Returns:
            List of conversation logs
        """
        if not self.collection:
            return []
        
        try:
            query = {}
            if session_id:
                query['session_id'] = session_id
            if user_id:
                query['user_id'] = user_id
            
            conversations = list(
                self.collection.find(query)
                .sort('timestamp', -1)
                .limit(limit)
            )
            
            # Convert ObjectId to string
            for conv in conversations:
                conv['_id'] = str(conv['_id'])
            
            return conversations
        
        except Exception as e:
            logger.error(f"Failed to get conversations: {e}")
            return []
    
    def get_statistics(self, days: int = 7) -> Dict:
        """
        Get conversation statistics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Statistics dictionary
        """
        if not self.collection:
            return {}
        
        try:
            from datetime import timedelta
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Total conversations
            total = self.collection.count_documents({
                'timestamp': {'$gte': start_date}
            })
            
            # Unique users
            unique_users = len(self.collection.distinct('user_id', {
                'timestamp': {'$gte': start_date}
            }))
            
            # Average response time
            pipeline = [
                {'$match': {'timestamp': {'$gte': start_date}}},
                {'$group': {
                    '_id': None,
                    'avg_retrieval': {'$avg': '$timing.retrieval'},
                    'avg_llm': {'$avg': '$timing.llm'},
                    'avg_total': {'$avg': '$timing.total'}
                }}
            ]
            
            timing_stats = list(self.collection.aggregate(pipeline))
            
            # Provider usage
            provider_pipeline = [
                {'$match': {'timestamp': {'$gte': start_date}}},
                {'$group': {
                    '_id': '$provider',
                    'count': {'$sum': 1}
                }}
            ]
            
            provider_usage = list(self.collection.aggregate(provider_pipeline))
            
            return {
                'total_conversations': total,
                'unique_users': unique_users,
                'avg_timing': timing_stats[0] if timing_stats else {},
                'provider_usage': {item['_id']: item['count'] for item in provider_usage},
                'period_days': days
            }
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


class FileLogger:
    """Fallback file-based logger"""
    
    def __init__(self, log_file: str = "logs/conversations.log"):
        """
        Initialize file logger
        
        Args:
            log_file: Path to log file
        """
        from pathlib import Path
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"File logger initialized: {log_file}")
    
    def log_conversation(
        self,
        session_id: str,
        user_id: Optional[str],
        query: str,
        response: str,
        **kwargs
    ):
        """Log conversation to file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_line = f"{datetime.now().isoformat()} | {session_id} | {user_id or 'anonymous'} | Q: {query} | A: {response}\n"
                f.write(log_line)
        
        except Exception as e:
            logger.error(f"Failed to write log: {e}")


# Create global logger instance
def create_logger(use_mongodb: bool = True) -> object:
    """
    Create appropriate logger
    
    Args:
        use_mongodb: Use MongoDB if True, else file logger
        
    Returns:
        Logger instance
    """
    if use_mongodb:
        try:
            return ConversationLogger()
        except Exception as e:
            logger.warning(f"Failed to create MongoDB logger, falling back to file: {e}")
            return FileLogger()
    else:
        return FileLogger()


if __name__ == "__main__":
    # Test logging
    conv_logger = ConversationLogger()
    
    log_id = conv_logger.log_conversation(
        session_id="test_session_123",
        user_id="user_001",
        query="Lãi suất tiết kiệm là bao nhiêu?",
        response="Lãi suất tiết kiệm MB Bank kỳ hạn 6 tháng là 6.0%/năm.",
        retrieved_docs=[
            {'id': 'doc_1', 'score': 0.95, 'content': 'Lãi suất...'}
        ],
        provider="openai",
        model="gpt-4-turbo-preview",
        timing={'retrieval': 0.15, 'llm': 1.2, 'total': 1.35}
    )
    
    print(f"Log ID: {log_id}")
    
    # Get statistics
    stats = conv_logger.get_statistics(days=7)
    print(f"Statistics: {stats}")
    
    conv_logger.close()
