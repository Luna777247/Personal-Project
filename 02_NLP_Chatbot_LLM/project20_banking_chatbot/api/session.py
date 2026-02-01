"""
Session management for conversations
"""
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    """Manage conversation sessions"""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize session manager
        
        Args:
            max_history: Maximum conversation turns to keep
        """
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create new session
        
        Args:
            user_id: Optional user ID
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'conversation': [],
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'metadata': {}
        }
        
        logger.info(f"Created session: {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None
        """
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, user_query: str, bot_response: str):
        """
        Update session with new turn
        
        Args:
            session_id: Session ID
            user_query: User query
            bot_response: Bot response
        """
        if session_id not in self.sessions:
            logger.warning(f"Session not found: {session_id}")
            return
        
        session = self.sessions[session_id]
        
        # Add user turn
        session['conversation'].append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now()
        })
        
        # Add assistant turn
        session['conversation'].append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(session['conversation']) > self.max_history * 2:
            session['conversation'] = session['conversation'][-(self.max_history * 2):]
        
        # Update last activity
        session['last_activity'] = datetime.now()
    
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history
        
        Args:
            session_id: Session ID
            limit: Maximum turns to return
            
        Returns:
            Conversation history
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        conversation = session['conversation']
        
        if limit:
            conversation = conversation[-limit * 2:]  # Each turn has user + assistant
        
        return conversation
    
    def delete_session(self, session_id: str):
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Clean up old sessions
        
        Args:
            max_age_hours: Maximum session age in hours
        """
        from datetime import timedelta
        
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            age = current_time - session['last_activity']
            
            if age > timedelta(hours=max_age_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_stats(self) -> Dict:
        """Get session statistics"""
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': len([s for s in self.sessions.values() 
                                   if (datetime.now() - s['last_activity']).seconds < 3600])
        }


# Global session manager instance
session_manager = SessionManager()
