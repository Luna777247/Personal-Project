"""
Prompt templates and management
"""
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptManager:
    """Manage prompts for chatbot"""
    
    def __init__(self):
        """Initialize prompt manager"""
        self.system_prompts = self._load_system_prompts()
        self.context_templates = self._load_context_templates()
    
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts"""
        return {
            'default': """Bạn là trợ lý ảo của MB Bank - một trong những ngân hàng thương mại hàng đầu Việt Nam.

Nhiệm vụ của bạn:
- Trả lời các câu hỏi của khách hàng về sản phẩm và dịch vụ ngân hàng
- Cung cấp thông tin về tiết kiệm, thẻ, vay, lãi suất, quy trình
- Giải thích các sản phẩm một cách rõ ràng, dễ hiểu
- Luôn cung cấp thông tin chính xác dựa trên dữ liệu được cung cấp
- Không được bịa đặt thông tin không có trong dữ liệu
- Nếu không biết câu trả lời, hãy thẳng thắn thừa nhận và gợi ý khách hàng liên hệ hotline

Phong cách giao tiếp:
- Lịch sự, thân thiện, chuyên nghiệp
- Sử dụng tiếng Việt chuẩn
- Tránh sử dụng thuật ngữ quá phức tạp
- Đặt lợi ích khách hàng lên hàng đầu""",
            
            'savings': """Bạn là chuyên gia tư vấn tiết kiệm của MB Bank.

Bạn có kiến thức sâu về:
- Các sản phẩm tiết kiệm của MB Bank
- Lãi suất tiết kiệm theo từng kỳ hạn
- Điều kiện và quyền lợi của từng loại tiết kiệm
- Cách tính lãi suất
- Quy trình mở sổ tiết kiệm

Hãy tư vấn chi tiết và so sánh các sản phẩm để giúp khách hàng chọn được gói tiết kiệm phù hợp nhất.""",
            
            'loans': """Bạn là chuyên gia tư vấn vay vốn của MB Bank.

Bạn có kiến thức về:
- Các sản phẩm vay của MB Bank (vay tiêu dùng, vay mua nhà, vay ô tô...)
- Lãi suất vay và các ưu đãi
- Điều kiện và hồ sơ vay vốn
- Quy trình phê duyệt và giải ngân
- Cách tính lãi suất và kế hoạch trả nợ

Hãy tư vấn cụ thể về điều kiện, thủ tục và giúp khách hàng hiểu rõ cam kết khi vay.""",
            
            'cards': """Bạn là chuyên gia tư vấn thẻ của MB Bank.

Bạn có kiến thức về:
- Các loại thẻ tín dụng và thẻ ghi nợ
- Hạn mức, lãi suất, phí thường niên
- Chương trình ưu đãi và hoàn tiền
- Quyền lợi và bảo hiểm kèm theo
- Quy trình đăng ký và kích hoạt thẻ

Hãy giới thiệu và so sánh các loại thẻ để khách hàng chọn được thẻ phù hợp với nhu cầu."""
        }
    
    def _load_context_templates(self) -> Dict[str, str]:
        """Load context templates"""
        return {
            'default': """Dựa trên thông tin sau đây về sản phẩm/dịch vụ của MB Bank:

{context}

Hãy trả lời câu hỏi của khách hàng một cách chính xác và thân thiện.""",
            
            'with_sources': """Dựa trên thông tin sau đây về sản phẩm/dịch vụ của MB Bank:

{context}

Nguồn: {sources}

Hãy trả lời câu hỏi của khách hàng một cách chính xác và thân thiện. Nếu cần, bạn có thể tham khảo nguồn thông tin.""",
            
            'no_context': """Bạn không có đủ thông tin để trả lời câu hỏi này một cách chính xác. 

Hãy lịch sự thông báo với khách hàng và gợi ý họ:
- Liên hệ hotline MB Bank: 1900 545 415
- Truy cập website: www.mbbank.com.vn
- Đến chi nhánh MB Bank gần nhất để được tư vấn chi tiết"""
        }
    
    def get_system_prompt(self, prompt_type: str = 'default') -> str:
        """
        Get system prompt
        
        Args:
            prompt_type: Type of prompt (default/savings/loans/cards)
            
        Returns:
            System prompt
        """
        return self.system_prompts.get(prompt_type, self.system_prompts['default'])
    
    def get_context_template(self, template_type: str = 'default') -> str:
        """
        Get context template
        
        Args:
            template_type: Template type
            
        Returns:
            Context template
        """
        return self.context_templates.get(template_type, self.context_templates['default'])
    
    def format_context_prompt(
        self,
        context: str,
        query: str,
        template_type: str = 'default',
        sources: Optional[List[str]] = None
    ) -> str:
        """
        Format context prompt
        
        Args:
            context: Retrieved context
            query: User query
            template_type: Template type
            sources: Source URLs
            
        Returns:
            Formatted prompt
        """
        template = self.get_context_template(template_type)
        
        # Format template
        if template_type == 'with_sources' and sources:
            sources_text = "\n".join(f"- {src}" for src in sources)
            prompt = template.format(context=context, sources=sources_text)
        elif template_type == 'no_context':
            prompt = template
        else:
            prompt = template.format(context=context)
        
        # Add query
        prompt += f"\n\nCâu hỏi: {query}"
        
        return prompt
    
    def build_messages(
        self,
        query: str,
        context: str,
        prompt_type: str = 'default',
        conversation_history: Optional[List[Dict]] = None,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Build complete message list
        
        Args:
            query: User query
            context: Retrieved context
            prompt_type: System prompt type
            conversation_history: Previous conversation
            sources: Source URLs
            
        Returns:
            List of messages
        """
        messages = []
        
        # Add system prompt
        system_prompt = self.get_system_prompt(prompt_type)
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            for turn in conversation_history[-5:]:  # Last 5 turns
                messages.append({
                    "role": turn.get('role', 'user'),
                    "content": turn.get('content', '')
                })
        
        # Add current query with context
        if context:
            user_prompt = self.format_context_prompt(
                context, query, 'default', sources
            )
        else:
            user_prompt = self.format_context_prompt('', query, 'no_context')
        
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def detect_intent(self, query: str) -> str:
        """
        Detect user intent from query
        
        Args:
            query: User query
            
        Returns:
            Intent type (savings/loans/cards/default)
        """
        query_lower = query.lower()
        
        # Savings keywords
        if any(kw in query_lower for kw in ['tiết kiệm', 'gửi tiền', 'lãi suất gửi']):
            return 'savings'
        
        # Loans keywords
        if any(kw in query_lower for kw in ['vay', 'tín dụng', 'vay vốn', 'mua nhà', 'mua xe']):
            return 'loans'
        
        # Cards keywords
        if any(kw in query_lower for kw in ['thẻ', 'card', 'credit', 'debit']):
            return 'cards'
        
        return 'default'
    
    def add_custom_prompt(self, name: str, prompt: str, prompt_type: str = 'system'):
        """
        Add custom prompt
        
        Args:
            name: Prompt name
            prompt: Prompt text
            prompt_type: Type (system/context)
        """
        if prompt_type == 'system':
            self.system_prompts[name] = prompt
        elif prompt_type == 'context':
            self.context_templates[name] = prompt
        
        logger.info(f"Added custom {prompt_type} prompt: {name}")


class ConversationFormatter:
    """Format conversation history"""
    
    @staticmethod
    def format_history(
        conversation: List[Dict],
        max_turns: int = 5
    ) -> List[Dict[str, str]]:
        """
        Format conversation history for LLM
        
        Args:
            conversation: Full conversation
            max_turns: Maximum turns to include
            
        Returns:
            Formatted messages
        """
        # Get recent turns
        recent = conversation[-max_turns * 2:] if len(conversation) > max_turns * 2 else conversation
        
        messages = []
        for turn in recent:
            messages.append({
                "role": turn.get('role', 'user'),
                "content": turn.get('content', '')
            })
        
        return messages
    
    @staticmethod
    def add_turn(
        conversation: List[Dict],
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Add turn to conversation
        
        Args:
            conversation: Current conversation
            role: Role (user/assistant)
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Updated conversation
        """
        turn = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        conversation.append(turn)
        
        return conversation


if __name__ == "__main__":
    # Example usage
    manager = PromptManager()
    
    # Test intent detection
    queries = [
        "Lãi suất tiết kiệm MB Bank là bao nhiêu?",
        "Tôi muốn vay mua nhà",
        "Thẻ tín dụng MB có ưu đãi gì không?"
    ]
    
    for query in queries:
        intent = manager.detect_intent(query)
        print(f"Query: {query}")
        print(f"Intent: {intent}\n")
    
    # Test prompt building
    context = "Lãi suất tiết kiệm MB Bank kỳ hạn 6 tháng là 6.0%/năm."
    query = "Gửi tiết kiệm 6 tháng lãi bao nhiêu?"
    
    messages = manager.build_messages(query, context, 'savings')
    
    print("Generated messages:")
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:100]}...")
