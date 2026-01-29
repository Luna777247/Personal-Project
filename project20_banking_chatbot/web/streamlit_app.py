"""
Streamlit UI for Banking Chatbot
"""
import streamlit as st
import requests
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen2.5:latest"
DEFAULT_TOP_K = 5

# Page config
st.set_page_config(
    page_title="MB Bank Chatbot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #F5F5F5;
        border-left: 4px solid #43A047;
    }
    .retrieved-doc {
        background-color: #FFF9C4;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "retrieved_docs" not in st.session_state:
        st.session_state.retrieved_docs = {}
    
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = set()


def check_api_health() -> bool:
    """Check if API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def send_chat_message(
    query: str,
    provider: str,
    model: str,
    top_k: int,
    use_streaming: bool = False
) -> Optional[Dict]:
    """Send chat message to API"""
    try:
        endpoint = f"{API_BASE_URL}/chat/stream" if use_streaming else f"{API_BASE_URL}/chat"
        
        payload = {
            "query": query,
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.user_id,
            "provider": provider,
            "model": model,
            "top_k": top_k
        }
        
        if use_streaming:
            return {"streaming": True, "payload": payload}
        
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    except Exception as e:
        st.error(f"Lỗi kết nối API: {str(e)}")
        return None


def stream_chat_response(payload: Dict):
    """Stream chat response from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=60
        )
        
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                yield line.decode('utf-8')
    
    except Exception as e:
        yield f"[Lỗi: {str(e)}]"


def submit_feedback(
    query: str,
    response: str,
    rating: int,
    comment: str = ""
):
    """Submit user feedback"""
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment,
            "user_id": st.session_state.user_id
        }
        
        result = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=10)
        result.raise_for_status()
        
        return True
    
    except Exception as e:
        st.error(f"Lỗi gửi đánh giá: {str(e)}")
        return False


def display_chat_message(role: str, content: str, index: int = -1):
    """Display chat message"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 Bạn:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 MB Bank Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Show retrieved documents if available
        if index >= 0 and index in st.session_state.retrieved_docs:
            with st.expander("📚 Nguồn tham khảo", expanded=False):
                docs = st.session_state.retrieved_docs[index]
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"""
                    <div class="retrieved-doc">
                        <strong>Nguồn {i}</strong> (Điểm: {doc.get('score', 0):.3f})<br>
                        {doc.get('content', '')[:200]}...
                    </div>
                    """, unsafe_allow_html=True)


def display_feedback_buttons(message_index: int, query: str, response: str):
    """Display feedback rating buttons"""
    if message_index in st.session_state.feedback_given:
        st.success("✅ Đã đánh giá")
        return
    
    st.write("**Đánh giá câu trả lời:**")
    
    cols = st.columns(5)
    
    for i, col in enumerate(cols, 1):
        if col.button(f"⭐ {i}", key=f"rating_{message_index}_{i}"):
            if submit_feedback(query, response, i):
                st.session_state.feedback_given.add(message_index)
                st.success(f"Cảm ơn bạn đã đánh giá {i} sao!")
                st.rerun()


def sidebar_settings():
    """Sidebar with settings"""
    st.sidebar.markdown("## ⚙️ Cài đặt")
    
    # API health check
    is_healthy = check_api_health()
    
    if is_healthy:
        st.sidebar.success("🟢 API đang hoạt động")
    else:
        st.sidebar.error("🔴 Không thể kết nối API")
    
    st.sidebar.markdown("---")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "Nhà cung cấp LLM",
        options=["ollama", "openai"],
        index=0,
        help="Chọn nhà cung cấp mô hình ngôn ngữ"
    )
    
    # Model selection
    if provider == "ollama":
        model_options = ["qwen2.5:latest", "llama3.1:latest"]
    else:
        model_options = ["gpt-4o-mini", "gpt-4o"]
    
    model = st.sidebar.selectbox(
        "Mô hình",
        options=model_options,
        help="Chọn mô hình cụ thể"
    )
    
    # Top-k setting
    top_k = st.sidebar.slider(
        "Số tài liệu tham khảo (Top-K)",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        help="Số lượng tài liệu liên quan được sử dụng"
    )
    
    # Streaming toggle
    use_streaming = st.sidebar.checkbox(
        "Phản hồi theo thời gian thực",
        value=False,
        help="Hiển thị câu trả lời theo từng phần"
    )
    
    st.sidebar.markdown("---")
    
    # Session info
    st.sidebar.markdown("## 📊 Thông tin phiên")
    st.sidebar.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.sidebar.text(f"User ID: {st.session_state.user_id[:12]}...")
    st.sidebar.text(f"Số tin nhắn: {len(st.session_state.messages)}")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retrieved_docs = {}
        st.session_state.feedback_given = set()
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    return provider, model, top_k, use_streaming


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🏦 MB Bank Chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Trợ lý ảo tư vấn sản phẩm ngân hàng - Hỗ trợ tiếng Việt"
        "</p>",
        unsafe_allow_html=True
    )
    
    # Sidebar
    provider, model, top_k, use_streaming = sidebar_settings()
    
    # Main chat area
    st.markdown("---")
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        display_chat_message(message["role"], message["content"], i)
        
        # Show feedback buttons for assistant messages
        if message["role"] == "assistant" and i > 0:
            user_msg = st.session_state.messages[i-1]["content"]
            display_feedback_buttons(i, user_msg, message["content"])
            st.markdown("---")
    
    # Chat input
    query = st.chat_input(
        "Nhập câu hỏi của bạn về sản phẩm MB Bank...",
        key="chat_input"
    )
    
    if query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Display user message
        display_chat_message("user", query)
        
        # Get response
        with st.spinner("🤔 Đang suy nghĩ..."):
            if use_streaming:
                # Streaming response
                result = send_chat_message(query, provider, model, top_k, use_streaming=True)
                
                if result and result.get("streaming"):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in stream_chat_response(result["payload"]):
                        full_response += chunk
                        response_placeholder.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 MB Bank Assistant:</strong><br>
                            {full_response}
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.01)
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                    st.rerun()
            
            else:
                # Normal response
                result = send_chat_message(query, provider, model, top_k)
                
                if result:
                    response = result.get("response", "")
                    retrieved_docs = result.get("retrieved_docs", [])
                    
                    # Store retrieved docs
                    msg_index = len(st.session_state.messages)
                    st.session_state.retrieved_docs[msg_index] = retrieved_docs
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
    
    # Example questions
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Câu hỏi gợi ý:")
        
        example_questions = [
            "Lãi suất tiết kiệm MB Bank là bao nhiêu?",
            "Thẻ tín dụng MB Bank có những loại nào?",
            "Làm thế nào để mở tài khoản tại MB Bank?",
            "MB Bank có hỗ trợ vay mua nhà không?",
            "Phí chuyển khoản liên ngân hàng là bao nhiêu?"
        ]
        
        cols = st.columns(2)
        
        for i, question in enumerate(example_questions):
            col = cols[i % 2]
            if col.button(f"💬 {question}", use_container_width=True, key=f"example_{i}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                st.rerun()


if __name__ == "__main__":
    main()
