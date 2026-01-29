"""
Gradio UI for Banking Chatbot
"""
import gradio as gr
import requests
import uuid
from typing import List, Tuple, Optional
import json

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "qwen2.5:latest"
DEFAULT_TOP_K = 5

# Session state
session_data = {
    "session_id": str(uuid.uuid4()),
    "user_id": f"user_{uuid.uuid4().hex[:8]}",
    "conversation_history": []
}


def check_api_health() -> str:
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return f"✅ API đang hoạt động\n\nTrạng thái dịch vụ:\n{json.dumps(data.get('services', {}), indent=2, ensure_ascii=False)}"
        else:
            return "⚠️ API không phản hồi đúng"
    except Exception as e:
        return f"❌ Không thể kết nối API: {str(e)}"


def send_message(
    query: str,
    provider: str,
    model: str,
    top_k: int,
    history: List[Tuple[str, str]]
) -> Tuple[List[Tuple[str, str]], str]:
    """Send message and get response"""
    if not query.strip():
        return history, ""
    
    try:
        # Prepare payload
        payload = {
            "query": query,
            "session_id": session_data["session_id"],
            "user_id": session_data["user_id"],
            "provider": provider.lower(),
            "model": model,
            "top_k": int(top_k)
        }
        
        # Send request
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract response
        answer = result.get("response", "")
        retrieved_docs = result.get("retrieved_docs", [])
        timing = result.get("timing", {})
        
        # Format response with sources
        formatted_answer = answer
        
        if retrieved_docs:
            formatted_answer += "\n\n📚 **Nguồn tham khảo:**\n"
            for i, doc in enumerate(retrieved_docs[:3], 1):
                score = doc.get("score", 0)
                content = doc.get("content", "")[:150]
                formatted_answer += f"\n{i}. (Điểm: {score:.3f}) {content}..."
        
        formatted_answer += f"\n\n⏱️ Thời gian xử lý: {timing.get('total', 0):.2f}s"
        
        # Update history
        history.append((query, formatted_answer))
        
        # Store in session
        session_data["conversation_history"] = history
        
        return history, ""
    
    except Exception as e:
        error_msg = f"❌ Lỗi: {str(e)}"
        history.append((query, error_msg))
        return history, ""


def submit_feedback(
    rating: int,
    comment: str,
    history: List[Tuple[str, str]]
) -> str:
    """Submit user feedback"""
    if not history:
        return "⚠️ Chưa có cuộc hội thoại nào để đánh giá"
    
    # Get last interaction
    last_query, last_response = history[-1]
    
    try:
        payload = {
            "session_id": session_data["session_id"],
            "query": last_query,
            "response": last_response,
            "rating": rating,
            "comment": comment,
            "user_id": session_data["user_id"]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/feedback",
            json=payload,
            timeout=10
        )
        
        response.raise_for_status()
        
        return f"✅ Cảm ơn bạn đã đánh giá {rating} sao!"
    
    except Exception as e:
        return f"❌ Lỗi gửi đánh giá: {str(e)}"


def clear_conversation():
    """Clear conversation history"""
    session_data["session_id"] = str(uuid.uuid4())
    session_data["conversation_history"] = []
    return [], ""


def get_session_info() -> str:
    """Get current session information"""
    return f"""
**Thông tin phiên làm việc:**

- Session ID: `{session_data['session_id'][:16]}...`
- User ID: `{session_data['user_id']}`
- Số tin nhắn: {len(session_data['conversation_history'])}
"""


def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="MB Bank Chatbot") as demo:
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🏦 MB Bank Chatbot</h1>
            <p>Trợ lý ảo tư vấn sản phẩm ngân hàng - Hỗ trợ tiếng Việt</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Cuộc hội thoại",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Nhập câu hỏi",
                        placeholder="Nhập câu hỏi của bạn về sản phẩm MB Bank...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Gửi 📤", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ Xóa lịch sử", size="sm")
                
                # Example questions
                gr.Markdown("### 💡 Câu hỏi gợi ý:")
                
                with gr.Row():
                    example_btn1 = gr.Button(
                        "Lãi suất tiết kiệm là bao nhiêu?",
                        size="sm"
                    )
                    example_btn2 = gr.Button(
                        "Thẻ tín dụng có những loại nào?",
                        size="sm"
                    )
                
                with gr.Row():
                    example_btn3 = gr.Button(
                        "Làm thế nào để mở tài khoản?",
                        size="sm"
                    )
                    example_btn4 = gr.Button(
                        "Phí chuyển khoản là bao nhiêu?",
                        size="sm"
                    )
            
            # Right column - Settings & Feedback
            with gr.Column(scale=1):
                # Settings
                gr.Markdown("## ⚙️ Cài đặt")
                
                provider_dropdown = gr.Dropdown(
                    label="Nhà cung cấp LLM",
                    choices=["ollama", "openai"],
                    value="ollama"
                )
                
                model_dropdown = gr.Dropdown(
                    label="Mô hình",
                    choices=["qwen2.5:latest", "llama3.1:latest"],
                    value="qwen2.5:latest"
                )
                
                top_k_slider = gr.Slider(
                    label="Số tài liệu tham khảo (Top-K)",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1
                )
                
                # Update model choices based on provider
                def update_models(provider):
                    if provider == "ollama":
                        return gr.Dropdown(choices=["qwen2.5:latest", "llama3.1:latest"])
                    else:
                        return gr.Dropdown(choices=["gpt-4o-mini", "gpt-4o"])
                
                provider_dropdown.change(
                    fn=update_models,
                    inputs=[provider_dropdown],
                    outputs=[model_dropdown]
                )
                
                gr.Markdown("---")
                
                # Feedback
                gr.Markdown("## ⭐ Đánh giá")
                
                rating_slider = gr.Slider(
                    label="Điểm đánh giá",
                    minimum=1,
                    maximum=5,
                    value=5,
                    step=1
                )
                
                comment_input = gr.Textbox(
                    label="Nhận xét (tùy chọn)",
                    placeholder="Nhập nhận xét của bạn...",
                    lines=3
                )
                
                feedback_btn = gr.Button("Gửi đánh giá 📝", variant="secondary")
                feedback_output = gr.Textbox(label="Kết quả", interactive=False)
                
                gr.Markdown("---")
                
                # Session info
                session_info = gr.Markdown(get_session_info())
                refresh_info_btn = gr.Button("🔄 Làm mới thông tin", size="sm")
                
                gr.Markdown("---")
                
                # API health
                health_output = gr.Textbox(
                    label="Trạng thái API",
                    value=check_api_health(),
                    interactive=False,
                    lines=8
                )
                health_btn = gr.Button("🔍 Kiểm tra API", size="sm")
        
        # Event handlers
        send_btn.click(
            fn=send_message,
            inputs=[query_input, provider_dropdown, model_dropdown, top_k_slider, chatbot],
            outputs=[chatbot, query_input]
        )
        
        query_input.submit(
            fn=send_message,
            inputs=[query_input, provider_dropdown, model_dropdown, top_k_slider, chatbot],
            outputs=[chatbot, query_input]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, query_input]
        )
        
        feedback_btn.click(
            fn=submit_feedback,
            inputs=[rating_slider, comment_input, chatbot],
            outputs=[feedback_output]
        )
        
        refresh_info_btn.click(
            fn=get_session_info,
            outputs=[session_info]
        )
        
        health_btn.click(
            fn=check_api_health,
            outputs=[health_output]
        )
        
        # Example buttons
        def set_query(text):
            return text
        
        example_btn1.click(
            fn=set_query,
            inputs=[gr.State("Lãi suất tiết kiệm MB Bank là bao nhiêu?")],
            outputs=[query_input]
        )
        
        example_btn2.click(
            fn=set_query,
            inputs=[gr.State("Thẻ tín dụng MB Bank có những loại nào?")],
            outputs=[query_input]
        )
        
        example_btn3.click(
            fn=set_query,
            inputs=[gr.State("Làm thế nào để mở tài khoản tại MB Bank?")],
            outputs=[query_input]
        )
        
        example_btn4.click(
            fn=set_query,
            inputs=[gr.State("Phí chuyển khoản liên ngân hàng là bao nhiêu?")],
            outputs=[query_input]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
