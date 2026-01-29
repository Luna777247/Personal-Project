# 🏦 MB Bank Chatbot - RAG-based Banking Assistant

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Trợ lý ảo tư vấn sản phẩm ngân hàng MB Bank sử dụng công nghệ RAG (Retrieval-Augmented Generation) với hỗ trợ tiếng Việt.

## 📋 Mục lục

- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [API Documentation](#-api-documentation)
- [Cấu hình](#-cấu-hình)
- [Docker Deployment](#-docker-deployment)
- [Đánh giá chất lượng](#-đánh-giá-chất-lượng)
- [Troubleshooting](#-troubleshooting)

## ✨ Tính năng

### Core Features
- 🤖 **RAG System**: Truy xuất tài liệu và sinh câu trả lời dựa trên ngữ cảnh
- 🌐 **Multi-LLM Support**: OpenAI (GPT-4o, GPT-4o-mini) và Ollama (Qwen2.5, Llama3.1)
- 🇻🇳 **Vietnamese Optimized**: Embeddings và LLM tối ưu cho tiếng Việt
- 💾 **Conversation Logging**: Lưu trữ lịch sử chat trong MongoDB
- 📊 **Quality Evaluation**: ROUGE, BERTScore, RAGAS metrics
- 🖥️ **Dual UI**: Streamlit và Gradio interfaces
- 🐳 **Docker Ready**: Containerized deployment với docker-compose

### Technical Features
- ⚡ **Fast API**: RESTful API với async support
- 🔄 **Streaming**: Real-time response streaming
- 📈 **Metrics**: Usage statistics và feedback tracking
- 🔍 **Session Management**: Stateful conversations
- 🎯 **Semantic Search**: ChromaDB vector database
- 📝 **Auto-evaluation**: 10% sampling cho quality assessment

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend Layer                       │
│  ┌──────────────────┐           ┌──────────────────┐       │
│  │  Streamlit UI    │           │   Gradio UI      │       │
│  │  (Port 8501)     │           │   (Port 7860)    │       │
│  └────────┬─────────┘           └─────────┬────────┘       │
└───────────┼───────────────────────────────┼────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
┌───────────────────────────┼────────────────────────────────┐
│                   FastAPI Backend (Port 8000)               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RAG Pipeline                            │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ │   │
│  │  │Embed │→ │Vector│→ │Retriev│→ │ LLM  │→ │Respon│ │   │
│  │  │der   │  │Store │  │er    │  │Router│  │se    │ │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Session Manager │ Conversation Logger │ Evaluator          │
└───────┬──────────┴──────────┬─────────┴────────────────────┘
        │                     │
┌───────┼─────────────────────┼────────────────────────────┐
│       │   Data Layer        │                             │
│  ┌────▼─────┐         ┌─────▼─────┐                      │
│  │ ChromaDB │         │  MongoDB   │                      │
│  │(Vectors) │         │  (Logs)    │                      │
│  └──────────┘         └────────────┘                      │
└──────────────────────────────────────────────────────────┘
```

## 📦 Yêu cầu hệ thống

### Minimum Requirements
- Python 3.9+
- 8GB RAM
- 10GB disk space

### Recommended
- Python 3.10+
- 16GB RAM
- 20GB disk space
- CUDA-capable GPU (optional, for faster embeddings)

### Dependencies
- FastAPI 0.109.0
- Sentence Transformers (paraphrase-multilingual-mpnet-base-v2)
- ChromaDB (vector database)
- MongoDB (conversation logging)
- OpenAI API (optional)
- Ollama (optional, for local LLM)

## 🚀 Cài đặt

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/banking-chatbot.git
cd banking-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
```bash
# Copy example file
cp .env.example .env

# Edit .env file
# Add your API keys:
# OPENAI_API_KEY=sk-your-key-here
# OLLAMA_HOST=http://localhost:11434
```

### 5. Initialize Database
```bash
# Start MongoDB (if local)
mongod --dbpath ./mongodb_data

# Or use Docker
docker run -d -p 27017:27017 --name mongodb mongo:7.0
```

### 6. Download Embedding Model
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')"
```

### 7. Scrape Data (Optional)
```bash
python src/data/mb_scraper.py
python src/data/data_processor.py
```

## 💻 Sử dụng

### Start FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Access API docs: http://localhost:8000/docs

### Start Streamlit UI
```bash
streamlit run web/streamlit_app.py --server.port 8501
```

Access UI: http://localhost:8501

### Start Gradio UI (Alternative)
```bash
python web/gradio_app.py
```

Access UI: http://localhost:7860

### Start Ollama (Local LLM)
```bash
# Install Ollama: https://ollama.ai

# Pull models
ollama pull qwen2.5:latest
ollama pull llama3.1:latest

# Start server (automatic on most systems)
ollama serve
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Chat (POST /chat)
Send a chat message and get response.

**Request:**
```json
{
  "query": "Lãi suất tiết kiệm MB Bank là bao nhiêu?",
  "session_id": "optional-uuid",
  "user_id": "user123",
  "provider": "ollama",
  "model": "qwen2.5:latest",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "Lãi suất tiết kiệm...",
  "response": "Lãi suất tiết kiệm MB Bank...",
  "session_id": "uuid",
  "retrieved_docs": [
    {
      "id": "doc1",
      "content": "...",
      "score": 0.85,
      "metadata": {}
    }
  ],
  "provider": "ollama",
  "timing": {
    "retrieval": 0.15,
    "llm": 2.3,
    "total": 2.45
  },
  "timestamp": "2025-12-11T..."
}
```

#### 2. Chat Stream (POST /chat/stream)
Stream chat response in real-time.

**Request:** Same as /chat

**Response:** Server-sent events stream

#### 3. Feedback (POST /feedback)
Submit user feedback.

**Request:**
```json
{
  "session_id": "uuid",
  "query": "...",
  "response": "...",
  "rating": 5,
  "comment": "Rất hữu ích!",
  "user_id": "user123"
}
```

#### 4. Health Check (GET /health)
Check service status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "...",
  "services": {
    "rag_pipeline": "ready",
    "session_manager": "ready",
    "feedback_store": "ready"
  }
}
```

#### 5. Conversation History (GET /conversation/{session_id})
Get conversation history.

**Query Params:** `limit=10`

#### 6. Metrics (GET /metrics)
Get system metrics.

#### 7. Providers (GET /providers)
Get available LLM providers.

## ⚙️ Cấu hình

### config.yaml
```yaml
embedding:
  model_name: "paraphrase-multilingual-mpnet-base-v2"
  device: "cuda"  # or "cpu"
  batch_size: 32

vector_store:
  type: "chroma"
  persist_directory: "data/embeddings/chroma_db"
  collection_name: "mb_bank_docs"

retriever:
  top_k: 5
  similarity_threshold: 0.7

llm:
  default_provider: "ollama"
  providers:
    openai:
      models:
        - "gpt-4o-mini"
        - "gpt-4o"
    ollama:
      models:
        - "qwen2.5:latest"
        - "llama3.1:latest"

evaluation:
  sample_rate: 0.1  # 10%
  min_samples: 10
  max_samples: 100
```

## 🐳 Docker Deployment

### Quick Start
```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services
- **API**: http://localhost:8000
- **Streamlit**: http://localhost:8501
- **Gradio**: http://localhost:7860
- **MongoDB**: localhost:27017
- **ChromaDB**: http://localhost:8001

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for details.

## 📊 Đánh giá chất lượng

### Metrics

#### 1. ROUGE Scores
Đo lường overlap giữa response và reference.
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

#### 2. BERTScore
Semantic similarity sử dụng PhoBERT.
- Precision, Recall, F1

#### 3. RAGAS Metrics
RAG-specific quality:
- Faithfulness: Response trung thực với context
- Answer Relevancy: Liên quan đến câu hỏi
- Context Precision: Chất lượng retrieved docs

### Run Evaluation
```python
from src.evaluation import create_auto_evaluator, create_logger

# Create logger and evaluator
conv_logger = create_logger(use_mongodb=True)
auto_eval = create_auto_evaluator(conv_logger, sample_rate=0.1)

# Run evaluation
report = auto_eval.run_evaluation(days=7)

print(f"Total Evaluated: {report['total_evaluated']}")
print(f"Low Quality Rate: {report['low_quality_rate']:.2%}")
```

## 🔧 Troubleshooting

### Issue: API không khởi động
**Solution:**
```bash
# Check logs
tail -f logs/app.log

# Verify dependencies
pip install -r requirements.txt

# Check ports
netstat -an | grep 8000
```

### Issue: Ollama connection failed
**Solution:**
```bash
# Start Ollama
ollama serve

# Test connection
curl http://localhost:11434/api/tags

# Set correct host in .env
OLLAMA_HOST=http://localhost:11434
```

### Issue: MongoDB connection error
**Solution:**
```bash
# Start MongoDB
docker run -d -p 27017:27017 mongo:7.0

# Or start local MongoDB
mongod --dbpath ./mongodb_data

# Update connection string in .env
MONGODB_URL=mongodb://localhost:27017
```

### Issue: Out of memory
**Solution:**
```bash
# Reduce batch size in config.yaml
embedding:
  batch_size: 16  # default: 32

# Use CPU instead of GPU
embedding:
  device: "cpu"
```

### Issue: Slow response time
**Solution:**
1. Reduce `top_k` value (5 → 3)
2. Use smaller LLM model (qwen2.5:latest → qwen2.5:7b)
3. Enable GPU for embeddings
4. Increase ChromaDB cache

## 📖 Documentation

- [RAG Architecture](docs/RAG_ARCHITECTURE.md)
- [Evaluation Methodology](docs/EVALUATION.md)
- [Docker Deployment](DOCKER_DEPLOYMENT.md)
- [Quick Start Guide](QUICKSTART.md)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- MB Bank for product information
- Sentence Transformers team
- FastAPI framework
- Streamlit and Gradio teams

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Email: support@example.com
- Documentation: https://docs.example.com

---

**Made with ❤️ using Python, FastAPI, and AI**
