# 🚀 Quick Start Guide - MB Bank Chatbot

Get started with the Banking Chatbot in 5 minutes!

## Prerequisites

- Python 3.9+
- 8GB RAM minimum
- Internet connection

## Step 1: Clone and Setup (1 min)

```bash
# Clone repository
git clone https://github.com/yourusername/banking-chatbot.git
cd banking-chatbot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment (1 min)

```bash
# Copy environment file
cp .env.example .env
```

Edit `.env` file:
```env
# For OpenAI (optional)
OPENAI_API_KEY=sk-your-key-here

# For Ollama (recommended for local)
OLLAMA_HOST=http://localhost:11434

# MongoDB (automatic with Docker)
MONGODB_URL=mongodb://localhost:27017
```

## Step 3: Start Services (2 min)

### Option A: Docker (Recommended)
```bash
# Start all services
docker-compose up -d

# Wait 30 seconds for initialization
# Access Streamlit UI: http://localhost:8501
```

### Option B: Local Development
```bash
# Terminal 1: Start MongoDB
docker run -d -p 27017:27017 mongo:7.0

# Terminal 2: Install and start Ollama
# Download from https://ollama.ai
ollama pull qwen2.5:latest
ollama serve

# Terminal 3: Start API
uvicorn api.main:app --reload

# Terminal 4: Start UI
streamlit run web/streamlit_app.py
```

## Step 4: Test the Chatbot (1 min)

Open browser: http://localhost:8501

Try example questions:
1. "Lãi suất tiết kiệm MB Bank là bao nhiêu?"
2. "Thẻ tín dụng có những loại nào?"
3. "Làm thế nào để mở tài khoản?"

## Verify Installation

### Check API Health
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "rag_pipeline": "ready",
    "session_manager": "ready",
    "feedback_store": "ready"
  }
}
```

### Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Lãi suất tiết kiệm là bao nhiêu?",
    "provider": "ollama",
    "model": "qwen2.5:latest"
  }'
```

## Common Issues

### Issue: Port 8000 already in use
```bash
# Find and kill process
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

### Issue: Ollama not found
```bash
# Install Ollama
# Windows: Download from https://ollama.ai
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Mac: brew install ollama

# Pull model
ollama pull qwen2.5:latest
```

### Issue: MongoDB connection failed
```bash
# Start MongoDB with Docker
docker run -d -p 27017:27017 --name mongodb mongo:7.0

# Or install MongoDB locally
# Windows: https://www.mongodb.com/try/download/community
```

## Next Steps

1. **Customize Data**: Add your own banking documents to `data/raw/`
2. **Configure LLM**: Edit `config/config.yaml` for different models
3. **Enable Evaluation**: Run `python -m src.evaluation.evaluator` for quality metrics
4. **Deploy**: Use `docker-compose.yml` for production deployment

## Useful Commands

```bash
# View logs
docker-compose logs -f api

# Restart services
docker-compose restart

# Stop all services
docker-compose down

# Update code and restart
git pull
docker-compose up -d --build
```

## Support

- 📖 Full documentation: [README.md](README.md)
- 🐳 Docker guide: [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
- 🏗️ Architecture: [docs/RAG_ARCHITECTURE.md](docs/RAG_ARCHITECTURE.md)

---

**Congratulations! 🎉 Your Banking Chatbot is ready to use!**
