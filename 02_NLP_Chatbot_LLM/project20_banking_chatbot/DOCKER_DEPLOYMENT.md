# Docker deployment scripts for Banking Chatbot

## Quick Start

### 1. Build and run all services
```bash
docker-compose up -d --build
```

### 2. View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f web
```

### 3. Check service status
```bash
docker-compose ps
```

### 4. Stop services
```bash
docker-compose down
```

### 5. Stop and remove volumes
```bash
docker-compose down -v
```

## Service URLs

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **Gradio UI**: http://localhost:7860
- **MongoDB**: localhost:27017
- **ChromaDB**: http://localhost:8001

## Environment Variables

Create `.env` file in project root:

```env
# OpenAI API (optional)
OPENAI_API_KEY=sk-your-api-key-here

# Ollama (if running locally)
OLLAMA_HOST=http://host.docker.internal:11434

# MongoDB (default)
MONGODB_URL=mongodb://mongodb:27017

# ChromaDB (default)
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

## Services

### MongoDB
- Purpose: Conversation logging and statistics
- Port: 27017
- Volume: `mongodb_data`
- Health check: MongoDB ping command

### ChromaDB
- Purpose: Vector embeddings storage
- Port: 8001 (mapped from 8000 inside container)
- Volume: `chromadb_data`
- Health check: `/api/v1/heartbeat`

### API (FastAPI)
- Purpose: Backend RAG service
- Port: 8000
- Dependencies: MongoDB, ChromaDB
- Health check: `/health` endpoint
- Volumes:
  - `./data:/app/data` - Data persistence
  - `./logs:/app/logs` - Log files
  - `./config:/app/config` - Configuration

### Web (Streamlit)
- Purpose: Web UI for chatbot
- Port: 8501
- Dependencies: API service
- Health check: `/_stcore/health`

### Gradio (Alternative UI)
- Purpose: Alternative web interface
- Port: 7860
- Dependencies: API service

## Troubleshooting

### Ollama connection
If using local Ollama, ensure it's accessible from Docker:
```bash
# Windows/Mac: Use host.docker.internal
OLLAMA_HOST=http://host.docker.internal:11434

# Linux: Use host IP
OLLAMA_HOST=http://172.17.0.1:11434
```

### MongoDB connection issues
```bash
# Check MongoDB logs
docker-compose logs mongodb

# Connect to MongoDB shell
docker exec -it banking_chatbot_mongodb mongosh
```

### API not starting
```bash
# Check API logs
docker-compose logs api

# Rebuild API image
docker-compose up -d --build api
```

### ChromaDB issues
```bash
# Check ChromaDB logs
docker-compose logs chromadb

# Test ChromaDB health
curl http://localhost:8001/api/v1/heartbeat
```

## Development Mode

### Mount source code for live reload
Modify `docker-compose.yml`:
```yaml
api:
  volumes:
    - ./src:/app/src
    - ./api:/app/api
```

Then restart:
```bash
docker-compose restart api
```

## Production Deployment

### Build optimized images
```bash
docker-compose build --no-cache
```

### Run with limited resources
```bash
docker-compose up -d --scale gradio=0  # Disable Gradio if not needed
```

### Backup volumes
```bash
# MongoDB
docker run --rm -v banking_chatbot_mongodb_data:/data -v $(pwd):/backup ubuntu tar czf /backup/mongodb_backup.tar.gz /data

# ChromaDB
docker run --rm -v banking_chatbot_chromadb_data:/data -v $(pwd):/backup ubuntu tar czf /backup/chromadb_backup.tar.gz /data
```

## Monitoring

### Resource usage
```bash
docker stats banking_chatbot_api banking_chatbot_web banking_chatbot_mongodb banking_chatbot_chromadb
```

### Service health
```bash
# API health
curl http://localhost:8000/health

# Streamlit health
curl http://localhost:8501/_stcore/health

# ChromaDB health
curl http://localhost:8001/api/v1/heartbeat
```
