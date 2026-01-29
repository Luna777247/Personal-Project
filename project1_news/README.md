# 📰 Real-time News Data Pipeline - Big Data Project

A sophisticated real-time news aggregation and streaming platform that demonstrates big data pipeline concepts through continuous news crawling, processing, and real-time WebSocket distribution.

## 🌟 Features

### 🚀 Core Pipeline Features
- **Real-time News Crawling**: Continuous scraping from multiple news sources
- **WebSocket Streaming**: Live data distribution to connected clients
- **Big Data Processing**: Handles streaming news data at scale
- **Automated Scheduling**: Background crawling with configurable intervals
- **Error Resilience**: Robust error handling and recovery mechanisms

### 🎨 Frontend Features
- **Real-time Updates**: Live news feed with instant updates
- **Smooth Animations**: Framer Motion powered transitions
- **Responsive Design**: Mobile-first approach with modern UI
- **Vietnamese Interface**: Localized user experience

### 🔧 Technical Stack
- **Backend**: FastAPI, Newspaper3k, WebSockets, AsyncIO
- **Frontend**: React, Framer Motion, WebSocket API
- **Data Processing**: Pandas, NumPy for news analysis
- **Real-time Communication**: WebSocket protocol for live streaming

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   News Sources  │───▶│   FastAPI       │───▶│   WebSocket     │
│   (Websites)    │    │   Backend       │    │   Clients       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Data          │
                       │   Processing    │
                       │   & Storage     │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r ../requirements.txt

# Run the server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health

## 📊 Data Pipeline Flow

1. **Data Ingestion**: Continuous crawling from news websites
2. **Data Processing**: Article parsing and metadata extraction
3. **Real-time Streaming**: WebSocket distribution to clients
4. **Client Rendering**: Live updates in the React frontend

## 🔧 Configuration

### News Sources
Edit `backend/app.py` to modify crawling sources:

```python
URLS = [
    "https://www.thehindu.com/news/cities/Madurai/",
    "https://timesofindia.indatimes.com/city/",
    # Add more sources here
]
```

### Crawling Interval
Modify the sleep interval in the crawling loop:

```python
await asyncio.sleep(60)  # Crawl every 60 seconds
```

## 🎯 Big Data Concepts Demonstrated

- **Streaming Data Processing**: Real-time news ingestion
- **Distributed Systems**: WebSocket-based client-server architecture
- **Scalability**: Asynchronous processing with FastAPI
- **Fault Tolerance**: Error handling and recovery
- **Real-time Analytics**: Live data streaming and visualization

## 📈 Performance Metrics

- **Crawling Speed**: ~1-2 seconds per article
- **Concurrent Connections**: Supports multiple WebSocket clients
- **Memory Usage**: Optimized for continuous operation
- **Error Recovery**: Automatic retry mechanisms

## 🔍 Monitoring & Debugging

### Backend Logs
```bash
# View real-time logs
tail -f backend/logs/app.log
```

### WebSocket Testing
```javascript
// Test WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

## 🚀 Advanced Features

### Custom News Sources
Add new news sources by extending the URLS list with proper selectors.

### Data Filtering
Implement content filtering based on keywords, categories, or sentiment.

### Analytics Dashboard
Add metrics collection for crawled articles, user engagement, etc.

### Multi-language Support
Extend to support multiple languages and regions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Newspaper3k**: For robust news article extraction
- **FastAPI**: For high-performance async web framework
- **React**: For modern frontend development
- **Framer Motion**: For smooth animations

---

**🎉 Experience the power of real-time big data pipelines with this live news streaming platform!**