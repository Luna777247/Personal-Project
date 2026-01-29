# Simulation Dashboard - Web-based Visualization for ABM Models

## Project Overview
Interactive dashboard for real-time visualization of Agent-Based Model (ABM) simulations. Built with React and D3.js for frontend, Python FastAPI for backend, integrating with GAMA Platform simulations.

## Features
- **Real-time Data Streaming**: Live updates from GAMA/Python simulation backends
- **Interactive Visualizations**: D3.js charts for pollution, traffic, weather, and agricultural metrics
- **Multi-model Support**: Compatible with climate change, urban mobility, and environmental simulations
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack
- **Frontend**: React.js, D3.js, Material-UI
- **Backend**: Python FastAPI, WebSocket for real-time data
- **Data Integration**: REST API + WebSocket connections to GAMA models
- **Visualization**: D3.js for interactive charts and maps

## Project Structure
```
project26_simulation_dashboard/
├── frontend/                 # React application
│   ├── public/
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   ├── pages/           # Dashboard pages
│   │   ├── services/        # API integration
│   │   ├── utils/           # Helper functions
│   │   └── visualizations/  # D3.js charts
├── backend/                  # Python FastAPI server
│   ├── app/
│   │   ├── api/             # API endpoints
│   │   ├── models/          # Data models
│   │   ├── services/        # Business logic
│   │   └── websocket/       # Real-time connections
├── data/                     # Sample data and configurations
├── docs/                     # Documentation
├── docker/                   # Docker configurations
└── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- GAMA Platform (for simulation backend)

### Installation

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd frontend
npm install
```

### Running the Application

#### Start Backend
```bash
cd backend
uvicorn app.main:app --reload
```

#### Start Frontend
```bash
cd frontend
npm start
```

### Integration with GAMA
1. Configure GAMA model to output data via TCP/WebSocket
2. Update backend configuration to connect to GAMA simulation
3. Dashboard will automatically receive and display real-time data

## Dashboard Components

### Pollution Monitoring
- Air quality indices
- Emission sources visualization
- Temporal pollution patterns

### Traffic Analysis
- Vehicle flow visualization
- Congestion hotspots
- Public transport efficiency

### Weather Data
- Temperature, humidity, precipitation
- Weather pattern animations
- Climate change indicators

### Agricultural Metrics
- Crop yield monitoring
- Irrigation efficiency
- Drought impact visualization

## API Endpoints

### REST API
- `GET /api/simulations` - List active simulations
- `GET /api/data/{simulation_id}` - Get simulation data
- `POST /api/simulations` - Start new simulation

### WebSocket
- `/ws/simulations/{id}` - Real-time data streaming

## Configuration
Update `backend/app/config.py` with:
- GAMA connection settings
- Database credentials (if needed)
- API keys for external data sources

## Development

### Adding New Visualizations
1. Create component in `frontend/src/visualizations/`
2. Implement D3.js logic
3. Connect to WebSocket data stream
4. Add to dashboard layout

### Backend Extensions
1. Add new endpoints in `backend/app/api/`
2. Implement data processing in services
3. Update WebSocket handlers

## Deployment

### Docker Deployment (Recommended)
```bash
# Build and run with Docker Compose
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Deployment

#### Backend Deployment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### Frontend Deployment
```bash
cd frontend
npm install
npm run build
npm install -g serve
serve -s build -l 3000
```

### Production Deployment
```bash
# Backend with Gunicorn
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Frontend with Nginx (after build)
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/frontend/build;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Integration Examples

### Connecting to Climate ABM
```python
# In GAMA model
socket <- tcp_open("localhost", 8080);
tcp_send(socket, simulation_data);
```

### Real-time Updates
```javascript
// In React component
useEffect(() => {
  const ws = new WebSocket('ws://localhost:8000/ws/simulations/climate');
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateVisualization(data);
  };
}, []);
```

## CV Highlights
- **SIMPLE Integration**: VR-ready visualization framework
- **Moov'Hanoi Compatible**: Urban mobility data visualization
- **GAMA Tools**: Native integration with GAMA visualization ecosystem
- **Real-time Systems**: WebSocket implementation for live data streaming
- **Modern Web Tech**: React + D3.js for interactive dashboards

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
MIT License