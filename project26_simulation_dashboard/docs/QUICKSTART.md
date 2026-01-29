# Simulation Dashboard - Quick Start Guide

## Project Setup

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../requirements.txt
uvicorn app.main:app --reload
```

Backend will be available at: http://localhost:8000

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

Frontend will be available at: http://localhost:3000

## API Documentation

### REST Endpoints
- `GET /api/simulations` - List all active simulations
- `GET /api/simulations/{id}` - Get simulation details
- `POST /api/simulations` - Create new simulation
- `GET /api/data/{id}` - Get simulation data
- `GET /api/metrics/{id}` - Get simulation metrics

### WebSocket Connection
- `ws://localhost:8000/ws/simulations/{id}` - Real-time data streaming

## Docker Deployment

```bash
docker-compose up -d
```

This will start:
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- GAMA Server: http://localhost:8080

## Features Implemented

### Pollution Monitoring
- Real-time Air Quality Index (AQI)
- Pollutant level visualization
- Emission source tracking

### Traffic Analysis
- Vehicle count and average speed
- Congestion level indicator
- Public transport efficiency
- Traffic hotspot detection

### Weather Data
- Temperature, humidity, pressure monitoring
- Wind speed and direction visualization
- Precipitation tracking
- Weather alerts

### Agricultural Monitoring
- Crop yield tracking
- Soil moisture levels
- Irrigation efficiency
- Drought index calculation
- Affected area monitoring

## Integration with GAMA Models

### Data Flow
1. GAMA simulation runs with configured parameters
2. Simulation outputs data via WebSocket or REST API
3. Backend receives and stores data
4. Frontend subscribes via WebSocket for real-time updates
5. Dashboard visualizations update automatically

### Connecting Your GAMA Model
1. Modify GAMA model to output data in JSON format
2. Configure WebSocket connection to backend
3. Send data to: `ws://backend-host:8000/ws/simulations/{id}`

Example GAMA code:
```
// Connect to WebSocket
websocket ws <- tcp_open("backend-host", 8000);
// Send simulation data
tcp_send(ws, json_object(simulation_metrics));
```

## Project Structure
```
project26_simulation_dashboard/
├── backend/              # FastAPI backend
├── frontend/             # React frontend
├── docker/               # Docker configurations
├── data/                 # Sample data
├── docs/                 # Documentation
└── docker-compose.yml    # Compose file
```

## Next Steps
1. Connect real GAMA simulations
2. Add more visualization types
3. Implement data export features
4. Add authentication and authorization
5. Deploy to production environment