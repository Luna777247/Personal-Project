# Integration with GAMA Platform

## Overview
This guide explains how to connect your GAMA simulations to the dashboard.

## Step 1: Prepare Your GAMA Model

Your GAMA model should output simulation data in JSON format. Add the following to your model:

```gaml
// In your GAMA model
global {
    string backend_url <- "ws://localhost:8000";
    string simulation_id <- "my_simulation";
    websocket connection;
    
    init {
        // Connect to dashboard backend
        connection <- tcp_open(backend_url, 8000);
    }
    
    reflex send_data {
        // Prepare data
        map<string, float> metrics <- [
            "temperature" :: temperature_value,
            "rainfall" :: rainfall_value,
            "flooded_areas" :: flooded_areas_count
        ];
        
        // Send to dashboard
        map<string, unknown> message <- [
            "simulation_id" :: simulation_id,
            "cycle" :: cycle,
            "timestamp" :: current_date,
            "metrics" :: metrics
        ];
        
        tcp_send(connection, json_from(message));
    }
}
```

## Step 2: Configure Backend

Update `backend/app/config.py` with your GAMA connection details:

```python
class Settings(BaseSettings):
    gama_host: str = "localhost"
    gama_port: int = 8080
    gama_timeout: int = 30
```

## Step 3: Map Data to Dashboard

The dashboard expects data in this format:

```json
{
    "simulation_id": "climate_v1",
    "timestamp": "2024-01-01T12:00:00Z",
    "cycle": 100,
    "weather": {
        "temperature": 25.5,
        "humidity": 65.0,
        "precipitation": 5.2,
        "wind_speed": 3.2,
        "wind_direction": 180.0,
        "pressure": 1013.25
    },
    "pollution": {
        "air_quality_index": 85.0,
        "pm25": 35.5,
        "pm10": 52.3,
        "no2": 28.5,
        "so2": 15.2,
        "co": 1.2,
        "o3": 45.0
    },
    "traffic": {
        "vehicle_count": 250,
        "average_speed": 45.2,
        "congestion_level": 0.65,
        "public_transport_efficiency": 0.82
    },
    "agriculture": {
        "crop_yield": 85.5,
        "irrigation_efficiency": 0.78,
        "soil_moisture": 55.0,
        "drought_index": 0.35,
        "affected_area": 12.5
    }
}
```

## Step 4: Start Dashboard

Start the dashboard services:

```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm start

# Terminal 3: GAMA
# Run your GAMA model
```

## Step 5: View Data

Open `http://localhost:3000` in your browser. Your simulation data will appear in real-time.

## Troubleshooting

### Connection Issues
1. Verify GAMA model is running
2. Check backend is accessible at `localhost:8000`
3. Verify WebSocket port is open
4. Check browser console for errors

### Data Not Showing
1. Check data format matches expected schema
2. Verify simulation ID is correctly set
3. Check backend logs for parsing errors
4. Verify CORS settings in backend

## Performance Tips
- Send data every N cycles instead of every cycle
- Aggregate data when possible
- Limit historical data retention
- Use data compression for large values