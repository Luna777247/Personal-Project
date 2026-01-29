"""
Simulation Dashboard Backend
FastAPI application for serving ABM simulation data and real-time updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uvicorn

from .config import settings
from .models import SimulationData, SimulationStatus
from .services import SimulationService
from .websocket import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Simulation Dashboard API",
    description="Backend API for ABM simulation visualization dashboard",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
simulation_service = SimulationService()
connection_manager = ConnectionManager()

# WebSocket connections for real-time updates
active_connections: Dict[str, WebSocket] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Simulation Dashboard Backend")
    await simulation_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Simulation Dashboard Backend")
    await simulation_service.cleanup()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simulation Dashboard API", "status": "running"}

@app.get("/api/simulations")
async def get_simulations():
    """Get list of active simulations"""
    simulations = await simulation_service.get_active_simulations()
    return {"simulations": simulations}

@app.get("/api/simulations/{simulation_id}")
async def get_simulation(simulation_id: str):
    """Get simulation details"""
    simulation = await simulation_service.get_simulation(simulation_id)
    if not simulation:
        return {"error": "Simulation not found"}, 404
    return simulation

@app.post("/api/simulations")
async def create_simulation(config: dict):
    """Create new simulation"""
    simulation_id = await simulation_service.create_simulation(config)
    return {"simulation_id": simulation_id, "status": "created"}

@app.get("/api/data/{simulation_id}")
async def get_simulation_data(simulation_id: str, limit: Optional[int] = 100):
    """Get simulation data"""
    data = await simulation_service.get_simulation_data(simulation_id, limit)
    return {"data": data}

@app.websocket("/ws/simulations/{simulation_id}")
async def simulation_websocket(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time simulation data"""
    await connection_manager.connect(websocket, simulation_id)

    try:
        # Send initial data
        initial_data = await simulation_service.get_simulation_data(simulation_id, limit=10)
        await websocket.send_json({
            "type": "initial_data",
            "data": initial_data,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Listen for client messages and send updates
        while True:
            # Wait for client message (optional)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Process client commands if needed
            except asyncio.TimeoutError:
                pass

            # Send real-time updates
            latest_data = await simulation_service.get_latest_data(simulation_id)
            if latest_data:
                await websocket.send_json({
                    "type": "update",
                    "data": latest_data,
                    "timestamp": datetime.utcnow().isoformat()
                })

            await asyncio.sleep(0.1)  # Small delay to prevent overwhelming

    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, simulation_id)
    except Exception as e:
        logger.error(f"WebSocket error for simulation {simulation_id}: {e}")
        connection_manager.disconnect(websocket, simulation_id)

@app.get("/api/metrics/{simulation_id}")
async def get_simulation_metrics(simulation_id: str):
    """Get aggregated metrics for simulation"""
    metrics = await simulation_service.get_simulation_metrics(simulation_id)
    return {"metrics": metrics}

@app.post("/api/simulations/{simulation_id}/control")
async def control_simulation(simulation_id: str, command: dict):
    """Send control commands to simulation"""
    result = await simulation_service.send_command(simulation_id, command)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )