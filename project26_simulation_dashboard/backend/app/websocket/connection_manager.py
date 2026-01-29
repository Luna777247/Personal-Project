"""
WebSocket Connection Manager
Handles real-time connections for simulation data streaming
"""

import logging
from typing import Dict, List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time data streaming"""

    def __init__(self):
        # active_connections[simulation_id] = [websocket1, websocket2, ...]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, simulation_id: str):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()

        if simulation_id not in self.active_connections:
            self.active_connections[simulation_id] = []

        self.active_connections[simulation_id].append(websocket)
        logger.info(f"Client connected to simulation {simulation_id}. Total connections: {len(self.active_connections[simulation_id])}")

    def disconnect(self, websocket: WebSocket, simulation_id: str):
        """Remove a WebSocket connection"""
        if simulation_id in self.active_connections:
            if websocket in self.active_connections[simulation_id]:
                self.active_connections[simulation_id].remove(websocket)
                logger.info(f"Client disconnected from simulation {simulation_id}. Remaining connections: {len(self.active_connections[simulation_id])}")

            # Clean up empty simulation entries
            if not self.active_connections[simulation_id]:
                del self.active_connections[simulation_id]

    async def broadcast_to_simulation(self, simulation_id: str, message: dict):
        """Broadcast message to all clients connected to a simulation"""
        if simulation_id not in self.active_connections:
            return

        disconnected_clients = []

        for websocket in self.active_connections[simulation_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected_clients.append(websocket)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.disconnect(client, simulation_id)

    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")

    def get_connection_count(self, simulation_id: str) -> int:
        """Get number of active connections for a simulation"""
        return len(self.active_connections.get(simulation_id, []))

    def get_all_simulations(self) -> List[str]:
        """Get list of all simulations with active connections"""
        return list(self.active_connections.keys())