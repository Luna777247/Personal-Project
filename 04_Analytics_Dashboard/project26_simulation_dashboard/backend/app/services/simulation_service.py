"""
Simulation Service
Handles communication with GAMA simulations and data management
"""

import asyncio
import json
import logging
import socket
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random

from ..models import (
    SimulationInfo, SimulationData, SimulationConfig,
    SimulationStatus, SimulationType, DashboardData,
    PollutionData, TrafficData, WeatherData, AgricultureData
)
from ..config import settings

logger = logging.getLogger(__name__)


class SimulationService:
    """Service for managing simulation connections and data"""

    def __init__(self):
        self.simulations: Dict[str, SimulationInfo] = {}
        self.simulation_data: Dict[str, List[Dict]] = {}
        self.data_lock = threading.Lock()
        self.running = False

    async def initialize(self):
        """Initialize the simulation service"""
        self.running = True
        logger.info("Simulation service initialized")

        # Start background data generation for demo
        asyncio.create_task(self._generate_demo_data())

    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        logger.info("Simulation service cleaned up")

    async def get_active_simulations(self) -> List[Dict]:
        """Get list of active simulations"""
        return [
            {
                "id": sim.id,
                "name": sim.name,
                "type": sim.type.value,
                "status": sim.status.value,
                "current_cycle": sim.current_cycle
            }
            for sim in self.simulations.values()
            if sim.status in [SimulationStatus.RUNNING, SimulationStatus.CREATED]
        ]

    async def get_simulation(self, simulation_id: str) -> Optional[Dict]:
        """Get simulation details"""
        sim = self.simulations.get(simulation_id)
        if not sim:
            return None

        return {
            "id": sim.id,
            "name": sim.name,
            "type": sim.type.value,
            "status": sim.status.value,
            "config": sim.config.dict(),
            "current_cycle": sim.current_cycle,
            "total_cycles": sim.total_cycles
        }

    async def create_simulation(self, config: Dict) -> str:
        """Create new simulation"""
        simulation_id = f"sim_{int(time.time())}_{random.randint(1000, 9999)}"

        sim_config = SimulationConfig(**config)
        simulation = SimulationInfo(
            id=simulation_id,
            name=config.get("name", f"Simulation {simulation_id}"),
            type=SimulationType(config.get("type", "climate")),
            status=SimulationStatus.CREATED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            config=sim_config,
            total_cycles=config.get("duration")
        )

        self.simulations[simulation_id] = simulation
        self.simulation_data[simulation_id] = []

        logger.info(f"Created simulation {simulation_id}")
        return simulation_id

    async def get_simulation_data(self, simulation_id: str, limit: int = 100) -> List[Dict]:
        """Get simulation data points"""
        with self.data_lock:
            data = self.simulation_data.get(simulation_id, [])
            return data[-limit:] if data else []

    async def get_latest_data(self, simulation_id: str) -> Optional[Dict]:
        """Get latest data point for simulation"""
        with self.data_lock:
            data = self.simulation_data.get(simulation_id, [])
            return data[-1] if data else None

    async def get_simulation_metrics(self, simulation_id: str) -> Dict:
        """Get aggregated metrics for simulation"""
        data = await self.get_simulation_data(simulation_id, limit=1000)

        if not data:
            return {}

        # Calculate metrics based on simulation type
        sim = self.simulations.get(simulation_id)
        if not sim:
            return {}

        metrics = {}

        if sim.type == SimulationType.CLIMATE:
            # Climate simulation metrics
            temperatures = [d.get('weather', {}).get('temperature', 0) for d in data]
            precipitations = [d.get('weather', {}).get('precipitation', 0) for d in data]

            if temperatures:
                metrics['temperature'] = {
                    'min': min(temperatures),
                    'max': max(temperatures),
                    'mean': sum(temperatures) / len(temperatures),
                    'current': temperatures[-1]
                }

            if precipitations:
                metrics['precipitation'] = {
                    'total': sum(precipitations),
                    'max': max(precipitations),
                    'current': precipitations[-1]
                }

        elif sim.type == SimulationType.POLLUTION:
            # Pollution metrics
            aqis = [d.get('pollution', {}).get('air_quality_index', 0) for d in data]
            if aqis:
                metrics['air_quality'] = {
                    'min': min(aqis),
                    'max': max(aqis),
                    'mean': sum(aqis) / len(aqis),
                    'current': aqis[-1]
                }

        elif sim.type == SimulationType.TRAFFIC:
            # Traffic metrics
            speeds = [d.get('traffic', {}).get('average_speed', 0) for d in data]
            if speeds:
                metrics['traffic_speed'] = {
                    'min': min(speeds),
                    'max': max(speeds),
                    'mean': sum(speeds) / len(speeds),
                    'current': speeds[-1]
                }

        return metrics

    async def send_command(self, simulation_id: str, command: Dict) -> Dict:
        """Send control command to simulation"""
        sim = self.simulations.get(simulation_id)
        if not sim:
            return {"error": "Simulation not found"}

        cmd = command.get("command")
        if cmd == "start":
            sim.status = SimulationStatus.RUNNING
        elif cmd == "pause":
            sim.status = SimulationStatus.PAUSED
        elif cmd == "stop":
            sim.status = SimulationStatus.COMPLETED
        elif cmd == "reset":
            sim.current_cycle = 0
            sim.status = SimulationStatus.CREATED

        sim.updated_at = datetime.utcnow()
        return {"status": "command_sent", "command": cmd}

    async def _generate_demo_data(self):
        """Generate demo data for testing (simulates GAMA output)"""
        await asyncio.sleep(2)  # Wait for initialization

        # Create demo simulations
        climate_sim = await self.create_simulation({
            "name": "Climate Change Impact Demo",
            "type": "climate",
            "duration": 1000
        })

        pollution_sim = await self.create_simulation({
            "name": "Urban Pollution Monitor",
            "type": "pollution",
            "duration": 1000
        })

        traffic_sim = await self.create_simulation({
            "name": "Traffic Flow Analysis",
            "type": "traffic",
            "duration": 1000
        })

        agriculture_sim = await self.create_simulation({
            "name": "Agricultural Impact Model",
            "type": "agriculture",
            "duration": 1000
        })

        # Start generating data
        cycle = 0
        while self.running:
            cycle += 1

            # Generate data for each simulation
            await self._generate_climate_data(climate_sim, cycle)
            await self._generate_pollution_data(pollution_sim, cycle)
            await self._generate_traffic_data(traffic_sim, cycle)
            await self._generate_agriculture_data(agriculture_sim, cycle)

            await asyncio.sleep(0.5)  # Generate data every 0.5 seconds

    async def _generate_climate_data(self, sim_id: str, cycle: int):
        """Generate climate simulation data"""
        weather_data = WeatherData(
            temperature=25 + 5 * random.random() + 2 * cycle / 100,
            humidity=60 + 20 * random.random(),
            precipitation=max(0, 10 * random.random() - 5),
            wind_speed=5 + 10 * random.random(),
            wind_direction=360 * random.random(),
            pressure=1013 + 10 * (random.random() - 0.5)
        )

        dashboard_data = DashboardData(
            simulation_id=sim_id,
            timestamp=datetime.utcnow(),
            cycle=cycle,
            weather=weather_data
        )

        with self.data_lock:
            self.simulation_data[sim_id].append(dashboard_data.dict())
            # Keep only last 1000 data points
            if len(self.simulation_data[sim_id]) > 1000:
                self.simulation_data[sim_id] = self.simulation_data[sim_id][-1000:]

        # Update simulation status
        sim = self.simulations.get(sim_id)
        if sim:
            sim.current_cycle = cycle
            sim.updated_at = datetime.utcnow()

    async def _generate_pollution_data(self, sim_id: str, cycle: int):
        """Generate pollution simulation data"""
        pollution_data = PollutionData(
            air_quality_index=50 + 100 * random.random(),
            pm25=10 + 50 * random.random(),
            pm10=20 + 80 * random.random(),
            no2=10 + 40 * random.random(),
            so2=5 + 25 * random.random(),
            co=0.5 + 2 * random.random(),
            o3=20 + 60 * random.random()
        )

        dashboard_data = DashboardData(
            simulation_id=sim_id,
            timestamp=datetime.utcnow(),
            cycle=cycle,
            pollution=pollution_data
        )

        with self.data_lock:
            self.simulation_data[sim_id].append(dashboard_data.dict())
            if len(self.simulation_data[sim_id]) > 1000:
                self.simulation_data[sim_id] = self.simulation_data[sim_id][-1000:]

        sim = self.simulations.get(sim_id)
        if sim:
            sim.current_cycle = cycle
            sim.updated_at = datetime.utcnow()

    async def _generate_traffic_data(self, sim_id: str, cycle: int):
        """Generate traffic simulation data"""
        traffic_data = TrafficData(
            vehicle_count=int(100 + 200 * random.random()),
            average_speed=max(5, 60 * random.random()),
            congestion_level=random.random(),
            public_transport_efficiency=0.6 + 0.4 * random.random()
        )

        dashboard_data = DashboardData(
            simulation_id=sim_id,
            timestamp=datetime.utcnow(),
            cycle=cycle,
            traffic=traffic_data
        )

        with self.data_lock:
            self.simulation_data[sim_id].append(dashboard_data.dict())
            if len(self.simulation_data[sim_id]) > 1000:
                self.simulation_data[sim_id] = self.simulation_data[sim_id][-1000:]

        sim = self.simulations.get(sim_id)
        if sim:
            sim.current_cycle = cycle
            sim.updated_at = datetime.utcnow()

    async def _generate_agriculture_data(self, sim_id: str, cycle: int):
        """Generate agriculture simulation data"""
        agriculture_data = AgricultureData(
            crop_yield=max(0, 100 * random.random()),
            irrigation_efficiency=0.5 + 0.5 * random.random(),
            soil_moisture=20 + 60 * random.random(),
            drought_index=random.random(),
            affected_area=10 * random.random()
        )

        dashboard_data = DashboardData(
            simulation_id=sim_id,
            timestamp=datetime.utcnow(),
            cycle=cycle,
            agriculture=agriculture_data
        )

        with self.data_lock:
            self.simulation_data[sim_id].append(dashboard_data.dict())
            if len(self.simulation_data[sim_id]) > 1000:
                self.simulation_data[sim_id] = self.simulation_data[sim_id][-1000:]

        sim = self.simulations.get(sim_id)
        if sim:
            sim.current_cycle = cycle
            sim.updated_at = datetime.utcnow()