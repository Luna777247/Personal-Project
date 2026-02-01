"""
Data models for Simulation Dashboard
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class SimulationStatus(str, Enum):
    """Simulation status enumeration"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class SimulationType(str, Enum):
    """Type of simulation"""
    CLIMATE = "climate"
    TRAFFIC = "traffic"
    POLLUTION = "pollution"
    AGRICULTURE = "agriculture"
    URBAN = "urban"


class SimulationConfig(BaseModel):
    """Configuration for a simulation"""
    name: str
    type: SimulationType
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    duration: Optional[int] = None  # in cycles
    step_size: Optional[float] = 1.0  # time step


class SimulationInfo(BaseModel):
    """Information about a simulation"""
    id: str
    name: str
    type: SimulationType
    status: SimulationStatus
    created_at: datetime
    updated_at: datetime
    config: SimulationConfig
    current_cycle: int = 0
    total_cycles: Optional[int] = None


class DataPoint(BaseModel):
    """Single data point from simulation"""
    timestamp: datetime
    cycle: int
    metrics: Dict[str, Any]


class SimulationData(BaseModel):
    """Complete simulation data"""
    simulation_id: str
    data_points: List[DataPoint]
    metadata: Dict[str, Any] = {}


class MetricSummary(BaseModel):
    """Summary statistics for a metric"""
    name: str
    min_value: float
    max_value: float
    mean_value: float
    current_value: float
    trend: str  # "increasing", "decreasing", "stable"


class SimulationMetrics(BaseModel):
    """Aggregated metrics for a simulation"""
    simulation_id: str
    metrics: List[MetricSummary]
    last_updated: datetime


class PollutionData(BaseModel):
    """Pollution-specific data"""
    air_quality_index: float
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float
    emission_sources: List[Dict[str, Any]] = []


class TrafficData(BaseModel):
    """Traffic-specific data"""
    vehicle_count: int
    average_speed: float
    congestion_level: float
    hotspots: List[Dict[str, Any]] = []
    public_transport_efficiency: float = 0.0


class WeatherData(BaseModel):
    """Weather-specific data"""
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    wind_direction: float
    pressure: float


class AgricultureData(BaseModel):
    """Agriculture-specific data"""
    crop_yield: float
    irrigation_efficiency: float
    soil_moisture: float
    drought_index: float
    affected_area: float


class DashboardData(BaseModel):
    """Combined data for dashboard display"""
    simulation_id: str
    timestamp: datetime
    cycle: int
    pollution: Optional[PollutionData] = None
    traffic: Optional[TrafficData] = None
    weather: Optional[WeatherData] = None
    agriculture: Optional[AgricultureData] = None
    custom_metrics: Dict[str, Any] = {}


class WebSocketMessage(BaseModel):
    """Message format for WebSocket communication"""
    type: str  # "update", "initial_data", "error", "control"
    data: Any
    timestamp: datetime
    simulation_id: str


class ControlCommand(BaseModel):
    """Command to control simulation"""
    command: str  # "start", "pause", "stop", "reset", "update_params"
    parameters: Dict[str, Any] = {}