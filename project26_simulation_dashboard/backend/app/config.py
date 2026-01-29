"""
Configuration settings for Simulation Dashboard Backend
"""

import os
from typing import List, Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS settings
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ]

    # GAMA Platform settings
    gama_host: str = "localhost"
    gama_port: int = 8080
    gama_timeout: int = 30

    # Data settings
    max_data_points: int = 10000
    data_retention_hours: int = 24

    # WebSocket settings
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10

    # External APIs (optional)
    openweather_api_key: Optional[str] = None
    mapbox_token: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()