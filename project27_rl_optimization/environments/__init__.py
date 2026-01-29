"""
RL Environments Package
Provides OpenAI Gymnasium-compatible environments for three optimization scenarios.
"""

from .base_env import BaseEnvironment
from .waste_management import WasteManagementEnv
from .traffic_light import TrafficLightEnv
from .smart_irrigation import SmartIrrigationEnv

__all__ = [
    "BaseEnvironment",
    "WasteManagementEnv",
    "TrafficLightEnv",
    "SmartIrrigationEnv"
]
