"""
Smart Irrigation Optimization Environment
Simulates crop irrigation control.
"""

import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict
from .base_env import BaseEnvironment


class SmartIrrigationEnv(BaseEnvironment):
    """
    Smart Irrigation Optimization Environment
    
    Agent controls irrigation system for crop growth.
    Goal: Maximize yield while minimizing water usage.
    
    Observations:
        - Soil moisture (0-100)
        - Temperature
        - Rainfall
        - Crop growth stage
        - Water usage
        
    Actions:
        - 0: No irrigation
        - 1: Light irrigation (10 units)
        - 2: Medium irrigation (25 units)
        - 3: Heavy irrigation (50 units)
        
    Rewards:
        - Bonus for optimal soil moisture (40-60)
        - Penalty for water usage
        - Bonus for crop growth
        - Penalty for crop stress
    """
    
    def __init__(
        self,
        num_crops: int = 5,
        max_steps: int = 365,  # 1 year
        seed: int = None
    ):
        """
        Initialize smart irrigation environment.
        
        Args:
            num_crops: Number of crop fields
            max_steps: Maximum steps per episode (days)
            seed: Random seed
        """
        super().__init__(max_steps=max_steps)
        
        self.num_crops = num_crops
        
        # State variables
        self.soil_moisture = np.full(num_crops, 50.0)  # 0-100
        self.temperature = 25.0
        self.rainfall = 0.0
        self.crop_growth = np.zeros(num_crops)  # 0-100
        self.water_used = 0.0
        self.total_yield = 0.0
        self.day = 0
        
        # Action space: 4 irrigation levels per field
        self.action_space = spaces.MultiDiscrete([4] * num_crops)
        
        # Observation space
        obs_size = num_crops * 2 + 4  # moisture + growth per crop + temp, rainfall, day, water_used
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )
        
        if seed is not None:
            self.seed(seed)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.num_crops * 2 + 4, dtype=np.float32)
        
        # Soil moisture and growth for each crop
        for i in range(self.num_crops):
            obs[i * 2] = self.soil_moisture[i]
            obs[i * 2 + 1] = self.crop_growth[i]
        
        # Environmental factors
        obs[self.num_crops * 2] = self.temperature
        obs[self.num_crops * 2 + 1] = self.rainfall
        obs[self.num_crops * 2 + 2] = self.day / self.max_steps * 100  # Normalized day
        obs[self.num_crops * 2 + 3] = min(self.water_used / 1000, 100)  # Normalized water
        
        return obs
    
    def _calculate_reward(self, actions) -> float:
        """Calculate reward for action."""
        if not isinstance(actions, (list, tuple, np.ndarray)):
            actions = [actions]
        
        reward = 0.0
        
        # Process each crop
        for i in range(self.num_crops):
            action = actions[i] if i < len(actions) else 0
            
            # Apply irrigation
            irrigation_amount = [0, 10, 25, 50][action]
            self.soil_moisture[i] += irrigation_amount
            self.water_used += irrigation_amount
            
            # Simulate rainfall
            if self.np_random.random() < 0.3:  # 30% chance of rain
                self.rainfall = self.np_random.uniform(5, 20)
                self.soil_moisture[i] += self.rainfall
            else:
                self.rainfall = 0.0
            
            # Natural water loss (evapotranspiration)
            evapotranspiration = 2 + self.temperature / 10
            self.soil_moisture[i] -= evapotranspiration
            
            # Bound soil moisture
            self.soil_moisture[i] = np.clip(self.soil_moisture[i], 0, 100)
            
            # Crop growth model
            if self.soil_moisture[i] < 20:  # Stress (wilting point)
                self.crop_growth[i] -= 2
                reward -= 5  # Penalty for stress
            elif 30 <= self.soil_moisture[i] <= 70:  # Optimal range
                self.crop_growth[i] += 1.5
                reward += 2  # Bonus for optimal
            else:  # Excessive moisture
                self.crop_growth[i] += 0.5
                reward -= 1  # Slight penalty
            
            # Bound crop growth
            self.crop_growth[i] = np.clip(self.crop_growth[i], 0, 100)
            
            # Water usage penalty (cost)
            reward -= 0.01 * irrigation_amount
        
        # Yield calculation
        self.total_yield = np.mean(self.crop_growth)
        
        # Reward for high yield
        reward += 0.05 * self.total_yield
        
        # Temperature cycling (seasonal)
        self.day += 1
        day_in_year = self.day % 365
        self.temperature = 25 + 15 * np.sin(2 * np.pi * day_in_year / 365)
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with single action or multiple actions."""
        # Handle both single and multi-action inputs
        if isinstance(action, (list, tuple)):
            actions = action
        elif isinstance(action, np.ndarray) and action.shape == (self.num_crops,):
            actions = action
        else:
            actions = [action] * self.num_crops
        
        # Validate actions
        for a in actions:
            if not self.action_space.contains(np.array(actions)):
                raise ValueError(f"Invalid action: {actions}")
        
        # Calculate reward
        reward = self._calculate_reward(actions)
        self.episode_reward += reward
        
        # Increment step counter
        self.episode_steps += 1
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = self.episode_steps >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Info dict
        info = self.get_info()
        info.update({
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward,
            "terminated": terminated,
            "truncated": truncated
        })
        
        return observation, reward, terminated, truncated, info
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Episode ends if all crops are dead (growth < 0)
        if np.any(self.crop_growth < 0):
            return True
        return False
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            self.seed(seed)
        
        self.soil_moisture = np.full(self.num_crops, 50.0)
        self.temperature = 25.0
        self.rainfall = 0.0
        self.crop_growth = np.zeros(self.num_crops)
        self.water_used = 0.0
        self.total_yield = 0.0
        self.day = 0
        
        return super().reset(seed=seed, options=options)
    
    def get_info(self) -> Dict:
        """Get environment info."""
        return {
            "average_moisture": float(np.mean(self.soil_moisture)),
            "average_growth": float(np.mean(self.crop_growth)),
            "total_yield": float(self.total_yield),
            "water_used": float(self.water_used),
            "day": self.day,
            "temperature": float(self.temperature),
            "water_efficiency": float(self.total_yield / max(self.water_used, 1))
        }
