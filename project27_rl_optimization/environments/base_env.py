"""
Base environment class for all RL environments.
Provides common functionality and interface.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class BaseEnvironment(gym.Env, ABC):
    """
    Abstract base class for all RL environments.
    
    Attributes:
        action_space: gymnasium.spaces.Space - Action space definition
        observation_space: gymnasium.spaces.Space - Observation space definition
        episode_steps: int - Current episode step count
        episode_reward: float - Cumulative episode reward
        max_steps: int - Maximum steps per episode
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, max_steps: int = 1000, render_mode: str = None):
        """
        Initialize base environment.
        
        Args:
            max_steps: Maximum steps per episode
            render_mode: Rendering mode ('human' or None)
        """
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.done = False
        
        # Define spaces (must be set by subclasses)
        self.action_space = None
        self.observation_space = None
        
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for taken action. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _is_done(self) -> bool:
        """Check if episode is done. Must be implemented by subclass."""
        pass
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            observation, info dict
        """
        super().reset(seed=seed)
        
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.done = False
        
        observation = self._get_observation()
        info = {
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info dict
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward
        
        # Increment step counter
        self.episode_steps += 1
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = self.episode_steps >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            "episode_steps": self.episode_steps,
            "episode_reward": self.episode_reward,
            "terminated": terminated,
            "truncated": truncated
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional)."""
        if self.render_mode == "human":
            self._render_human()
    
    def _render_human(self):
        """Override to implement human rendering."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed: int = None):
        """Set random seed."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
