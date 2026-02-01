"""
Traffic Light Control Optimization Environment
Simulates traffic intersection control.
"""

import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict
from .base_env import BaseEnvironment


class TrafficLightEnv(BaseEnvironment):
    """
    Traffic Light Control Optimization Environment
    
    Agent controls traffic light timing at intersection.
    Goal: Minimize average waiting time and congestion.
    
    Observations:
        - Queue lengths (N-S, E-W)
        - Current light state
        - Time since last change
        - Waiting times
        
    Actions:
        - 0-10: Green light duration for North-South (0=0s, 10=10s)
        
    Rewards:
        - -1 per vehicle waiting
        - -0.5 per second of delay
        - +10 for clearing queue
        - -50 for congestion
    """
    
    def __init__(
        self,
        num_lanes: int = 4,  # N, S, E, W
        max_steps: int = 500,
        arrival_rate: float = 0.3,
        seed: int = None
    ):
        """
        Initialize traffic light environment.
        
        Args:
            num_lanes: Number of lanes (4 = 4-way intersection)
            max_steps: Maximum steps per episode
            arrival_rate: Vehicle arrival probability per step
            seed: Random seed
        """
        super().__init__(max_steps=max_steps)
        
        self.num_lanes = num_lanes
        self.arrival_rate = arrival_rate
        
        # State: queue length for each lane
        self.queue_lengths = np.zeros(num_lanes, dtype=np.int32)
        self.waiting_times = np.zeros(num_lanes, dtype=np.float32)
        self.green_light_ns = 0  # Green light duration N-S
        self.green_light_ew = 0  # Green light duration E-W
        self.current_light = 0   # 0: N-S green, 1: E-W green
        self.time_since_change = 0
        self.total_wait_time = 0
        self.vehicles_passed = 0
        
        # Action space: green light duration 0-10 seconds
        self.action_space = spaces.Discrete(11)
        
        # Observation space
        obs_size = num_lanes + 3  # queue lengths + light state + time since change + current green
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )
        
        if seed is not None:
            self.seed(seed)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.num_lanes + 3, dtype=np.float32)
        
        obs[:self.num_lanes] = self.queue_lengths.astype(np.float32)
        obs[self.num_lanes] = float(self.current_light)
        obs[self.num_lanes + 1] = float(self.time_since_change)
        obs[self.num_lanes + 2] = (self.green_light_ns if self.current_light == 0 
                                    else self.green_light_ew)
        
        return obs
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for action."""
        reward = 0.0
        
        # Set green light duration
        if self.current_light == 0:
            self.green_light_ns = action
        else:
            self.green_light_ew = action
        
        # Vehicle arrivals
        for i in range(self.num_lanes):
            if self.np_random.random() < self.arrival_rate:
                self.queue_lengths[i] += 1
        
        # Process vehicles (clear based on green light)
        if self.current_light == 0:  # N-S green
            for i in [0, 1]:  # N, S
                cleared = min(self.queue_lengths[i], self.green_light_ns)
                self.queue_lengths[i] -= cleared
                self.vehicles_passed += cleared
        else:  # E-W green
            for i in [2, 3]:  # E, W
                cleared = min(self.queue_lengths[i], self.green_light_ew)
                self.queue_lengths[i] -= cleared
                self.vehicles_passed += cleared
        
        # Increment waiting times
        self.waiting_times += self.queue_lengths.astype(np.float32)
        self.total_wait_time += np.sum(self.queue_lengths)
        
        # Calculate reward
        total_queue = np.sum(self.queue_lengths)
        
        # Penalize waiting vehicles
        reward -= 0.5 * total_queue
        
        # Penalize total wait time
        reward -= 0.1 * np.sum(self.waiting_times)
        
        # Reward for clearing queues
        if total_queue == 0:
            reward += 10
        
        # Penalize congestion (queue > 20)
        if total_queue > 20:
            reward -= 50
        
        # Switch light
        self.time_since_change += 1
        if self.time_since_change >= max(self.green_light_ns, self.green_light_ew) + 2:
            self.current_light = 1 - self.current_light
            self.time_since_change = 0
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Episode ends at max steps
        return False
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            self.seed(seed)
        
        self.queue_lengths = np.zeros(self.num_lanes, dtype=np.int32)
        self.waiting_times = np.zeros(self.num_lanes, dtype=np.float32)
        self.green_light_ns = 5
        self.green_light_ew = 5
        self.current_light = 0
        self.time_since_change = 0
        self.total_wait_time = 0
        self.vehicles_passed = 0
        
        return super().reset(seed=seed, options=options)
    
    def get_info(self) -> Dict:
        """Get environment info."""
        avg_wait = (self.total_wait_time / max(self.vehicles_passed, 1) 
                    if self.vehicles_passed > 0 else 0)
        return {
            "total_queue_length": int(np.sum(self.queue_lengths)),
            "average_wait_time": float(avg_wait),
            "vehicles_passed": self.vehicles_passed,
            "max_queue": int(np.max(self.queue_lengths)),
            "congestion_level": float(np.sum(self.queue_lengths) / 100)
        }
