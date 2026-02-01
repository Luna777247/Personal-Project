"""
Waste Management Route Optimization Environment
Simulates waste truck routing through urban grid.
"""

import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict
from .base_env import BaseEnvironment


class WasteManagementEnv(BaseEnvironment):
    """
    Waste Management Optimization Environment
    
    Agent controls a waste truck through a grid city.
    Goal: Collect waste from all bins in minimum distance/time.
    
    Observations:
        - Current truck position (x, y)
        - Remaining bins locations and waste levels
        - Current load
        - Fuel level
        
    Actions:
        - 0: Move North
        - 1: Move South
        - 2: Move East
        - 3: Move West
        - 4: Collect waste from current location
        - 5: Return to depot
        
    Rewards:
        - +10 for collecting waste bin
        - -0.1 per step (distance penalty)
        - -0.05 per distance unit traveled
        - +50 for returning all waste to depot
        - -50 for exceeding fuel/capacity
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        num_bins: int = 10,
        max_steps: int = 500,
        capacity: int = 100,
        fuel_capacity: int = 500,
        seed: int = None
    ):
        """
        Initialize waste management environment.
        
        Args:
            grid_size: Size of grid (grid_size x grid_size)
            num_bins: Number of waste bins in grid
            max_steps: Maximum steps per episode
            capacity: Truck capacity for waste
            fuel_capacity: Truck fuel capacity
            seed: Random seed
        """
        super().__init__(max_steps=max_steps)
        
        self.grid_size = grid_size
        self.num_bins = num_bins
        self.capacity = capacity
        self.fuel_capacity = fuel_capacity
        self.max_fuel = fuel_capacity
        
        # State variables
        self.truck_pos = np.array([0, 0])  # Start at depot
        self.depot_pos = np.array([0, 0])
        self.bin_positions = None
        self.bin_waste_levels = None
        self.current_load = 0
        self.current_fuel = fuel_capacity
        self.bins_collected = None
        self.total_distance = 0
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: [truck_x, truck_y, load, fuel, bin_distances (num_bins), waste_levels (num_bins), collected (num_bins)]
        obs_size = 4 + num_bins * 3  # truck pos(2) + load(1) + fuel(1) + bins info(num_bins*3)
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size, capacity, fuel_capacity),
            shape=(obs_size,), dtype=np.float32
        )
        
        if seed is not None:
            self.seed(seed)
            
        self.reset()
    
    def _initialize_bins(self):
        """Initialize bin positions and waste levels."""
        self.bin_positions = np.array([
            self.np_random.integers(1, self.grid_size, size=2)
            for _ in range(self.num_bins)
        ], dtype=np.float32)
        
        self.bin_waste_levels = self.np_random.integers(
            10, self.capacity, size=self.num_bins
        )
        self.bins_collected = np.zeros(self.num_bins, dtype=bool)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(4 + self.num_bins * 3, dtype=np.float32)
        
        # Truck state
        obs[0] = self.truck_pos[0]
        obs[1] = self.truck_pos[1]
        obs[2] = self.current_load
        obs[3] = self.current_fuel
        
        # Bin information
        for i in range(self.num_bins):
            distance = np.linalg.norm(
                self.bin_positions[i] - self.truck_pos
            )
            obs[4 + i * 3] = distance
            obs[4 + i * 3 + 1] = self.bin_waste_levels[i]
            obs[4 + i * 3 + 2] = float(self.bins_collected[i])
        
        return obs
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for action."""
        reward = 0.0
        old_pos = self.truck_pos.copy()
        
        # Movement actions
        if action == 0:  # North
            self.truck_pos[1] = min(self.truck_pos[1] + 1, self.grid_size - 1)
        elif action == 1:  # South
            self.truck_pos[1] = max(self.truck_pos[1] - 1, 0)
        elif action == 2:  # East
            self.truck_pos[0] = min(self.truck_pos[0] + 1, self.grid_size - 1)
        elif action == 3:  # West
            self.truck_pos[0] = max(self.truck_pos[0] - 1, 0)
        elif action == 4:  # Collect waste
            self._collect_waste()
            return reward
        elif action == 5:  # Return to depot
            self._return_to_depot()
            return reward
        
        # Calculate distance traveled
        distance = np.linalg.norm(self.truck_pos - old_pos)
        self.total_distance += distance
        
        # Penalize distance traveled
        reward -= 0.1  # Step penalty
        reward -= 0.05 * distance  # Distance penalty
        
        # Penalize fuel consumption
        self.current_fuel -= distance
        if self.current_fuel < 0:
            reward -= 50
            self.current_fuel = 0
        
        return reward
    
    def _collect_waste(self):
        """Collect waste from current location."""
        # Check if any bin at current position
        for i in range(self.num_bins):
            if np.allclose(self.truck_pos, self.bin_positions[i]):
                if not self.bins_collected[i]:
                    waste_amount = min(
                        self.bin_waste_levels[i],
                        self.capacity - self.current_load
                    )
                    self.current_load += waste_amount
                    self.bin_waste_levels[i] -= waste_amount
                    
                    if self.bin_waste_levels[i] == 0:
                        self.bins_collected[i] = True
                        self.episode_reward += 10  # Bonus for collecting
                    break
    
    def _return_to_depot(self):
        """Return to depot and unload waste."""
        # Check if at depot
        if np.allclose(self.truck_pos, self.depot_pos):
            self.current_load = 0
            # Bonus if all bins collected
            if np.all(self.bins_collected):
                self.episode_reward += 50
                self.done = True
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        # Done if all bins collected and returned to depot
        if (np.all(self.bins_collected) and 
            np.allclose(self.truck_pos, self.depot_pos)):
            return True
        
        # Done if out of fuel
        if self.current_fuel <= 0:
            return True
            
        return self.done
    
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            self.seed(seed)
        
        # Initialize state
        self.truck_pos = self.depot_pos.copy()
        self.current_load = 0
        self.current_fuel = self.fuel_capacity
        self.total_distance = 0
        self._initialize_bins()
        
        return super().reset(seed=seed, options=options)
    
    def get_info(self) -> Dict:
        """Get environment info."""
        return {
            "total_distance": self.total_distance,
            "bins_collected": int(np.sum(self.bins_collected)),
            "total_bins": self.num_bins,
            "current_load": self.current_load,
            "current_fuel": self.current_fuel,
            "collection_rate": float(np.sum(self.bins_collected)) / self.num_bins
        }
