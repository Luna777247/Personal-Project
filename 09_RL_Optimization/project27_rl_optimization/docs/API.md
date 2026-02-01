# API Documentation

## Environments API

### BaseEnvironment

Abstract base class for all custom environments.

```python
from environments.base_env import BaseEnvironment

class CustomEnv(BaseEnvironment):
    def __init__(self, max_steps=1000):
        super().__init__(max_steps=max_steps)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=100, shape=(10,))
    
    def _get_observation(self):
        # Return current observation
        return np.zeros(10)
    
    def _calculate_reward(self, action):
        # Calculate reward for action
        return 0.0
    
    def _is_done(self):
        # Check episode termination
        return False
```

### WasteManagementEnv

Waste collection route optimization environment.

```python
from environments import WasteManagementEnv

# Create environment
env = WasteManagementEnv(
    grid_size=20,        # City grid size
    num_bins=10,         # Number of waste bins
    max_steps=500,       # Episode length
    capacity=100,        # Truck capacity
    fuel_capacity=500,   # Truck fuel
    seed=42
)

# Reset and get initial observation
obs, info = env.reset()

# Take action
action = 0  # Move North
obs, reward, terminated, truncated, info = env.step(action)

# Get environment metrics
metrics = env.get_info()
print(f"Bins collected: {metrics['bins_collected']}/{metrics['total_bins']}")
print(f"Distance traveled: {metrics['total_distance']:.2f}")
```

**State Space**: 
- Truck position (x, y)
- Current load
- Fuel level
- Distance to each bin
- Waste level at each bin
- Collection status for each bin

**Action Space**: 
- 0: Move North
- 1: Move South
- 2: Move East
- 3: Move West
- 4: Collect waste
- 5: Return to depot

**Rewards**:
- +10 per bin collected
- -0.1 per step
- -0.05 per distance unit
- +50 for completing tour
- -50 for exceeding limits

### TrafficLightEnv

Traffic light control at intersection.

```python
from environments import TrafficLightEnv

# Create environment
env = TrafficLightEnv(
    num_lanes=4,           # 4-way intersection
    max_steps=500,         # Episode length
    arrival_rate=0.3,      # Vehicle arrival probability
    seed=42
)

# Reset and step through episode
obs, info = env.reset()
action = 5  # Green light duration
obs, reward, terminated, truncated, info = env.step(action)

# Get metrics
metrics = env.get_info()
print(f"Queue length: {metrics['total_queue_length']}")
print(f"Avg wait time: {metrics['average_wait_time']:.2f}s")
```

**State Space**:
- Queue lengths (N, S, E, W)
- Current light state (N-S or E-W)
- Time since last change
- Current green light duration

**Action Space**:
- 0-10: Green light duration (0-10 seconds)

**Rewards**:
- -0.5 per queued vehicle
- -0.1 per cumulative wait time
- +10 for clearing queues
- -50 for congestion (queue > 20)

### SmartIrrigationEnv

Smart crop irrigation optimization.

```python
from environments import SmartIrrigationEnv

# Create environment
env = SmartIrrigationEnv(
    num_crops=5,        # Number of fields
    max_steps=365,      # Days in season
    seed=42
)

# Reset and step through episode
obs, info = env.reset()
actions = [1, 0, 2, 0, 1]  # Action per crop
obs, reward, terminated, truncated, info = env.step(actions)

# Get metrics
metrics = env.get_info()
print(f"Average yield: {metrics['average_growth']:.2f}")
print(f"Water efficiency: {metrics['water_efficiency']:.2f}")
```

**State Space**:
- Soil moisture (0-100%) per crop
- Crop growth (0-100) per crop
- Temperature
- Rainfall
- Day of season
- Water used

**Action Space** (per crop):
- 0: No irrigation
- 1: Light (10 units)
- 2: Medium (25 units)
- 3: Heavy (50 units)

**Rewards**:
- +2 for optimal moisture (30-70)
- -5 for stress (< 20)
- -1 for excess (> 70)
- -0.01 per unit water
- +0.05 per unit growth

## Training API

### Training Functions

```python
from training.train_waste import train_waste_agent
from training.train_traffic import train_traffic_agent
from training.train_agriculture import train_agriculture_agent

# Train waste management agent
train_waste_agent(
    algorithm="ppo",           # ppo, dqn, a2c
    total_timesteps=500000,
    learning_rate=0.0003,
    batch_size=64,
    n_epochs=10,
    num_envs=4,
    seed=42,
    save_dir="models/waste/",
    log_dir="results/logs/waste/",
    render=False,
    verbose=1
)

# Train traffic agent
train_traffic_agent(
    algorithm="ppo",
    total_timesteps=500000,
    learning_rate=0.0003,
    # ... other parameters
)

# Train agriculture agent
train_agriculture_agent(
    algorithm="ppo",
    total_timesteps=500000,
    num_crops=5,
    # ... other parameters
)
```

### Using Stable-Baselines3 Directly

```python
from stable_baselines3 import PPO, DQN, A2C
from environments import WasteManagementEnv

# Create single environment
env = WasteManagementEnv(seed=42)

# Create model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
    tensorboard_log="logs/"
)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("models/waste/ppo_model")

# Load and use
loaded_model = PPO.load("models/waste/ppo_model")
obs, info = env.reset()
action, _states = loaded_model.predict(obs)
```

### Callbacks

```python
from training.callbacks import (
    create_callbacks,
    TensorboardCallback,
    ProgressCallback,
    MetricsCallback,
    CheckpointCallback
)

# Create all callbacks
callbacks = create_callbacks(
    env_type="waste",
    log_dir="logs/",
    checkpoint_dir="models/checkpoints/",
    eval_freq=50000,
    n_eval_episodes=10,
    verbose=1
)

# Use in training
model.learn(total_timesteps=500000, callback=callbacks)
```

## Evaluation API

### RLMetrics

Track and analyze agent performance.

```python
from evaluation import RLMetrics, evaluate_agent

# Manual metrics tracking
metrics = RLMetrics()

for episode in range(10):
    obs, info = env.reset()
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    
    metrics.add_episode(
        reward=episode_reward,
        length=episode_length,
        success=(episode_length < 500),
        metrics=env.get_info()
    )

# Get summary
summary = metrics.get_summary()
print(summary)

# Save metrics
metrics.save("results/metrics.json")
```

### Evaluation Functions

```python
from evaluation import (
    evaluate_agent,
    compare_agents,
    plot_learning_curve,
    plot_metrics_comparison
)

# Evaluate single agent
mean_reward, std_reward, metrics = evaluate_agent(
    model=trained_model,
    env=test_env,
    num_episodes=10,
    deterministic=True,
    render=False
)

# Compare multiple agents
models = {
    "PPO": ppo_model,
    "DQN": dqn_model,
    "A2C": a2c_model
}

results = compare_agents(
    models_dict=models,
    env=test_env,
    num_episodes=10,
    metric_name="mean_reward"
)

# Visualization
plot_learning_curve(
    rewards=[...],
    window=100,
    title="Learning Curve",
    filepath="plots/learning_curve.png"
)

plot_metrics_comparison(
    metrics_dict=results,
    title="Algorithm Comparison",
    filepath="plots/comparison.png"
)
```

## Vectorized Training

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Create vectorized environment
def make_env():
    def _init():
        return WasteManagementEnv(seed=42)
    return _init

env = make_vec_env(make_env(), n_envs=4, seed=42)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Train on vectorized environment
model = PPO("MlpPolicy", env, learning_rate=0.0003)
model.learn(total_timesteps=500000)

# Save normalization
env.save("models/vec_normalize.pkl")

# Load for inference
env = WasteManagementEnv()
env = VecNormalize.load("models/vec_normalize.pkl", env)
```

## Model Loading and Inference

```python
from stable_baselines3 import PPO
from environments import WasteManagementEnv

# Load trained model
model = PPO.load("models/waste/ppo_trained")

# Create fresh environment
env = WasteManagementEnv(seed=123)

# Run episode
obs, info = env.reset()
done = False
episode_reward = 0

while not done:
    # Deterministic prediction
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    done = terminated or truncated

print(f"Episode reward: {episode_reward}")
env.close()
```

## Custom Reward Shaping

```python
from environments import WasteManagementEnv

class CustomRewardWasteEnv(WasteManagementEnv):
    def _calculate_reward(self, action):
        base_reward = super()._calculate_reward(action)
        
        # Add custom reward components
        if np.allclose(self.truck_pos, self.depot_pos):
            return base_reward + 10  # Bonus at depot
        
        # Penalize long episodes
        if self.episode_steps > 400:
            return base_reward - 1
        
        return base_reward

# Use custom environment
env = CustomRewardWasteEnv()
```

## Configuration Management

```python
# Create config file (config.json)
{
    "ppo": {
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99
    },
    "dqn": {
        "learning_rate": 0.0003,
        "buffer_size": 50000,
        "target_update_interval": 1000
    },
    "environments": {
        "waste": {
            "grid_size": 20,
            "num_bins": 10
        }
    }
}

# Load and use
import json

with open("config.json") as f:
    config = json.load(f)

model = PPO(**config["ppo"])
env = WasteManagementEnv(**config["environments"]["waste"])
```

## Performance Optimization

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Use SubprocVecEnv for better parallelization
def make_env(rank):
    def _init():
        return WasteManagementEnv(seed=42+rank)
    return _init

env = SubprocVecEnv([make_env(i) for i in range(4)])

# This is faster than DummyVecEnv for CPU-bound environments
```

## Monitoring and Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# During training
logger.info(f"Trained {num_timesteps} timesteps")
logger.warning(f"Reward decreased significantly")
logger.error(f"Training failed with: {error}")
```

## Troubleshooting

### Common Issues and Solutions

**Issue**: `ImportError: cannot import name 'WasteManagementEnv'`
```python
# Ensure proper path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments import WasteManagementEnv
```

**Issue**: `RuntimeError: CUDA out of memory`
```python
# Reduce batch size and parallel environments
model = PPO(
    "MlpPolicy",
    env,
    batch_size=32,  # Reduced from 64
    n_steps=1024    # Reduced from 2048
)

# Or use CPU only
import torch
torch.cuda.is_available = lambda: False
```

**Issue**: NaN values in training
```python
# Clip gradients and reduce learning rate
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.00001,  # Reduced learning rate
    max_grad_norm=0.5       # Gradient clipping
)
```

## References

- [Stable-Baselines3 API](https://stable-baselines3.readthedocs.io/en/master/modules/base_class.html)
- [Gymnasium API](https://gymnasium.farama.org/api/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

**API Version**: 1.0.0
**Last Updated**: 2024
**Compatible with**: stable-baselines3>=1.8.0, gymnasium>=0.28.0, torch>=2.0.0
