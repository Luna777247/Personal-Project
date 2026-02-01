# Training Guide

## Overview

This guide provides detailed instructions for training RL agents for the three optimization scenarios.

## Prerequisites

1. **Python 3.8+**: Verify with `python --version`
2. **Virtual Environment**: `python -m venv venv`
3. **Dependencies**: `pip install -r requirements.txt`
4. **GPU (Optional)**: For faster training, install CUDA-enabled PyTorch

## Step-by-Step Training

### 1. Waste Management Agent

#### Basic Training
```bash
python training/train_waste.py
```

This trains a PPO agent to optimize waste collection routes.

#### Understanding the Output
```
============================================================
Training Waste Management Agent
============================================================
Algorithm: PPO
Total timesteps: 500,000
Learning rate: 0.0003
Batch size: 64
Number of parallel environments: 4
Random seed: 42
Run: ppo_20240115_143022
============================================================

Model created: PPO
Policy: MlpPolicy
Parameters: 131,328

Starting training...

[Progress bar showing training completion]

Final model saved to: models/waste/ppo_20240115_143022_final
Environment normalization saved

============================================================
Final Evaluation
============================================================
Episode 1: Reward=215.34, Steps=487, Distance=95.23, Collected=100.0%
Episode 2: Reward=218.92, Steps=492, Distance=92.15, Collected=100.0%
...
Average Reward: 216.45 ± 2.34
Average Episode Length: 489.4 ± 3.6
============================================================
```

#### Advanced Configuration

**Custom Grid Size (larger city)**:
```bash
# Modify train_waste.py
env = WasteManagementEnv(
    grid_size=30,  # Larger grid
    num_bins=20,   # More bins
    capacity=150,  # Larger truck
)
```

**Different Algorithms**:
```bash
# DQN (discrete action optimization)
python training/train_waste.py --algorithm dqn --timesteps 1000000

# A2C (fast convergence)
python training/train_waste.py --algorithm a2c --lr 0.0005
```

**Hyperparameter Tuning**:
```bash
# High learning rate (faster but less stable)
python training/train_waste.py --lr 0.001

# More parallelism (faster but more memory)
python training/train_waste.py --num-envs 8 --batch-size 128

# Longer training (better performance)
python training/train_waste.py --timesteps 2000000
```

### 2. Traffic Light Agent

#### Basic Training
```bash
python training/train_traffic.py
```

#### Key Metrics During Training
- **Queue length**: Target < 5 vehicles
- **Average wait time**: Target < 30 seconds
- **Throughput**: Target > 20 vehicles/episode

#### Optimal Settings for Different Scenarios
```bash
# Light traffic (low arrival rate)
python training/train_traffic.py --timesteps 250000

# Medium traffic (default)
python training/train_traffic.py

# Heavy traffic (high arrival rate)
python training/train_traffic.py --timesteps 1000000
```

### 3. Smart Irrigation Agent

#### Basic Training
```bash
python training/train_agriculture.py
```

#### Key Metrics During Training
- **Soil moisture**: Optimal range 40-60%
- **Crop growth**: Target > 80 (end of season)
- **Water efficiency**: Target > 0.8 (yield/water)

#### Multi-Crop Training
```bash
# 3 crops (simple)
python training/train_agriculture.py --num-crops 3

# 5 crops (default)
python training/train_agriculture.py

# 10 crops (complex)
python training/train_agriculture.py --num-crops 10 --timesteps 1000000
```

## Monitoring Training

### Real-time Tensorboard Visualization

1. **Start tensorboard** (in project directory):
```bash
tensorboard --logdir results/logs/
```

2. **Open browser**:
   - Navigate to `http://localhost:6006`
   - View real-time training metrics

3. **Key Charts**:
   - **Episode Reward**: Should increase over time
   - **Episode Length**: Should stabilize
   - **Learning Rate**: Tracks optimization progress
   - **Policy Loss**: Should decrease
   - **Value Loss**: Should decrease

### Command Line Monitoring

```bash
# Watch training progress
watch -n 5 'tail -20 training.log'

# Check GPU usage (if available)
nvidia-smi -l 1

# Monitor memory usage
ps aux | grep python
```

## Training Tips

### 1. Convergence Indicators
✅ **Good signs**:
- Reward increasing smoothly
- Episode length stabilizing
- Loss curves decreasing
- No large reward spikes

❌ **Bad signs**:
- Reward oscillating wildly
- Loss increasing over time
- NaN or Inf values
- Memory increasing indefinitely

### 2. Troubleshooting

**Issue: Out of Memory**
```bash
# Reduce batch size
python training/train_waste.py --batch-size 32

# Reduce parallel environments
python training/train_waste.py --num-envs 2
```

**Issue: Training Too Slow**
```bash
# Increase parallel environments
python training/train_waste.py --num-envs 8

# Use GPU-accelerated PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue: Unstable Training**
```bash
# Reduce learning rate
python training/train_waste.py --lr 0.00001

# Change algorithm
python training/train_waste.py --algorithm a2c
```

**Issue: Poor Performance**
```bash
# Train longer
python training/train_waste.py --timesteps 2000000

# Different random seed
python training/train_waste.py --seed 123
```

### 3. Best Practices

1. **Start with defaults**: They're optimized for most use cases
2. **Use tensorboard**: Visual feedback is invaluable
3. **Train longer**: 1M timesteps usually beats 500k
4. **Multiple runs**: Try different seeds, compare results
5. **Save checkpoints**: Enable automatic model saving
6. **Monitor memory**: Watch for memory leaks during training

## Hyperparameter Guide

### PPO Hyperparameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| lr | 0.0003 | 0.00001-0.001 | Learning rate, lower = slower learning |
| batch_size | 64 | 32-256 | Larger = more stable, more memory |
| n_epochs | 10 | 5-20 | Gradient updates per batch |
| gamma | 0.99 | 0.95-0.999 | Discount factor, higher = long-term focus |
| gae_lambda | 0.95 | 0.9-0.99 | GAE smoothing |
| clip_range | 0.2 | 0.1-0.4 | Policy update clipping |

### DQN Hyperparameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| lr | 0.0003 | 0.00001-0.001 | Learning rate |
| buffer_size | 50000 | 10000-1000000 | Experience replay buffer |
| learning_starts | 1000 | 100-10000 | Steps before training starts |
| target_update | 1000 | 100-10000 | Q-target network update frequency |
| exploration | decaying | 0-1 | Exploration-exploitation tradeoff |

### A2C Hyperparameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| lr | 0.0003 | 0.00001-0.001 | Learning rate |
| gamma | 0.99 | 0.95-0.999 | Discount factor |
| gae_lambda | 0.95 | 0.9-0.99 | GAE smoothing |
| ent_coef | 0.01 | 0-0.1 | Entropy coefficient |
| vf_coef | 0.5 | 0-1 | Value function loss coefficient |

## Performance Benchmarks

### Training Speed
```
Environment: 4-core CPU, 8GB RAM
Parallel Envs: 4
Batch Size: 64

Waste Management:
- Steps/sec: ~3,500
- Training time: ~2.5 hours (500k timesteps)

Traffic Light:
- Steps/sec: ~4,000
- Training time: ~2 hours (500k timesteps)

Smart Irrigation:
- Steps/sec: ~3,000
- Training time: ~3 hours (500k timesteps)
```

### Expected Results
```
Waste Management:
- Episode reward convergence: 150-250
- Success rate: 95%+
- Distance reduction: 15-25%

Traffic Light:
- Episode reward convergence: 100-200
- Average wait time: 15-25 seconds
- Throughput: 25+ vehicles/episode

Smart Irrigation:
- Episode reward convergence: 200-400
- Final yield: 70-100
- Water efficiency: 0.8-1.2
```

## Checkpoints and Resuming

### Automatic Checkpointing
Models are saved every 50,000 timesteps to `models/{task}/`.

### Resume Training
```python
from stable_baselines3 import PPO
from environments.waste_management import WasteManagementEnv

# Load existing model
model = PPO.load("models/waste/ppo_20240115_143022_500000_steps")

# Continue training
model.learn(total_timesteps=500000, reset_num_timesteps=False)
```

## Distributed Training

### Multi-GPU Training (future enhancement)

```python
# Use stable-baselines3 with PyTorch's distributed training
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create on separate GPUs
env = SubprocVecEnv([...])  # Runs on multiple GPUs
```

## Analysis and Visualization

### Learning Curves
```bash
python -c "
from evaluation.metrics import plot_learning_curve
import json

with open('results/logs/waste/progress.json') as f:
    data = json.load(f)

plot_learning_curve(
    [x[1] for x in data],  # rewards
    window=100,
    title='Waste Management - Learning Curve',
    filepath='results/plots/learning_curve.png'
)
"
```

### Model Comparison
```bash
# Train same environment with different algorithms
python training/train_waste.py --algorithm ppo --seed 42
python training/train_waste.py --algorithm dqn --seed 42
python training/train_waste.py --algorithm a2c --seed 42

# Compare in jupyter notebook
import json
# Load and compare results
```

## Next Steps

1. **Train all three agents**: Run training for waste, traffic, and irrigation
2. **Evaluate performance**: Compare metrics across algorithms
3. **Hyperparameter optimization**: Fine-tune for best results
4. **Integration**: Connect to Project 26 dashboard
5. **Deployment**: Use trained models in production

## Troubleshooting Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] CUDA installed (if using GPU)
- [ ] Sufficient disk space (10GB recommended)
- [ ] Sufficient RAM (8GB minimum, 16GB recommended)
- [ ] Training script is executable
- [ ] Random seed set for reproducibility
- [ ] Tensorboard running for monitoring
- [ ] GPU memory checked before training

## References

- Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
- OpenAI Gymnasium: https://gymnasium.farama.org/
- PyTorch: https://pytorch.org/docs/stable/index.html
- Deep RL Course: https://www.deeplearningwizardry.com/
