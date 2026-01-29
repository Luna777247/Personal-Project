# Quick Start Guide - RL Optimization

## Installation

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Train Waste Management Agent

```bash
# Train with default settings (PPO, 500k timesteps)
python training/train_waste.py

# Train with DQN algorithm
python training/train_waste.py --algorithm dqn --timesteps 1000000

# Train with custom learning rate
python training/train_waste.py --lr 0.0001 --batch-size 128
```

**Options**:
- `--algorithm`: ppo, dqn, or a2c (default: ppo)
- `--timesteps`: Total training timesteps (default: 500000)
- `--lr`: Learning rate (default: 0.0003)
- `--batch-size`: Batch size (default: 64)
- `--num-envs`: Parallel environments (default: 4)
- `--seed`: Random seed (default: 42)

### 2. Train Traffic Light Agent

```bash
# Basic training
python training/train_traffic.py

# Train with PPO and 1M timesteps
python training/train_traffic.py --algorithm ppo --timesteps 1000000

# Train with A2C
python training/train_traffic.py --algorithm a2c
```

### 3. Train Smart Irrigation Agent

```bash
# Basic training
python training/train_agriculture.py

# Train with 5 crops
python training/train_agriculture.py --num-crops 5

# Train with DQN
python training/train_agriculture.py --algorithm dqn
```

## Expected Training Times

- **Waste Management**: ~30 minutes (500k timesteps)
- **Traffic Light**: ~30 minutes (500k timesteps)
- **Smart Irrigation**: ~40 minutes (500k timesteps)

## Outputs

All training outputs are saved to:
- **Models**: `models/{task_type}/`
- **Logs**: `results/logs/{task_type}/`
- **Plots**: `results/plots/`

## Viewing Training Progress

```bash
# Start tensorboard (in project directory)
tensorboard --logdir results/logs/
```

Then open `http://localhost:6006` in your browser.

## Advanced Usage

### Custom Environment Configuration

Modify environment parameters in training scripts:

```python
# In train_waste.py
env = WasteManagementEnv(
    grid_size=30,        # Larger grid
    num_bins=20,         # More bins
    max_steps=1000,      # Longer episodes
    capacity=200,        # Larger truck capacity
)
```

### Using Different Algorithms

```bash
# PPO (best for complex tasks)
python training/train_waste.py --algorithm ppo

# DQN (good for discrete actions)
python training/train_traffic.py --algorithm dqn

# A2C (faster convergence)
python training/train_agriculture.py --algorithm a2c
```

### Multi-GPU Training

For better performance, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Out of Memory
Reduce batch size or number of parallel environments:
```bash
python training/train_waste.py --batch-size 32 --num-envs 2
```

### Slow Training
Increase number of parallel environments:
```bash
python training/train_waste.py --num-envs 8
```

### Unstable Training
Reduce learning rate:
```bash
python training/train_waste.py --lr 0.00001
```

## Next Steps

After training completes:

1. **Evaluate models**: `python evaluation/evaluate.py`
2. **Visualize results**: Check `results/plots/`
3. **Compare algorithms**: Run same task with different algorithms
4. **Deploy model**: Load and use trained model in production

## Performance Metrics

### Waste Management
- Success metric: Collect all bins and return to depot
- Target: < 100 steps per episode
- Expected: 15-25% distance reduction vs baseline

### Traffic Light
- Success metric: Minimize average waiting time
- Target: < 30 seconds average wait
- Expected: 20-35% improvement

### Smart Irrigation
- Success metric: Maximize crop yield, minimize water
- Target: 0.8+ water efficiency ratio
- Expected: 25-40% water reduction

## Tips for Better Training

1. **Start with default settings** - They're optimized for most cases
2. **Monitor tensorboard** - Watch loss curves during training
3. **Use longer training** - 1M timesteps usually beats 500k
4. **Parallel environments** - More = faster training but more memory
5. **Save checkpoints** - Models auto-save at intervals
6. **Evaluate regularly** - Check validation performance

## Resources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym/Gymnasium](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep RL Algorithms](https://arxiv.org/)

## Support

For issues or questions:
1. Check error messages carefully
2. Verify all dependencies installed: `pip list`
3. Check environment variables: `echo $PYTHONPATH`
4. Review tensorboard logs for training progress
5. Try with smaller dataset/shorter training first
