"""
Training utilities and callbacks for RL models.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.n_steps = 0
    
    def _on_step(self) -> bool:
        # Log custom metrics
        if len(self.model.ep_info_buffer) > 0:
            mean_ep_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_ep_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            self.logger.record("rollouts/mean_episode_reward", mean_ep_reward)
            self.logger.record("rollouts/mean_episode_length", mean_ep_length)
        
        self.n_steps += 1
        
        return True


class ProgressCallback(BaseCallback):
    """
    Callback to print training progress.
    """
    
    def __init__(self, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_percent = 0
    
    def _on_step(self) -> bool:
        current = self.num_timesteps
        percent = int((current / self.total_timesteps) * 100)
        
        if percent % 10 == 0 and percent != self.last_percent:
            self.logger.info(f"Training progress: {percent}% ({current}/{self.total_timesteps} timesteps)")
            self.last_percent = percent
        
        return True


class MetricsCallback(BaseCallback):
    """
    Callback to track environment-specific metrics.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "custom_metrics": []
        }
    
    def _on_step(self) -> bool:
        # Track episode info
        for info in self.model.env.env_method("get_info"):
            self.metrics_history["custom_metrics"].append(info)
        
        return True
    
    def get_metrics(self) -> Dict[str, list]:
        """Get collected metrics."""
        return self.metrics_history
    
    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=4, default=str)


class CheckpointCallback(BaseCallback):
    """
    Callback to save model checkpoints at regular intervals.
    """
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "models/checkpoints/",
        name_prefix: str = "model",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(str(path))
            
            if self.verbose > 0:
                self.logger.info(f"Saving model checkpoint: {path}")
        
        return True


def create_callbacks(
    env_type: str,
    log_dir: str = "logs/",
    checkpoint_dir: str = "models/checkpoints/",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    verbose: int = 1
) -> list:
    """
    Create callback list for training.
    
    Args:
        env_type: Type of environment (waste, traffic, agriculture)
        log_dir: Directory for tensorboard logs
        checkpoint_dir: Directory for model checkpoints
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of episodes for evaluation
        verbose: Verbosity level
        
    Returns:
        List of callbacks
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        TensorboardCallback(verbose=verbose),
        ProgressCallback(total_timesteps=1000000, verbose=verbose),
        MetricsCallback(verbose=verbose),
        CheckpointCallback(
            save_freq=50000,
            save_path=checkpoint_dir,
            name_prefix=f"{env_type}_ppo",
            verbose=verbose
        )
    ]
    
    return callbacks


def load_callbacks(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Load callback history from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint logs
        
    Returns:
        Dictionary of callback data
    """
    data = {}
    
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        for file in checkpoint_path.glob("*.json"):
            with open(file, 'r') as f:
                data[file.stem] = json.load(f)
    
    return data
