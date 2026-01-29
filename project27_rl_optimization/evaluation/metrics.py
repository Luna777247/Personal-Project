"""
Evaluation and metrics module for RL agents.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class RLMetrics:
    """Compute and track RL performance metrics."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.custom_metrics = {}
    
    def add_episode(
        self,
        reward: float,
        length: int,
        success: bool = False,
        metrics: Dict[str, Any] = None
    ):
        """Add episode data."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_success.append(success)
        
        if metrics:
            for key, value in metrics.items():
                if key not in self.custom_metrics:
                    self.custom_metrics[key] = []
                self.custom_metrics[key].append(value)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
            "std_reward": float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
            "max_reward": float(np.max(self.episode_rewards)) if self.episode_rewards else 0,
            "min_reward": float(np.min(self.episode_rewards)) if self.episode_rewards else 0,
            "mean_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0,
            "success_rate": float(np.mean(self.episode_success)) if self.episode_success else 0,
        }
    
    def save(self, filepath: str):
        """Save metrics to file."""
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_success": self.episode_success,
            "custom_metrics": self.custom_metrics,
            "summary": self.get_summary(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
    
    def load(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.episode_rewards = data.get("episode_rewards", [])
        self.episode_lengths = data.get("episode_lengths", [])
        self.episode_success = data.get("episode_success", [])
        self.custom_metrics = data.get("custom_metrics", {})


def evaluate_agent(
    model,
    env,
    num_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
) -> Tuple[float, float, RLMetrics]:
    """
    Evaluate trained agent.
    
    Args:
        model: Trained model
        env: Environment
        num_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        render: Render episodes
        
    Returns:
        mean_reward, std_reward, metrics
    """
    metrics = RLMetrics()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_metrics = {}
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        # Get environment-specific metrics
        if hasattr(env, 'get_info'):
            episode_metrics = env.get_info()
        
        metrics.add_episode(
            reward=episode_reward,
            length=episode_length,
            success=(episode_length < 500),  # Success if completed
            metrics=episode_metrics
        )
    
    summary = metrics.get_summary()
    return summary["mean_reward"], summary["std_reward"], metrics


def compare_agents(
    models_dict: Dict[str, Any],
    env,
    num_episodes: int = 10,
    metric_name: str = "mean_reward"
) -> Dict[str, float]:
    """
    Compare multiple agents.
    
    Args:
        models_dict: Dictionary of {name: model} pairs
        env: Environment
        num_episodes: Number of evaluation episodes
        metric_name: Metric to compare
        
    Returns:
        Dictionary of {model_name: metric_value}
    """
    results = {}
    
    for name, model in models_dict.items():
        mean_reward, std_reward, metrics = evaluate_agent(
            model, env, num_episodes=num_episodes
        )
        
        if metric_name == "mean_reward":
            results[name] = mean_reward
        elif metric_name == "success_rate":
            summary = metrics.get_summary()
            results[name] = summary["success_rate"]
        else:
            results[name] = metrics.custom_metrics.get(metric_name, [np.mean])[0]
    
    return results


def plot_learning_curve(
    rewards: List[float],
    window: int = 100,
    title: str = "Learning Curve",
    filepath: str = None
):
    """
    Plot learning curve with moving average.
    
    Args:
        rewards: List of episode rewards
        window: Moving average window
        title: Plot title
        filepath: Save to file if provided
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    
    # Plot moving average
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_metrics_comparison(
    metrics_dict: Dict[str, float],
    title: str = "Metrics Comparison",
    filepath: str = None
):
    """
    Plot metrics comparison across models.
    
    Args:
        metrics_dict: Dictionary of {model_name: metric_value}
        title: Plot title
        filepath: Save to file if provided
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.ylabel('Score')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_episodes_statistics(
    episode_rewards: List[float],
    episode_lengths: List[int],
    title: str = "Episode Statistics",
    filepath: str = None
):
    """
    Plot episode statistics side by side.
    
    Args:
        episode_rewards: Episode rewards
        episode_lengths: Episode lengths
        title: Plot title
        filepath: Save to file if provided
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rewards histogram
    axes[0].hist(episode_rewards, bins=20, color='steelblue', edgecolor='black')
    axes[0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(episode_rewards):.2f}')
    axes[0].set_xlabel('Reward')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Episode Rewards Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Episode lengths histogram
    axes[1].hist(episode_lengths, bins=20, color='coral', edgecolor='black')
    axes[1].axvline(np.mean(episode_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[1].set_xlabel('Episode Length')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Episode Lengths Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()
