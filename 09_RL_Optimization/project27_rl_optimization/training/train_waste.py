"""
Training script for Waste Management Optimization Agent
Uses PPO algorithm from Stable-Baselines3
"""

import os
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# Import custom environment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from environments.waste_management import WasteManagementEnv
from training.callbacks import create_callbacks


def train_waste_agent(
    algorithm: str = "ppo",
    total_timesteps: int = 500000,
    learning_rate: float = 0.0003,
    batch_size: int = 64,
    n_epochs: int = 10,
    num_envs: int = 4,
    seed: int = 42,
    save_dir: str = "models/waste/",
    log_dir: str = "results/logs/waste/",
    render: bool = False,
    verbose: int = 1
):
    """
    Train waste management agent.
    
    Args:
        algorithm: RL algorithm (ppo, dqn, a2c)
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        batch_size: Batch size for training
        n_epochs: Number of epochs for PPO
        num_envs: Number of parallel environments
        seed: Random seed
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        render: Whether to render environment
        verbose: Verbosity level
    """
    
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Training Waste Management Agent")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Number of parallel environments: {num_envs}")
    print(f"Random seed: {seed}")
    print(f"Run: {run_name}")
    print(f"{'='*60}\n")
    
    # Create vectorized environments
    def make_env():
        def _init():
            env = WasteManagementEnv(
                grid_size=20,
                num_bins=10,
                max_steps=500,
                seed=seed
            )
            return env
        return _init
    
    env = make_vec_env(make_env(), n_envs=num_envs, seed=seed)
    
    # Normalize observations
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create model
    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": learning_rate,
        "verbose": verbose,
        "tensorboard_log": log_dir,
        "seed": seed,
    }
    
    if algorithm.lower() == "ppo":
        model_kwargs.update({
            "n_steps": 2048,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        })
        model = PPO(**model_kwargs)
    elif algorithm.lower() == "dqn":
        model_kwargs.pop("verbose")
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=50000,
            learning_starts=1000,
            target_update_interval=1000,
            verbose=verbose,
            tensorboard_log=log_dir,
            seed=seed,
        )
    elif algorithm.lower() == "a2c":
        model = A2C(**model_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"Model created: {type(model).__name__}")
    print(f"Policy: MlpPolicy")
    print(f"Parameters: {model.num_parameters():,}\n")
    
    # Create evaluation environment
    eval_env = WasteManagementEnv(
        grid_size=20,
        num_bins=10,
        max_steps=500,
        seed=seed + 1000
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=50000,
        n_eval_episodes=10,
        deterministic=True,
        render=render,
    )
    
    callbacks = [eval_callback]
    
    # Train model
    print(f"Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_path = Path(save_dir) / f"{algorithm}_{timestamp}_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")
    
    # Save environment normalization
    env.save(str(Path(save_dir) / f"vec_normalize_{timestamp}.pkl"))
    print(f"Environment normalization saved")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print(f"Final Evaluation")
    print(f"{'='*60}")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(10):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        env_info = eval_env.get_info()
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Steps={episode_length}, "
              f"Distance={env_info['total_distance']:.2f}, "
              f"Collected={env_info['collection_rate']:.1%}")
    
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'='*60}\n")
    
    eval_env.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train waste management RL agent"
    )
    parser.add_argument(
        "--algorithm", type=str, default="ppo",
        choices=["ppo", "dqn", "a2c"],
        help="RL algorithm to use"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of epochs for PPO"
    )
    parser.add_argument(
        "--num-envs", type=int, default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level"
    )
    
    args = parser.parse_args()
    
    train_waste_agent(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        num_envs=args.num_envs,
        seed=args.seed,
        render=args.render,
        verbose=args.verbose
    )
