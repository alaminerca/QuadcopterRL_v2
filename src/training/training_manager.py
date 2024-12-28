"""Training Manager module for drone RL"""
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from .training_config import TrainingConfig
from ..environments.drone_env import DroneEnv


class TrainingManager:
    """Manages the training process for the drone"""

    def __init__(self, config: TrainingConfig, env_config_path: str):
        """Initialize training manager"""
        # 1. Basic setup
        self.config = config
        self.env_config_path = env_config_path

        # 2. Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 3. Setup directories
        self.output_dir = Path("output")
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self._setup_directories()

        # 4. Initialize environment
        self.env = self._create_env()

        # 5. Initialize model
        self.model = self._create_model()

    def _setup_directories(self) -> None:
        """Create necessary directories"""
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

    def _create_env(self) -> gym.Env:
        """Create and vectorize environment"""
        env = DroneEnv(config_path=self.env_config_path)
        return DummyVecEnv([lambda: env])

    def _create_model(self) -> PPO:
        """Create PPO model"""
        return PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=str(self.log_dir),
            device=self.config.device,
            **self.config.get_sb3_params()
        )

    def _create_callbacks(self) -> list:
        """Create training callbacks"""
        callbacks = []

        # Checkpoint saving
        if self.config.save_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.save_freq,
                save_path=str(self.model_dir),
                name_prefix="drone_model"
            )
            callbacks.append(checkpoint_callback)

        # Evaluation
        if self.config.eval_freq > 0:
            # Create separate env for evaluation
            eval_env = DroneEnv(config_path=self.env_config_path)
            eval_env = DummyVecEnv([lambda: eval_env])

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir / "best_model"),
                log_path=str(self.log_dir / "evaluation"),
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True
            )
            callbacks.append(eval_callback)

        return callbacks

    def train(self, resume_from: Optional[str] = None) -> None:
        """Train the model"""
        if resume_from:
            self.logger.info(f"Loading model from {resume_from}")
            self.model = PPO.load(resume_from, env=self.env)

        callbacks = self._create_callbacks()

        try:
            self.logger.info("Starting training...")
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=True
            )

            # Save final model
            final_model_path = self.model_dir / "final_model.zip"
            self.model.save(final_model_path)
            self.logger.info(f"Training completed. Model saved to {final_model_path}")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def evaluate(self, model_path: str, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate a trained model"""
        model = PPO.load(model_path)
        env = DroneEnv(config_path=self.env_config_path)

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths)
        }

        return metrics

    def save_config(self) -> None:
        """Save training configuration"""
        config_path = self.output_dir / "training_config.yaml"
        self.config.save(config_path)