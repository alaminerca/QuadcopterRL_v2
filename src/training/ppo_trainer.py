# src/training/ppo_trainer.py
"""PPO Trainer implementation for drone environment"""
from typing import Dict, Optional
import logging
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..environments.drone_env import DroneEnv
from ..utils.config_utils import ConfigManager


class PPOTrainer:
    """PPO training implementation with proper tensor handling"""

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)

        self.output_dir = Path("output")
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self._setup_directories()

        self.env_config_path = self.config.config_paths.env_config
        self.env = self._create_env()
        self.model = self._setup_model()

    def _setup_directories(self) -> None:
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

    def _create_env(self):
        def make_env():
            return DroneEnv(config_path=self.env_config_path)

        env = DummyVecEnv([make_env])

        # Remove normalization to simplify testing
        return env

    def _setup_model(self) -> PPO:
        train_config = self.config.get_config('training')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure policy and network architecture
        policy_kwargs = {
            'net_arch': train_config['policy']['net_arch'],
            'activation_fn': getattr(torch.nn, train_config['policy']['activation_fn'])
        }

        return PPO(
            train_config['policy']['type'],
            self.env,
            learning_rate=train_config['hyperparameters']['learning_rate'],
            n_steps=train_config['hyperparameters']['n_steps'],
            batch_size=train_config['hyperparameters']['batch_size'],
            n_epochs=train_config['hyperparameters']['n_epochs'],
            gamma=train_config['hyperparameters']['gamma'],
            gae_lambda=train_config['hyperparameters']['gae_lambda'],
            clip_range=train_config['hyperparameters']['clip_range'],
            ent_coef=train_config['hyperparameters']['ent_coef'],
            vf_coef=train_config['hyperparameters']['vf_coef'],
            max_grad_norm=train_config['hyperparameters']['max_grad_norm'],
            tensorboard_log=str(self.log_dir),
            device=device,
            policy_kwargs=policy_kwargs
        )

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert observation to tensor with correct dtype"""
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.model.device)

    def train(self, total_timesteps: Optional[int] = None) -> None:
        try:
            timesteps = total_timesteps or self.config.get_config('training.total_timesteps')
            self.model.learn(
                total_timesteps=timesteps,
                progress_bar=False
            )
            self.model.save(str(self.model_dir / "final_model.zip"))
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs = self.env.reset()[0]
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = self.model.predict(
                    self._obs_to_tensor(obs),
                    deterministic=True
                )
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths))
        }