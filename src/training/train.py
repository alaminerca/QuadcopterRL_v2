# src/training/train.py
"""Training script for drone navigation"""
import os
import logging
from pathlib import Path
import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.preprocessing import get_obs_shape

from ..environments.drone_env import DroneEnv


class DroneTrainer:
    """Manages training of drone navigation policy"""

    def __init__(self, config_path: str, env_config_path: str):
        """
        Initialize trainer

        Args:
            config_path: Path to training configuration file
            env_config_path: Path to environment configuration file
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Store paths
        self.config_path = config_path
        self.env_config_path = env_config_path

        # Load configurations
        self.config = self._load_config(config_path)
        self.output_dir = Path("output")
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize environment with proper observation space normalization
        self.env = self._create_env()

        # Setup PPO model with correct tensor handling
        self.model = self._setup_model()

    def _load_config(self, path: str) -> dict:
        """Load training configuration"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config['training']

    def _create_env(self):
        """Create and vectorize environment with proper observation handling"""

        def make_env():
            env = DroneEnv(config_path=self.env_config_path)
            return env

        # Create vectorized environment
        vec_env = DummyVecEnv([make_env])

        # Add normalization wrapper with correct tensor handling
        normalized_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.config['hyperparameters']['gamma']
        )

        # Ensure observation space is correctly formatted
        obs_shape = get_obs_shape(normalized_env.observation_space)
        self.logger.info(f"Observation shape: {obs_shape}")

        return normalized_env

    def _setup_model(self) -> PPO:
        """Initialize PPO model with proper tensor handling"""
        params = self.config['hyperparameters']

        # Set device explicitly
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {device}")

        model = PPO(
            self.config['policy']['type'],
            self.env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=params['gamma'],
            gae_lambda=params['gae_lambda'],
            clip_range=params['clip_range'],
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            max_grad_norm=params['max_grad_norm'],
            tensorboard_log=str(self.log_dir),
            device=device,
            policy_kwargs={
                'net_arch': self.config['policy']['net_arch'],
                'activation_fn': getattr(torch.nn, self.config['policy']['activation_fn'])
            },
            verbose=1
        )

        return model

    def _create_callbacks(self) -> list:
        """Create training callbacks with proper error handling"""
        callbacks = []

        try:
            # Checkpoint saving
            if self.config['save_freq'] > 0:
                checkpoint_callback = CheckpointCallback(
                    save_freq=self.config['save_freq'],
                    save_path=str(self.model_dir),
                    name_prefix="drone_model",
                    save_vecnormalize=True
                )
                callbacks.append(checkpoint_callback)

            # Evaluation
            if self.config['eval_freq'] > 0:
                # Create separate environment for evaluation
                eval_env = self._create_env()

                eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=str(self.model_dir / "best_model"),
                    log_path=str(self.log_dir / "evaluation"),
                    eval_freq=self.config['eval_freq'],
                    n_eval_episodes=self.config['n_eval_episodes'],
                    deterministic=True
                )
                callbacks.append(eval_callback)

        except Exception as e:
            self.logger.error(f"Error creating callbacks: {e}")
            raise

        return callbacks

    def train(self, resume_from: str = None):
        """
        Train the model with proper error handling and tensor management

        Args:
            resume_from: Optional path to model to resume training from
        """
        try:
            # Load existing model if resuming
            if resume_from:
                self.logger.info(f"Loading model from {resume_from}")
                # Load both model and normalization stats
                self.model = PPO.load(
                    resume_from,
                    env=self.env,
                    custom_objects={
                        "learning_rate": self.config['hyperparameters']['learning_rate'],
                        "cliprange": self.config['hyperparameters']['clip_range']
                    }
                )

            callbacks = self._create_callbacks()

            # Start training with tensor error handling
            self.logger.info("Starting training...")
            # Disable progress bar in tests
            use_progress_bar = not bool(os.getenv('PYTEST_CURRENT_TEST'))

            self.model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=callbacks,
                progress_bar=use_progress_bar,
                reset_num_timesteps=resume_from is None
            )

            # Save final model and environment statistics
            final_model_path = self.model_dir / "final_model.zip"
            self.model.save(final_model_path)

            # Save normalization stats
            vec_stats_path = self.model_dir / "vec_normalize.pkl"
            self.env.save(str(vec_stats_path))

            self.logger.info(f"Training completed. Model saved to {final_model_path}")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        # src/training/train.py

    # src/training/train.py

    def evaluate(self, model_path: str, n_episodes: int = 10) -> dict:
        """
        Evaluate a trained model with proper tensor handling

        Args:
            model_path: Path to model to evaluate
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load model and environment normalization stats
            env_stats_path = Path(model_path).parent / "vec_normalize.pkl"
            eval_env = VecNormalize.load(
                str(env_stats_path),
                self._create_env()
            )

            # Don't update stats during evaluation
            eval_env.training = False
            eval_env.norm_reward = False

            # Load model
            model = PPO.load(model_path, env=eval_env)

            # Metrics storage
            episode_rewards = []
            episode_lengths = []
            waypoints_reached = []
            collisions = []

            # Run evaluation episodes
            for episode in range(n_episodes):
                # Reset environment
                state = eval_env.reset()
                if isinstance(state, tuple):
                    state = state[0]  # Newer versions return (state, info)

                done = False
                episode_reward = 0
                steps = 0
                waypoints = 0
                had_collision = False

                # Run episode
                while not done:
                    # Get action from model
                    action, _ = model.predict(state, deterministic=True)

                    # Step environment
                    next_state = eval_env.step(action)

                    # Handle different step return formats
                    if len(next_state) == 5:  # Newer gym versions
                        state, reward, term, trunc, info = next_state
                        done = term or trunc
                    else:  # Older gym versions
                        state, reward, done, info = next_state

                    # Update metrics
                    episode_reward += reward if np.isscalar(reward) else reward[0]
                    steps += 1

                    # Handle info dict from vectorized env
                    info_dict = info[0] if isinstance(info, (list, tuple)) else info
                    if info_dict.get('waypoint_reached', False):
                        waypoints += 1
                    if info_dict.get('collision', False):
                        had_collision = True

                # Store episode metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)
                waypoints_reached.append(waypoints)
                collisions.append(had_collision)

                # Log progress
                self.logger.info(
                    f"Episode {episode + 1}/{n_episodes}: "
                    f"Reward={episode_reward:.2f}, "
                    f"Steps={steps}, "
                    f"Waypoints={waypoints}, "
                    f"Collision={'Yes' if had_collision else 'No'}"
                )

            # Calculate final metrics
            metrics = {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'mean_episode_length': float(np.mean(episode_lengths)),
                'mean_waypoints_reached': float(np.mean(waypoints_reached)),
                'collision_rate': float(np.mean(collisions))
            }

            # Log final metrics
            self.logger.info("\nEvaluation Results:")
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.2f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise