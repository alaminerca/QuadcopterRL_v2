# tests/test_training/test_training.py
"""Tests for training system"""
import os

import pytest
import yaml
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from src.training.train import DroneTrainer


@pytest.fixture
def test_configs(tmp_path):
    """Create test configurations"""
    # Training config
    train_config = {
        'training': {
            'total_timesteps': 1000,  # Short training for testing
            'save_freq': 100,
            'log_freq': 100,
            'eval_freq': 100,
            'n_eval_episodes': 2,
            'algorithm': 'PPO',
            'hyperparameters': {
                'learning_rate': 3.0e-4,
                'n_steps': 128,
                'batch_size': 32,
                'n_epochs': 3,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'policy': {
                'type': 'MlpPolicy',
                'net_arch': {
                    'pi': [32, 32],
                    'vf': [32, 32]
                },
                'activation_fn': 'ReLU'
            }
        }
    }

    # Environment config
    env_config = {
        'environment': {
            'simulation': {
                'time_step': 0.02,
                'max_steps': 100
            },
            'drone': {
                'mass': 0.7,
                'dimensions': [0.2, 0.2, 0.1],
                'target_height': 1.0,
                'rotors': {
                    'count': 4,
                    'mass': 0.05,
                    'max_thrust': 2.58,
                    'radius': 0.02,
                    'height': 0.005,
                    'positions': [
                        [0.08, 0.08, 0.51],
                        [-0.08, 0.08, 0.51],
                        [-0.08, -0.08, 0.51],
                        [0.08, -0.08, 0.51]
                    ]
                }
            },
            'physics': {
                'gravity': -9.81,
                'debug': False
            },
            'obstacles': {
                'enabled': False,
                'static': {
                    'walls': [],
                    'boxes': []
                },
                'dynamic': {
                    'enabled': False,
                    'moving_obstacles': []
                }
            },
            'navigation': {
                'enabled': True,
                'waypoints': {
                    'default_radius': 0.5,
                    'default_speed': 0.5,
                    'min_height': 0.5
                }
            }
        }
    }

    # Save configs
    train_path = tmp_path / "test_training_config.yaml"
    env_path = tmp_path / "test_env_config.yaml"

    with open(train_path, 'w') as f:
        yaml.safe_dump(train_config, f)
    with open(env_path, 'w') as f:
        yaml.safe_dump(env_config, f)

    return {
        'training': str(train_path),
        'environment': str(env_path)
    }


@pytest.fixture
def clean_output_dirs(tmp_path):
    """Ensure clean output directories for each test"""
    output_dir = Path("output")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    return tmp_path


def test_trainer_initialization(test_configs, clean_output_dirs):
    """Test trainer initialization"""
    trainer = DroneTrainer(test_configs['training'], test_configs['environment'])
    assert trainer is not None
    assert trainer.model is not None
    assert isinstance(trainer.env, VecNormalize)


def test_short_training(test_configs, clean_output_dirs):
    """Test short training run"""
    # Initialize trainer
    trainer = DroneTrainer(test_configs['training'], test_configs['environment'])

    # Run short training with progress bar disabled
    os.environ['PYTEST_CURRENT_TEST'] = 'True'  # Mark as running in pytest
    try:
        # Run training
        trainer.train()

        # Verify model files were saved
        model_path = trainer.model_dir / "final_model.zip"
        stats_path = trainer.model_dir / "vec_normalize.pkl"
        assert model_path.exists(), "Model file was not created"
        assert stats_path.exists(), "Environment stats file was not created"

        # Verify model can be loaded
        loaded_model = PPO.load(str(model_path))
        assert loaded_model is not None, "Failed to load saved model"

        # Verify environment stats can be loaded
        loaded_env = VecNormalize.load(str(stats_path), trainer.env)
        assert loaded_env is not None, "Failed to load environment stats"

        # Verify model architecture
        assert hasattr(loaded_model, 'policy'), "Model missing policy network"
        assert hasattr(loaded_model, 'policy_kwargs'), "Model missing policy configuration"

        # Verify policy architecture matches configuration
        policy_arch = loaded_model.policy_kwargs.get('net_arch', {})
        config_arch = trainer.config['policy']['net_arch']
        assert policy_arch == config_arch, "Model architecture doesn't match configuration"

    finally:
        # Clean up test environment
        os.environ.pop('PYTEST_CURRENT_TEST', None)

        # Clean up PyBullet
        if hasattr(trainer, 'env'):
            trainer.env.close()


def test_model_evaluation(test_configs, clean_output_dirs):
    """Test model evaluation"""
    trainer = DroneTrainer(test_configs['training'], test_configs['environment'])
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate(
        str(trainer.model_dir / "final_model.zip"),
        n_episodes=2
    )

    # Check metrics
    required_metrics = ['mean_reward', 'mean_waypoints_reached', 'collision_rate']
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float, np.number))


def test_training_resumption(test_configs, clean_output_dirs):
    """Test resuming training from a saved model"""
    # Initialize first trainer
    trainer1 = DroneTrainer(test_configs['training'], test_configs['environment'])
    trainer2 = None  # Initialize to None for proper cleanup
    os.environ['PYTEST_CURRENT_TEST'] = 'True'  # Disable progress bar for tests

    try:
        # Step 1: Initial training
        trainer1.train()
        initial_model_path = trainer1.model_dir / "final_model.zip"
        assert initial_model_path.exists(), "Initial model was not saved"

        # Step 2: Create new trainer and resume training
        trainer2 = DroneTrainer(test_configs['training'], test_configs['environment'])
        trainer2.train(resume_from=str(initial_model_path))

        # Step 3: Verify model files exist
        resumed_model_path = trainer2.model_dir / "final_model.zip"
        assert resumed_model_path.exists(), "Resumed model was not saved"

        # Verify both models can be loaded
        initial_model = PPO.load(str(initial_model_path))
        resumed_model = PPO.load(str(resumed_model_path))
        assert initial_model is not None
        assert resumed_model is not None

    finally:
        # Clean up
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        if hasattr(trainer1, 'env'):
            trainer1.env.close()
        if trainer2 is not None and hasattr(trainer2, 'env'):
            trainer2.env.close()

        # Cleanup PyBullet
        import pybullet as p
        if p.isConnected():
            p.disconnect()