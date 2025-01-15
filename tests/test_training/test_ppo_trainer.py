# tests/test_training/test_ppo_trainer.py
"""Tests for PPO Trainer implementation"""
import pytest
import yaml
import numpy as np
import torch
from pathlib import Path
from src.training.ppo_trainer import PPOTrainer
from src.utils.config_utils import ConfigManager, ConfigPaths


@pytest.fixture
def test_configs(tmp_path):
    """Create test configurations"""
    train_config = {
        'training': {
            'total_timesteps': 100,
            'algorithm': 'PPO',
            'hyperparameters': {
                'learning_rate': 3.0e-4,
                'n_steps': 64,
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
            }
        }
    }

    train_path = tmp_path / "test_training_config.yaml"
    env_path = tmp_path / "test_env_config.yaml"

    with open(train_path, 'w') as f:
        yaml.safe_dump(train_config, f)
    with open(env_path, 'w') as f:
        yaml.safe_dump(env_config, f)

    return ConfigManager(ConfigPaths(
        env_config=str(env_path),
        training_config=str(train_path)
    ))


@pytest.fixture(scope='function')
def trainer(test_configs):
    """Create trainer instance for tests"""
    torch.set_default_dtype(torch.float32)
    return PPOTrainer(test_configs)


def test_trainer_initialization(trainer):
    """Test initialization"""
    assert trainer.model is not None
    assert trainer.env is not None

    obs = trainer.env.reset()[0]
    assert isinstance(obs, np.ndarray)
    obs_tensor = trainer._obs_to_tensor(obs)
    assert isinstance(obs_tensor, torch.Tensor)
    assert obs_tensor.dtype == torch.float32


def test_short_training(trainer):
    """Test short training run"""
    trainer.train(total_timesteps=10)
    assert (trainer.model_dir / "final_model.zip").exists()


def test_evaluation(trainer):
    """Test evaluation"""
    trainer.train(total_timesteps=10)
    metrics = trainer.evaluate(n_episodes=2)

    assert 'mean_reward' in metrics
    assert 'mean_episode_length' in metrics
    assert isinstance(metrics['mean_reward'], float)
    assert isinstance(metrics['mean_episode_length'], float)


def test_tensor_device_handling(trainer):
    """Test tensor device handling"""
    device = trainer.model.device
    assert isinstance(device, torch.device)

    obs = trainer.env.reset()[0]
    obs_tensor = trainer._obs_to_tensor(obs)
    assert obs_tensor.device == device