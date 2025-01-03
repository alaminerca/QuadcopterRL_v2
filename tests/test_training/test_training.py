"""Test module for training system"""
import pytest
import yaml
from pathlib import Path
from src.training.training_config import TrainingConfig
from src.training.training_manager import TrainingManager


@pytest.fixture
def test_env_config(tmp_path):
    """Create temporary environment config file"""
    config = {
        'environment': {
            'simulation': {
                'time_step': 0.02,
                'max_steps': 1000
            },
            'drone': {
                'mass': 0.7,
                'dimensions': [0.2, 0.2, 0.1],
                'target_height': 1.0,
                'rotors': {
                    'count': 4,
                    'mass': 0.05,
                    'max_thrust': 2.58,
                    'positions': [
                        [0.1, 0.1, 0.51],
                        [-0.1, 0.1, 0.51],
                        [-0.1, -0.1, 0.51],
                        [0.1, -0.1, 0.51]
                    ]
                }
            },
            'physics': {
                'gravity': -9.81,
                'debug': False
            }
        }
    }

    config_path = tmp_path / "test_env_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

    return str(config_path)


def test_training_setup(test_env_config):
    """Test if training system initializes correctly"""
    config = TrainingConfig(
        total_timesteps=10000,  # Short training
        save_freq=2000,  # Save more frequently
        log_freq=500,  # Log more frequently
        eval_freq=2000,  # Evaluate more frequently
        n_eval_episodes=2  # Fewer eval episodes
    )

    trainer = TrainingManager(
        config=config,
        env_config_path=test_env_config
    )

    assert trainer is not None
    assert trainer.config.total_timesteps == 10000


def test_short_training(test_env_config):
    """Test if training runs without errors"""
    config = TrainingConfig(
        total_timesteps=1000,  # Very short training
        save_freq=500,
        log_freq=100,
        eval_freq=500,
        n_eval_episodes=1
    )

    trainer = TrainingManager(
        config=config,
        env_config_path=test_env_config
    )

    # Should run without raising exceptions
    trainer.train()

    # Check if model was saved
    assert (trainer.model_dir / "final_model.zip").exists()


def test_evaluation(test_env_config):
    """Test model evaluation"""
    config = TrainingConfig(total_timesteps=1000)
    trainer = TrainingManager(
        config=config,
        env_config_path=test_env_config
    )

    trainer.train()
    metrics = trainer.evaluate(
        str(trainer.model_dir / "final_model.zip"),
        n_episodes=2
    )

    assert 'mean_reward' in metrics
    assert 'mean_episode_length' in metrics