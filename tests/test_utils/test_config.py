import pytest
import os
import yaml
from src.utils.config_utils import ConfigManager, ConfigPaths, ConfigValidationError


@pytest.fixture
def temp_config_files(tmp_path):
    """Create temporary config files for testing"""
    # Create env config
    env_config = {
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

    # Create training config
    training_config = {
        'training': {
            'algorithm': 'PPO',
            'total_timesteps': 1000000,
            'hyperparameters': {
                'learning_rate': 3e-4,
                'gamma': 0.99
            }
        }
    }

    # Write config files
    env_path = tmp_path / "env_config.yaml"
    train_path = tmp_path / "training_config.yaml"

    with open(env_path, 'w') as f:
        yaml.safe_dump(env_config, f)
    with open(train_path, 'w') as f:
        yaml.safe_dump(training_config, f)

    return {'env': str(env_path), 'training': str(train_path)}


def test_config_loading(temp_config_files):
    """Test basic config loading"""
    config_paths = ConfigPaths(
        env_config=temp_config_files['env'],
        training_config=temp_config_files['training']
    )

    config_manager = ConfigManager(config_paths)
    assert config_manager.config['environment']['simulation']['time_step'] == 0.02
    assert config_manager.config['training']['algorithm'] == 'PPO'


def test_validation_error(tmp_path, temp_config_files):
    """Test configuration validation"""
    # Create invalid config file
    invalid_config = {
        'environment': {
            'simulation': {
                'time_step': -1  # Invalid negative time step
            },
            'drone': {
                'mass': 0.7,
                'dimensions': [0.2, 0.2],  # Missing dimension
                'rotors': {'count': 4}
            },
            'physics': {
                'gravity': -9.81
            }
        }
    }

    invalid_path = tmp_path / "invalid_config.yaml"
    with open(invalid_path, 'w') as f:
        yaml.safe_dump(invalid_config, f)

    with pytest.raises(ConfigValidationError):
        ConfigManager(ConfigPaths(
            env_config=str(invalid_path),
            training_config=temp_config_files['training']
        ))


def test_config_update(temp_config_files):
    """Test configuration updates"""
    config_paths = ConfigPaths(
        env_config=temp_config_files['env'],
        training_config=temp_config_files['training']
    )

    config_manager = ConfigManager(config_paths)
    config_manager.set_config('environment.simulation.time_step', 0.05)

    assert config_manager.get_config('environment.simulation.time_step') == 0.05


def test_config_comparison(temp_config_files):
    """Test config comparison functionality"""
    config1 = ConfigManager(ConfigPaths(
        env_config=temp_config_files['env'],
        training_config=temp_config_files['training']
    ))

    config2 = ConfigManager(ConfigPaths(
        env_config=temp_config_files['env'],
        training_config=temp_config_files['training']
    ))

    # Modify config2
    config2.set_config('environment.simulation.time_step', 0.05)

    # Check differences
    diffs = config1.diff_configs(config2)
    assert 'environment.simulation.time_step' in diffs