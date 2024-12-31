import pytest
import numpy as np
import yaml
from pathlib import Path

from src.environments.drone_env import DroneEnv


@pytest.fixture
def test_env_config(tmp_path):
    """Create test environment config with obstacles"""
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
            },
            'obstacles': {
                'enabled': True,
                'static': {
                    'walls': [{
                        'start': [1.0, -1.0, 0.0],
                        'end': [1.0, 1.0, 0.0],
                        'height': 2.0,
                        'thickness': 0.1
                    }]
                }
            }
        }
    }

    config_path = tmp_path / "test_env_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

    return str(config_path)


def test_obstacle_initialization(test_env_config):
    """Test if obstacles are properly initialized"""
    env = DroneEnv(config_path=test_env_config)

    # Check if obstacle exists
    obstacles = env.obstacle_manager.get_all_obstacles()
    assert len(obstacles) > 0
    print(f"Found {len(obstacles)} obstacles")


def test_collision_detection(test_env_config):
    """Test collision detection and reward penalty"""
    env = DroneEnv(config_path=test_env_config)
    state, _ = env.reset()

    # Move drone towards wall with constant thrust
    for _ in range(50):
        state, reward, terminated, _, _ = env.step([0.8, 0.8, 0.8, 0.8])
        pos = state[0:3]
        print(f"Position: {pos}, Reward: {reward}")
        if terminated:
            print("Collision detected!")
            break

    # Should have lower reward near wall
    assert reward < 0


def test_obstacle_avoidance_reward(test_env_config):
    env = DroneEnv(config_path=test_env_config)
    state, _ = env.reset()

    _, reward1, _, _, _ = env.step([0.5, 0.5, 0.5, 0.5])

    for _ in range(20):
        state, reward2, _, _, _ = env.step([0.7, 0.7, 0.7, 0.7])

    assert abs(reward2) > abs(reward1)  # Higher penalty = larger absolute value
    print(f"Baseline reward: {reward1}")
    print(f"Near obstacle reward: {reward2}")

