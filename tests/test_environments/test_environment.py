# tests/test_environments/test_environment.py
"""Tests for Drone Environment"""
import time
import pytest
import os
from pathlib import Path
import yaml

from src.environments.drone_env import DroneEnv
from src.utils.config_utils import ConfigPaths


@pytest.fixture
def test_configs(tmp_path):
    """
    Create temporary test configs

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        str: Path to temporary config file
    """
    # Create minimal env config for testing
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
                    'radius': 0.02,
                    'height': 0.005,
                    'positions': [
                        [0.08, 0.08, 0.51],  # Front Right
                        [-0.08, 0.08, 0.51],  # Front Left
                        [-0.08, -0.08, 0.51],  # Rear Left
                        [0.08, -0.08, 0.51]  # Rear Right
                    ],
                    'colors': [
                        [1.0, 0.0, 0.0, 0.8],  # Red
                        [0.0, 1.0, 0.0, 0.8],  # Green
                        [1.0, 1.0, 0.0, 0.8],  # Yellow
                        [1.0, 0.5, 0.0, 0.8]  # Orange
                    ]
                }
            },
            'physics': {
                'gravity': -9.81,
                'debug': False,
                'wind': {
                    'enabled': False,
                    'base_magnitude': 0.0,
                    'variability': 0.0
                },
                'drag': {
                    'enabled': True,
                    'coefficient': 0.5
                }
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
            }
        }
    }

    # Write config to temp directory
    env_path = tmp_path / "test_env.yaml"
    with open(env_path, 'w') as f:
        yaml.safe_dump(env_config, f)

    return str(env_path)


def test_drone_hovering(test_configs):
    """Test basic drone hovering behavior"""
    # Initialize environment with test config
    env = DroneEnv(config_path=test_configs)

    try:
        # Reset and run simulation
        state, _ = env.reset()
        initial_height = state[2]

        for i in range(100):  # Reduced steps for testing
            # Apply equal thrust to all rotors
            action = [0.5, 0.5, 0.5, 0.5]
            state, reward, terminated, truncated, info = env.step(action)

            # Log progress every 20 steps
            if i % 20 == 0:
                print(f"Step {i:3d} - Height: {state[2]:.2f}m, Reward: {reward:.2f}")

            if terminated or truncated:
                print(f"Episode ended at step {i}")
                break

            time.sleep(1 / 240)  # Slow down for visualization

    finally:
        env.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])