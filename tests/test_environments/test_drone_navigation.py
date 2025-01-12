# tests/test_environments/test_drone_navigation.py
"""Tests for drone environment with navigation enabled"""
import pytest
import numpy as np
import yaml
from pathlib import Path
import pybullet as p

from src.environments.drone_env import DroneEnv


@pytest.fixture
def nav_env_config(tmp_path):
    """Create test environment config with navigation enabled"""
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
            },
            'navigation': {
                'enabled': True,
                'waypoints': {
                    'default_radius': 0.5,
                    'default_speed': 0.5,
                    'min_height': 0.5
                },
                'rewards': {
                    'waypoint_reached': 10.0,
                    'progress': 1.0,
                    'heading': 0.5,
                    'speed': 0.5
                },
                'initial_waypoints': [
                    {
                        'position': [1.0, 0.0, 1.0],
                        'radius': 0.5
                    },
                    {
                        'position': [1.0, 1.0, 1.0],
                        'radius': 0.5
                    }
                ]
            }
        }
    }

    config_path = tmp_path / "test_nav_env_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

    return str(config_path)


def test_navigation_initialization(nav_env_config):
    """Test environment initialization with navigation enabled"""
    env = DroneEnv(config_path=nav_env_config)

    assert env.waypoint_manager is not None
    assert len(env.waypoint_manager.waypoints) == 2

    # Check observation space includes navigation
    assert env.observation_space.shape[0] == 17  # Base state (12) + waypoint dir (3) + dist (1) + progress (1)


def test_navigation_state(nav_env_config):
    """Test navigation state in observations"""
    env = DroneEnv(config_path=nav_env_config)
    state, _ = env.reset()

    # Check state vector components
    position = state[0:3]  # Position [x, y, z]
    orientation = state[3:6]  # Orientation [roll, pitch, yaw]
    linear_vel = state[6:9]  # Linear velocity [vx, vy, vz]
    angular_vel = state[9:12]  # Angular velocity [wx, wy, wz]
    waypoint_dir = state[12:15]  # Direction to waypoint [dx, dy, dz]
    waypoint_dist = state[15]  # Distance to waypoint
    path_progress = state[16]  # Path progress [0-1]

    # Verify state dimensions
    assert state.shape[0] == 17
    assert len(waypoint_dir) == 3

    # Verify state properties
    assert np.all(np.abs(waypoint_dir) <= 1.0)  # Direction vector should be normalized
    assert waypoint_dist >= 0.0  # Distance can't be negative
    assert 0.0 <= path_progress <= 1.0  # Progress should be between 0 and 1


def test_waypoint_rewards(nav_env_config):
    """Test reward calculation with waypoints"""
    env = DroneEnv(config_path=nav_env_config)
    state, _ = env.reset()

    # Move drone to first waypoint
    initial_reward = env.compute_reward(state, np.zeros(4), False)

    # Create state near waypoint
    waypoint_pos = env.waypoint_manager.get_current_waypoint().position
    state[0:3] = waypoint_pos  # Set position to waypoint

    near_waypoint_reward = env.compute_reward(state, np.zeros(4), False)
    assert near_waypoint_reward > initial_reward


def test_path_completion(nav_env_config):
    """Test completing the navigation path"""
    env = DroneEnv(config_path=nav_env_config)
    state, _ = env.reset()

    # Print initial waypoint configuration
    current_wp = env.waypoint_manager.get_current_waypoint()
    print(f"\nInitial waypoint position: {current_wp.position}")
    print(f"Initial waypoint radius: {current_wp.radius}")
    print(f"Initial drone position: {state[0:3]}\n")

    # Get first waypoint and try to reach it
    waypoint = env.waypoint_manager.waypoints[0]

    # Reset drone to waypoint position
    p.resetBasePositionAndOrientation(
        env.drone_id,
        waypoint.position,
        p.getQuaternionFromEuler([0, 0, 0])
    )
    p.resetBaseVelocity(env.drone_id, [0, 0, 0], [0, 0, 0])
    p.stepSimulation()  # Let physics settle

    # Take one step and check
    state, reward, terminated, truncated, info = env.step(np.array([0.5, 0.5, 0.5, 0.5]))
    print(f"After step:")
    print(f"Drone position: {state[0:3]}")
    print(f"Info: {info}")

    assert info['waypoint_reached'], "Failed to reach waypoint"


def test_navigation_reset(nav_env_config):
    """Test environment reset with navigation"""
    env = DroneEnv(config_path=nav_env_config)

    # Complete first waypoint
    state, _ = env.reset()
    waypoint = env.waypoint_manager.get_current_waypoint()
    state[0:3] = waypoint.position
    env.step(np.array([0.5, 0.5, 0.5, 0.5]))

    # Reset and verify navigation state
    reset_state, reset_info = env.reset()
    assert reset_info['path_progress'] == 0.0
    assert not reset_info['waypoint_reached']
    assert env.waypoint_manager.current_index == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])