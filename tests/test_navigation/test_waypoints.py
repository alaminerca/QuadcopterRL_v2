# tests/test_navigation/test_waypoints.py
"""Tests for waypoint navigation system"""
import pytest
import numpy as np
from src.components.navigation.waypoint_manager import WaypointManager, Waypoint


@pytest.fixture
def waypoint_config():
    """Test configuration for waypoint manager"""
    return {
        'default_radius': 0.5,
        'default_speed': 0.5,
        'min_height': 0.5,
        'use_trajectories': False  # Test base functionality first
    }


@pytest.fixture
def trajectory_config():
    """Test configuration with trajectories enabled"""
    return {
        'default_radius': 0.5,
        'default_speed': 0.5,
        'min_height': 0.5,
        'use_trajectories': True,
        'trajectory': {
            'max_velocity': 2.0,
            'max_acceleration': 1.0,
            'curve_resolution': 20
        }
    }


def test_waypoint_creation(waypoint_config):
    """Test basic waypoint creation and management"""
    manager = WaypointManager(waypoint_config)

    manager.add_waypoint([1.0, 0.0, 1.0])
    manager.add_waypoint([2.0, 0.0, 1.0], radius=0.3)

    assert len(manager.waypoints) == 2
    assert manager.waypoints[0].radius == 0.5  # Default radius
    assert manager.waypoints[1].radius == 0.3  # Custom radius
    assert manager.total_distance == 1.0  # Distance between points


def test_waypoint_update(waypoint_config):
    """Test waypoint reaching and updates"""
    manager = WaypointManager(waypoint_config)

    # Add waypoints in straight line
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([1.0, 0.0, 1.0])

    # Test not reaching waypoint
    reached, distance = manager.update(np.array([0.0, 0.0, 0.8]))
    assert not reached
    assert np.isclose(distance, 0.2)
    assert manager.current_index == 0

    # Test reaching waypoint needs stability
    pos = np.array([0.1, 0.0, 1.0])
    for _ in range(manager.required_stable_steps - 1):
        reached, _ = manager.update(pos)
        assert not reached  # Not enough stable steps yet

    # Final step should reach waypoint
    reached, _ = manager.update(pos)
    assert reached
    assert manager.current_index == 1


def test_direction_vectors(waypoint_config):
    """Test direction vector calculations"""
    manager = WaypointManager(waypoint_config)
    manager.add_waypoint([1.0, 0.0, 1.0])

    direction = manager.get_direction_to_waypoint(np.array([0.0, 0.0, 1.0]))
    assert np.allclose(direction, [1.0, 0.0, 0.0])


def test_minimum_height(waypoint_config):
    """Test minimum height enforcement"""
    manager = WaypointManager(waypoint_config)

    # Try to add waypoint below minimum height
    manager.add_waypoint([1.0, 0.0, 0.2])

    assert manager.waypoints[0].position[2] == 0.5  # Should be raised to min height


def test_path_progress(waypoint_config):
    """Test path progress calculation"""
    manager = WaypointManager(waypoint_config)

    # Empty path
    assert manager.get_path_progress() == 0.0

    # Add waypoints
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([1.0, 0.0, 1.0])
    manager.add_waypoint([2.0, 0.0, 1.0])

    # Test progress updates
    assert manager.get_path_progress() == 0.0

    # Reach first waypoint (need stable steps)
    pos = np.array([0.1, 0.0, 1.0])
    for _ in range(manager.required_stable_steps):
        manager.update(pos)

    assert manager.get_path_progress() == 0.5


def test_trajectory_integration(trajectory_config):
    """Test waypoint manager with trajectory generation"""
    manager = WaypointManager(trajectory_config)

    # Add waypoints
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([1.0, 0.0, 1.0])
    manager.add_waypoint([1.0, 1.0, 1.0])

    # Should have trajectory generator
    assert manager.trajectory_generator is not None

    # Should generate lookahead points
    points = manager.get_lookahead_points()
    assert len(points) > 0

    # Test direction with trajectory
    direction = manager.get_direction_to_waypoint(np.array([0.0, 0.0, 1.0]))
    assert np.all(np.abs(direction) <= 1.0)  # Should be normalized


def test_trajectory_progress(trajectory_config):
    """Test progress updates with trajectories"""
    manager = WaypointManager(trajectory_config)

    # Add waypoints
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([1.0, 0.0, 1.0])

    # Update with drone following trajectory
    positions = [
        [0.0, 0.0, 1.0],
        [0.2, 0.0, 1.0],
        [0.4, 0.0, 1.0],
        [0.6, 0.0, 1.0]
    ]

    for pos in positions:
        manager.update(np.array(pos))
        points = manager.get_lookahead_points()
        assert len(points) > 0