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
        'min_height': 0.5
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

    # Test reaching waypoint
    reached, distance = manager.update(np.array([0.1, 0.0, 1.0]))
    assert reached
    assert manager.current_index == 1

    # Test path completion
    reached, distance = manager.update(np.array([1.0, 0.0, 1.0]))
    assert reached
    assert manager.path_completed


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
    manager.update(np.array([0.1, 0.0, 1.0]))  # Reach first waypoint
    assert manager.get_path_progress() == 0.5

    # Complete path
    manager.update(np.array([1.1, 0.0, 1.0]))
    manager.update(np.array([2.0, 0.0, 1.0]))
    assert manager.get_path_progress() == 1.0