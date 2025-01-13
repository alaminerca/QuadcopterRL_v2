# tests/test_navigation/test_navigation_integration.py
"""Integration tests for navigation components"""
import pytest
import numpy as np
from src.components.navigation.waypoint_manager import WaypointManager
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def nav_config():
    """Test configuration for navigation system"""
    return {
        'default_radius': 0.5,
        'default_speed': 0.5,
        'min_height': 0.5,
        'use_trajectories': True,
        'trajectory': {
            'max_velocity': 2.0,
            'max_acceleration': 1.0,
            'curve_resolution': 20
        },
        'optimizer': {
            'min_distance': 0.5,
            'max_velocity': 2.0,
            'max_acceleration': 1.0,
            'smoothing_factor': 0.1,
            'path_resolution': 0.1
        },
        'visualization': {
            'enabled': False  # Disable for testing
        }
    }


def test_optimized_trajectory_generation(nav_config):
    """Test integration of optimization and trajectory generation"""
    manager = WaypointManager(nav_config)

    # Add waypoints in zigzag pattern
    waypoints = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0]
    ]

    for pos in waypoints:
        manager.add_waypoint(pos)

    # Verify trajectory was generated
    assert manager.trajectory_generator is not None
    assert len(manager.trajectory_generator.curves) == len(waypoints) - 1

    # Check lookahead points
    points = manager.get_lookahead_points()
    assert len(points) > 0

    # Verify points follow a smoother path than original zigzag
    # Calculate maximum angle in the path
    max_angle = 0
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        max_angle = max(max_angle, angle)

    # Smoothed path should have less sharp turns
    assert np.degrees(max_angle) < 90


def test_velocity_constraints(nav_config):
    """Test velocity and acceleration constraints are maintained"""
    manager = WaypointManager(nav_config)

    # Add waypoints with long distance to reach max velocity
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([5.0, 0.0, 1.0])

    # Get path points
    points = manager.get_lookahead_points()

    # Get velocities from trajectory generator
    velocities = manager.trajectory_generator.velocities

    if velocities:  # If velocities are stored
        # Check velocity limits
        assert all(v <= nav_config['trajectory']['max_velocity'] for v in velocities)

        # Check acceleration limits
        for i in range(1, len(velocities)):
            accel = abs(velocities[i] - velocities[i - 1]) / nav_config['optimizer']['path_resolution']
            assert accel <= nav_config['trajectory']['max_acceleration'] * 1.1  # Allow small numerical errors


def test_height_constraints(nav_config):
    """Test minimum height constraints are maintained"""
    manager = WaypointManager(nav_config)

    # Try to add waypoints below minimum height
    manager.add_waypoint([0.0, 0.0, 0.2])  # Below min_height
    manager.add_waypoint([1.0, 0.0, 0.3])  # Below min_height

    # Check all path points respect minimum height
    points = manager.get_lookahead_points()
    min_height = nav_config['min_height']

    assert all(point[2] >= min_height for point in points)

    # Check waypoints were adjusted
    assert all(wp.position[2] >= min_height for wp in manager.waypoints)


def test_path_following(nav_config, caplog):
    """Test path following behavior"""
    # Set up logging for test
    caplog.set_level(logging.DEBUG)

    manager = WaypointManager(nav_config)
    logger.info("Starting path following test")

    # Add waypoints
    waypoints = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ]

    for pos in waypoints:
        manager.add_waypoint(pos)

    # Test progression through waypoints
    pos = np.array([0.0, 0.0, 1.0])  # Start at first waypoint
    logger.info(f"Testing with {manager.required_stable_steps} required stable steps")

    # Verify first waypoint not reached immediately (needs stable steps)
    reached, _ = manager.update(pos)
    assert not reached
    assert manager.current_index == 0

    # Run updates until just before reaching stability
    for step in range(manager.required_stable_steps - 1):
        reached, _ = manager.update(pos)
        assert not reached  # Should not be reached yet
        logger.debug(f"Step {step + 1}: stable_count={manager.stable_count}")

    # Final update should reach the waypoint
    logger.info("Executing final update")
    reached, _ = manager.update(pos)

    # Verify final state
    logger.info(f"Final state - reached: {reached}, index: {manager.current_index}")
    assert reached  # Should be reached now
    assert manager.current_index == 1  # Should advance to next waypoint


def test_reset_behavior(nav_config):
    """Test navigation system reset"""
    manager = WaypointManager(nav_config)

    # Add waypoints
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([1.0, 0.0, 1.0])

    # Progress to first waypoint
    pos = np.array([0.0, 0.0, 1.0])
    for _ in range(manager.required_stable_steps):
        manager.update(pos)

    # Reset navigation
    manager.reset()

    # Verify reset state
    assert manager.current_index == 0
    assert not manager.path_completed
    assert manager.stable_count == 0

    # Verify trajectory was regenerated
    points = manager.get_lookahead_points()
    assert len(points) > 0