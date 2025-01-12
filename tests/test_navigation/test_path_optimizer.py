# tests/test_navigation/test_path_optimizer.py
"""Tests for path optimization system"""
import pytest
import numpy as np
from src.components.navigation.path_optimizer import PathOptimizer


@pytest.fixture
def optimizer_config():
    """Test configuration for path optimizer"""
    return {
        'min_distance': 0.5,
        'max_velocity': 2.0,
        'max_acceleration': 1.0,
        'smoothing_factor': 0.1,
        'path_resolution': 0.1
    }


def test_path_smoothing(optimizer_config):
    """Test basic path smoothing"""
    optimizer = PathOptimizer(optimizer_config)

    # Create zigzag path
    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 1.0, 1.0])
    ]

    smooth_path, _ = optimizer.optimize_path(waypoints)

    # Check path properties
    assert len(smooth_path) > len(waypoints)  # Should add intermediate points
    assert np.allclose(smooth_path[0], waypoints[0])  # Should maintain start
    assert np.allclose(smooth_path[-1], waypoints[-1])  # Should maintain end


def test_velocity_optimization(optimizer_config):
    """Test velocity profile generation"""
    optimizer = PathOptimizer(optimizer_config)

    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([2.0, 0.0, 1.0])
    ]

    _, velocities = optimizer.optimize_path(waypoints)

    # Check velocity constraints
    assert velocities[0] == 0.0  # Start from rest
    assert all(0.0 <= v <= optimizer_config['max_velocity'] for v in velocities)

    # Check acceleration constraints
    for i in range(1, len(velocities)):
        acceleration = abs(velocities[i] - velocities[i - 1]) / optimizer.path_resolution
        assert acceleration <= optimizer_config['max_acceleration'] * 1.1  # Allow small numerical errors


def test_obstacle_avoidance(optimizer_config):
    """Test path adjustment for obstacle avoidance"""
    optimizer = PathOptimizer(optimizer_config)

    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([2.0, 0.0, 1.0])
    ]

    obstacles = [
        np.array([1.0, 0.0, 1.0])  # Obstacle directly in path
    ]

    path, _ = optimizer.optimize_path(waypoints, obstacles)

    # Check minimum distance to obstacle
    min_distance = min(
        np.linalg.norm(point - obstacles[0])
        for point in path
    )
    assert min_distance >= optimizer_config['min_distance']


def test_path_safety_check(optimizer_config):
    """Test path safety verification"""
    optimizer = PathOptimizer(optimizer_config)

    # Safe path
    path = [
        np.array([0.0, 0.0, 1.0]),
        np.array([2.0, 0.0, 1.0])
    ]

    # Distant obstacle
    obstacles = [
        np.array([1.0, 1.0, 1.0])
    ]

    assert optimizer.check_path_safety(path, obstacles)

    # Add close obstacle
    obstacles.append(np.array([0.1, 0.0, 1.0]))  # Too close to path
    assert not optimizer.check_path_safety(path, obstacles)


def test_empty_path(optimizer_config):
    """Test handling of empty or single-point paths"""
    optimizer = PathOptimizer(optimizer_config)

    # Empty path
    empty_path, velocities = optimizer.optimize_path([])
    assert len(empty_path) == 0
    assert len(velocities) == 0

    # Single point
    single_point = [np.array([0.0, 0.0, 1.0])]
    path, velocities = optimizer.optimize_path(single_point)
    assert len(path) == 1
    assert len(velocities) == 1
    assert velocities[0] == 0.0