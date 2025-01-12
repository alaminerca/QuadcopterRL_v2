# tests/test_navigation/test_trajectory.py
"""Tests for trajectory generation system"""
import pytest
import numpy as np
from src.components.navigation.trajectory_generator import TrajectoryGenerator, BezierCurve


@pytest.fixture
def trajectory_config():
    """Test configuration for trajectory generator"""
    return {
        'max_velocity': 2.0,
        'max_acceleration': 1.0,
        'curve_resolution': 20
    }


def test_bezier_curve():
    """Test basic Bezier curve properties"""
    # Create simple cubic Bezier curve
    control_points = np.array([
        [0.0, 0.0, 0.0],  # Start
        [1.0, 0.0, 0.0],  # Control point 1
        [1.0, 1.0, 0.0],  # Control point 2
        [2.0, 1.0, 0.0]  # End
    ])

    curve = BezierCurve(control_points)

    # Test start point
    start = curve.evaluate(0.0)
    assert np.allclose(start, control_points[0])

    # Test end point
    end = curve.evaluate(1.0)
    assert np.allclose(end, control_points[-1])

    # Test point at t=0.5
    mid = curve.evaluate(0.5)
    assert 0.0 <= mid[0] <= 2.0  # X should be between start and end
    assert 0.0 <= mid[1] <= 1.0  # Y should be between start and end


def test_trajectory_generation(trajectory_config):
    """Test trajectory generation through waypoints"""
    generator = TrajectoryGenerator(trajectory_config)

    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0])
    ]

    generator.generate_trajectory(waypoints)

    # Should create curves between waypoints
    assert len(generator.curves) == len(waypoints) - 1

    # Test initial target
    position, velocity = generator.get_current_target()
    assert np.allclose(position, waypoints[0])

    # Test velocity limits
    assert np.linalg.norm(velocity) <= trajectory_config['max_velocity']


def test_trajectory_update(trajectory_config):
    """Test trajectory state updates"""
    generator = TrajectoryGenerator(trajectory_config)

    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0])
    ]

    generator.generate_trajectory(waypoints)

    # Test progress updates
    generator.update(dt=0.1, progress=0.0)
    pos1, _ = generator.get_current_target()

    generator.update(dt=0.1, progress=0.5)
    pos2, _ = generator.get_current_target()

    # Position should change
    assert not np.allclose(pos1, pos2)


def test_lookahead_points(trajectory_config):
    """Test lookahead point generation"""
    generator = TrajectoryGenerator(trajectory_config)

    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0])
    ]

    generator.generate_trajectory(waypoints)

    points = generator.get_lookahead_points(num_points=5)
    assert len(points) == 5

    # Points should progress toward goal
    for i in range(len(points) - 1):
        assert points[i + 1][0] > points[i][0]  # X should increase


def test_trajectory_velocity_limits(trajectory_config):
    """Test velocity limiting"""
    generator = TrajectoryGenerator(trajectory_config)
    max_vel = trajectory_config['max_velocity']

    # Create waypoints that would naturally create high velocities
    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([10.0, 0.0, 1.0])  # Far point
    ]

    generator.generate_trajectory(waypoints)

    # Check velocities at various points
    for t in np.linspace(0, 1, 10):
        generator.current_param = t
        _, velocity = generator.get_current_target()
        assert np.linalg.norm(velocity) <= max_vel


if __name__ == "__main__":
    pytest.main(["-v", __file__])