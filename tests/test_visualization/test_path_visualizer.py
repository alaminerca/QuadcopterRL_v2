# tests/test_visualization/test_path_visualizer.py
"""Tests for path visualization system"""
import pytest
import numpy as np
import pybullet as p
from src.components.visualization.path_visualizer import PathVisualizer


@pytest.fixture(scope="module")
def physics_client():
    """Setup PyBullet physics client"""
    physics_id = p.connect(p.DIRECT)  # Headless mode for testing
    yield physics_id
    p.disconnect()


def test_visualizer_initialization():
    """Test visualizer initialization"""
    visualizer = PathVisualizer()
    assert visualizer.debug_items['waypoints'] == []
    assert visualizer.debug_items['path'] == []
    assert visualizer.debug_items['trajectory'] == []
    assert visualizer.debug_items['current'] is None


def test_path_visualization(physics_client):
    """Test basic path visualization"""
    visualizer = PathVisualizer()

    # Create simple path
    waypoints = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0])
    ]

    # Update visualization
    visualizer.update(waypoints)

    # Should create visualization items
    assert len(visualizer.debug_items['waypoints']) == len(waypoints)
    assert len(visualizer.debug_items['path']) == len(waypoints) - 1


def test_trajectory_visualization(physics_client):
    """Test trajectory point visualization"""
    visualizer = PathVisualizer()

    waypoints = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
    trajectory = [
        np.array([0.2, 0.0, 1.0]),
        np.array([0.4, 0.0, 1.0]),
        np.array([0.6, 0.0, 1.0]),
        np.array([0.8, 0.0, 1.0])
    ]

    visualizer.update(waypoints, trajectory)
    assert len(visualizer.debug_items['trajectory']) == len(trajectory)


def test_clear_visualization(physics_client):
    """Test clearing visualizations"""
    visualizer = PathVisualizer()

    # Create and then clear visualization
    waypoints = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
    visualizer.update(waypoints)
    visualizer.clear()

    # All debug items should be cleared
    assert len(visualizer.debug_items['waypoints']) == 0
    assert len(visualizer.debug_items['path']) == 0
    assert len(visualizer.debug_items['trajectory']) == 0
    assert visualizer.debug_items['current'] is None


def test_current_target_visualization(physics_client):
    """Test current target visualization"""
    visualizer = PathVisualizer()

    waypoints = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])]
    current = np.array([0.5, 0.0, 1.0])

    visualizer.update(waypoints, current_target=current)
    assert visualizer.debug_items['current'] is not None