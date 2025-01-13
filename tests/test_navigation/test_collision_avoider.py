# tests/test_navigation/test_collision_avoider.py
"""Tests for collision avoidance system"""
import pytest
import numpy as np
import pybullet as p
from src.components.navigation.collision_avoider import CollisionAvoider


@pytest.fixture(scope="module")
def physics_client():
    """Setup PyBullet physics client"""
    client = p.connect(p.DIRECT)  # Headless mode for testing
    yield client
    p.disconnect()


@pytest.fixture
def avoider_config():
    """Test configuration for collision avoidance"""
    return {
        'safe_distance': 1.0,
        'max_detection_distance': 3.0,
        'height_adjust': 0.5
    }


@pytest.fixture
def test_bodies(physics_client):
    """Create test bodies in PyBullet"""
    # Create drone
    drone_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1])
    drone = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=drone_shape,
        basePosition=[0, 0, 1]
    )

    # Create obstacle closer to drone
    obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    obstacle = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obstacle_shape,
        basePosition=[1.0, 0, 1]  # Moved closer to drone
    )

    return {'drone': drone, 'obstacle': obstacle}


def test_basic_collision_check(avoider_config, test_bodies):
    """Test basic collision risk detection"""
    avoider = CollisionAvoider(avoider_config)

    # Drone near obstacle
    p.resetBasePositionAndOrientation(
        test_bodies['drone'],
        [0.5, 0.0, 1.0],  # Position drone closer to obstacle
        [0, 0, 0, 1]
    )

    drone_pos = np.array([0.5, 0.0, 1.0])
    target_pos = np.array([4.0, 0.0, 1.0])  # Target past obstacle

    risk, direction = avoider.check_collision_risk(
        test_bodies['drone'],
        drone_pos,
        target_pos,
        [test_bodies['obstacle']]
    )

    # Should detect risk when obstacle is in path
    assert risk
    assert direction is not None
    # Direction should be normalized
    assert np.isclose(np.linalg.norm(direction), 1.0, rtol=1e-5)


def test_safe_distance(avoider_config, test_bodies):
    """Test safe distance threshold"""
    avoider = CollisionAvoider(avoider_config)

    # Position drone far from obstacle
    p.resetBasePositionAndOrientation(
        test_bodies['drone'],
        [-2.0, 0.0, 1.0],
        [0, 0, 0, 1]
    )

    drone_pos = np.array([-2.0, 0.0, 1.0])
    target_pos = np.array([4.0, 0.0, 1.0])

    risk, _ = avoider.check_collision_risk(
        test_bodies['drone'],
        drone_pos,
        target_pos,
        [test_bodies['obstacle']]
    )

    # Should be safe when far away
    assert not risk


def test_avoidance_direction(avoider_config, test_bodies):
    """Test avoidance direction calculation"""
    avoider = CollisionAvoider(avoider_config)

    # Position drone close to obstacle
    p.resetBasePositionAndOrientation(
        test_bodies['drone'],
        [0.75, 0.0, 1.0],
        [0, 0, 0, 1]
    )

    drone_pos = np.array([0.75, 0.0, 1.0])
    target_pos = np.array([4.0, 0.0, 1.0])

    risk, direction = avoider.check_collision_risk(
        test_bodies['drone'],
        drone_pos,
        target_pos,
        [test_bodies['obstacle']]
    )

    assert risk  # Should detect collision risk
    assert direction is not None

    # Direction should be normalized
    assert np.isclose(np.linalg.norm(direction), 1.0, rtol=1e-5)

    # Should include vertical component
    assert direction[2] > 0


def test_empty_obstacle_list(avoider_config, test_bodies):
    """Test behavior with no obstacles"""
    avoider = CollisionAvoider(avoider_config)

    drone_pos = np.array([0.0, 0.0, 1.0])
    target_pos = np.array([4.0, 0.0, 1.0])

    risk, direction = avoider.check_collision_risk(
        test_bodies['drone'],
        drone_pos,
        target_pos,
        []  # Empty obstacle list
    )

    assert not risk
    assert direction is None