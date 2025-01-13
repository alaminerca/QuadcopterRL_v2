# tests/test_navigation/test_waypoint_collision.py
"""Tests for waypoint manager collision avoidance integration"""
import pytest
import numpy as np
import pybullet as p
from src.components.navigation.waypoint_manager import WaypointManager


@pytest.fixture(scope="module")
def physics_client():
    """Setup PyBullet physics client"""
    client = p.connect(p.DIRECT)  # Headless mode for testing
    yield client
    p.disconnect()


@pytest.fixture
def nav_config():
    """Test configuration for navigation"""
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
        'collision_avoidance': {
            'safe_distance': 1.5,  # Increased safe distance
            'max_detection_distance': 3.0,
            'height_adjust': 0.5
        }
    }


@pytest.fixture
def test_setup(physics_client):
    """Create test drone and obstacle"""
    # Create drone
    drone_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.1])
    drone = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=drone_shape,
        basePosition=[0, 0, 1]
    )

    # Create obstacle close to drone's path
    obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    obstacle = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obstacle_shape,
        basePosition=[1.0, 0, 1]  # Moved closer
    )

    return {'drone': drone, 'obstacle': obstacle}


def test_collision_avoidance_activation(nav_config, test_setup):
    """Test collision avoidance gets activated"""
    manager = WaypointManager(nav_config)

    # Add waypoints with obstacle between them
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([4.0, 0.0, 1.0])

    # Move drone very close to obstacle
    drone_pos = np.array([0.7, 0.0, 1.0])  # Closer to obstacle

    # Update position in PyBullet
    p.resetBasePositionAndOrientation(
        test_setup['drone'],
        drone_pos,
        [0, 0, 0, 1]
    )
    p.stepSimulation()  # Important: step simulation to update physics

    # Update with collision checking
    reached, _ = manager.update(
        drone_pos,
        drone_id=test_setup['drone'],
        obstacles=[test_setup['obstacle']]
    )

    # Should detect collision and activate avoidance
    assert manager.avoiding_collision
    assert manager.avoidance_direction is not None
    assert not reached  # Shouldn't reach waypoint while avoiding


def test_avoidance_direction(nav_config, test_setup):
    """Test avoidance direction affects path"""
    manager = WaypointManager(nav_config)
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([4.0, 0.0, 1.0])

    # Position drone very close to obstacle
    drone_pos = np.array([0.7, 0.0, 1.0])

    # Update position in PyBullet
    p.resetBasePositionAndOrientation(
        test_setup['drone'],
        drone_pos,
        [0, 0, 0, 1]
    )
    p.stepSimulation()

    # Get normal direction without avoidance
    normal_direction = manager.get_direction_to_waypoint(drone_pos)

    # Update with collision checking
    manager.update(
        drone_pos,
        drone_id=test_setup['drone'],
        obstacles=[test_setup['obstacle']]
    )

    # Get direction with avoidance
    avoidance_direction = manager.get_direction_to_waypoint(drone_pos)

    # Direction should change when avoiding
    assert not np.allclose(normal_direction, avoidance_direction)
    assert avoidance_direction[2] > 0  # Should include upward component


def test_resume_after_avoidance(nav_config, test_setup):
    """Test normal navigation resumes after avoiding"""
    manager = WaypointManager(nav_config)
    manager.add_waypoint([0.0, 0.0, 1.0])
    manager.add_waypoint([4.0, 0.0, 1.0])

    # First update near obstacle
    drone_pos = np.array([0.7, 0.0, 1.0])
    p.resetBasePositionAndOrientation(
        test_setup['drone'],
        drone_pos,
        [0, 0, 0, 1]
    )
    p.stepSimulation()

    manager.update(
        drone_pos,
        drone_id=test_setup['drone'],
        obstacles=[test_setup['obstacle']]
    )
    assert manager.avoiding_collision

    # Move away from obstacle
    drone_pos = np.array([0.7, 2.0, 1.5])  # Moved to side and up
    p.resetBasePositionAndOrientation(
        test_setup['drone'],
        drone_pos,
        [0, 0, 0, 1]
    )
    p.stepSimulation()

    manager.update(
        drone_pos,
        drone_id=test_setup['drone'],
        obstacles=[test_setup['obstacle']]
    )

    # Should no longer be avoiding
    assert not manager.avoiding_collision
    assert manager.avoidance_direction is None