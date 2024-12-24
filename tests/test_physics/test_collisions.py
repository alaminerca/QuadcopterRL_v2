import pytest
import numpy as np
import pybullet as p
from src.components.physics.collisions import CollisionManager, CollisionData


@pytest.fixture(scope="module")
def physics_client():
    """Set up PyBullet physics client for tests"""
    physics_id = p.connect(p.DIRECT)  # Headless mode for testing
    p.setGravity(0, 0, -9.81)  # Add gravity for realistic testing
    yield physics_id
    p.disconnect()


@pytest.fixture
def test_bodies(physics_client):
    """Create two test boxes for collision testing"""
    # Create boxes
    box1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    box2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])

    # Create multibodies
    body1 = p.createMultiBody(1.0, box1, basePosition=[0, 0, 0])
    body2 = p.createMultiBody(1.0, box2, basePosition=[0.5, 0, 0])

    return body1, body2


def test_collision_detection(test_bodies):
    """Test basic collision detection"""
    manager = CollisionManager()
    body1, body2 = test_bodies

    # Add collision pair
    manager.add_collision_pair(body1, body2)

    # Initially no collision
    collisions = manager.check_collisions()
    assert len(collisions) == 0

    # Move boxes together to cause collision
    p.resetBasePositionAndOrientation(body2, [0.15, 0, 0], [0, 0, 0, 1])
    p.stepSimulation()

    collisions = manager.check_collisions()
    assert len(collisions) > 0
    assert isinstance(collisions[0], CollisionData)


def test_raycast_collision(test_bodies):
    """Test raycast collision detection"""
    manager = CollisionManager()
    body1, _ = test_bodies

    # Cast ray from above box1
    result = manager.check_raycast_collision(
        start_pos=[0, 0, 1],
        end_pos=[0, 0, -1]
    )

    assert result is not None
    assert result['body_id'] == body1
    assert result['distance'] > 0


def test_box_collision(test_bodies):
    """Test box volume collision detection"""
    manager = CollisionManager()
    body1, _ = test_bodies

    # Check for objects in box volume around body1
    overlapping = manager.check_box_collision(
        body_id=body1,
        half_extents=[0.2, 0.2, 0.2],
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1]
    )

    assert body1 in overlapping


def test_collision_callback(test_bodies):
    """Test collision callback functionality"""
    manager = CollisionManager()
    body1, body2 = test_bodies
    callback_data = {'called': False, 'collision': None}

    def on_collision(collision_data: CollisionData):
        callback_data['called'] = True
        callback_data['collision'] = collision_data

    # Register callback and cause collision
    manager.register_collision_callback(body1, body2, on_collision)
    p.resetBasePositionAndOrientation(body2, [0.15, 0, 0], [0, 0, 0, 1])
    p.stepSimulation()

    manager.check_collisions()
    assert callback_data['called']
    assert callback_data['collision'] is not None


def test_closest_points(test_bodies):
    """Test closest points detection"""
    manager = CollisionManager()
    body1, body2 = test_bodies

    # Reset body2 to known position
    p.resetBasePositionAndOrientation(body2, [0.5, 0, 0], [0, 0, 0, 1])
    p.stepSimulation()

    # Get closest points between bodies
    points = manager.get_closest_points(body1, body2, max_distance=1.0)
    assert len(points) > 0
    assert isinstance(points[0], tuple)

    # Verify distance calculation
    point1, point2 = points[0][5:8], points[0][8:11]  # Extract contact points
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    assert distance > 0