"""Tests for Obstacle Management System"""
import pytest
import numpy as np
import pybullet as p
from src.components.obstacles.obstacle_manager import ObstacleManager, ObstacleConfig


@pytest.fixture(scope="module")
def physics_client():
    """Setup PyBullet physics client"""
    physics_id = p.connect(p.DIRECT)  # Headless mode for testing
    p.setGravity(0, 0, -9.81)
    yield physics_id
    p.disconnect()


@pytest.fixture
def obstacle_manager(physics_client):
    """Create obstacle manager for each test"""
    manager = ObstacleManager()
    yield manager
    manager.clear_all_obstacles()


def test_wall_creation(obstacle_manager):
    """Test wall creation and properties"""
    # Create a simple wall
    wall_id = obstacle_manager.add_wall(
        start=[0, 0, 0],
        end=[1, 0, 0],
        height=2.0,
        thickness=0.1,
        color=[1, 0, 0, 1]  # Red wall
    )

    # Verify wall exists and print info
    print(f"\nWall created with ID: {wall_id}")
    assert wall_id in obstacle_manager.get_all_obstacles()

    # Check configuration
    config = obstacle_manager.get_obstacle_config(wall_id)
    print(f"Wall dimensions: {config.dimensions}")
    assert config.obstacle_type == "box"
    assert config.is_static is True
    assert config.mass == 0.0  # Static wall

    # Check dimensions
    assert np.isclose(config.dimensions[0], 1.0)  # Length
    assert np.isclose(config.dimensions[1], 0.1)  # Thickness
    assert np.isclose(config.dimensions[2], 2.0)  # Height


def test_box_creation(obstacle_manager):
    """Test box obstacle creation"""
    box_id = obstacle_manager.add_box(
        position=[1, 1, 1],
        dimensions=[0.5, 0.5, 0.5],
        is_static=False,
        mass=1.0
    )

    print(f"\nBox created with ID: {box_id}")
    config = obstacle_manager.get_obstacle_config(box_id)
    assert config.mass == 1.0
    assert not config.is_static
    assert config.obstacle_type == "box"

    # Check position
    pos, _ = p.getBasePositionAndOrientation(box_id)
    print(f"Box position: {pos}")
    assert np.allclose(pos, [1, 1, 1])


def test_sphere_creation(obstacle_manager):
    """Test spherical obstacle creation"""
    sphere_id = obstacle_manager.add_sphere(
        position=[0, 0, 1],
        radius=0.5,
        is_static=True
    )

    print(f"\nSphere created with ID: {sphere_id}")
    config = obstacle_manager.get_obstacle_config(sphere_id)
    print(f"Sphere radius: {config.dimensions[0]}")
    assert config.dimensions[0] == 0.5  # radius
    assert config.is_static
    assert config.obstacle_type == "sphere"


def test_moving_obstacle(obstacle_manager):
    """Test dynamic obstacle movement"""
    config = ObstacleConfig(
        position=[0, 0, 1],
        dimensions=[0.3],  # radius
        obstacle_type="sphere",
        mass=1.0,
        is_static=False
    )

    obstacle_id = obstacle_manager.add_moving_obstacle(
        config=config,
        movement_type="circular",
        movement_params={
            'center': [0, 0, 1],
            'radius': 1.0,
            'frequency': 1.0
        }
    )

    print(f"\nMoving obstacle created with ID: {obstacle_id}")
    initial_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
    print(f"Initial position: {initial_pos}")

    obstacle_manager.update(time_step=0.5)
    new_pos, _ = p.getBasePositionAndOrientation(obstacle_id)
    print(f"New position after update: {new_pos}")

    assert not np.allclose(initial_pos, new_pos)


def test_obstacle_removal(obstacle_manager):
    """Test obstacle removal"""
    wall_id = obstacle_manager.add_wall([0, 0, 0], [1, 0, 0], 2.0)
    box_id = obstacle_manager.add_box([1, 1, 1], [0.5, 0.5, 0.5])

    print(f"\nCreated wall ({wall_id}) and box ({box_id})")
    obstacle_manager.remove_obstacle(wall_id)
    print(f"Removed wall {wall_id}")

    remaining = obstacle_manager.get_all_obstacles()
    print(f"Remaining obstacles: {remaining}")
    assert wall_id not in remaining
    assert box_id in remaining


def test_clear_all_obstacles(obstacle_manager):
    """Test clearing all obstacles"""
    # Add multiple obstacles
    ids = []
    ids.append(obstacle_manager.add_wall([0, 0, 0], [1, 0, 0], 2.0))
    ids.append(obstacle_manager.add_box([1, 1, 1], [0.5, 0.5, 0.5]))
    ids.append(obstacle_manager.add_sphere([0, 0, 1], 0.5))

    print(f"\nCreated obstacles with IDs: {ids}")
    obstacle_manager.clear_all_obstacles()
    print("Cleared all obstacles")

    remaining = obstacle_manager.get_all_obstacles()
    print(f"Remaining obstacles: {remaining}")
    assert len(remaining) == 0