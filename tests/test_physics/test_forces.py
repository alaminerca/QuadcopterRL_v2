import pytest
import numpy as np
import pybullet as p
from src.components.physics.forces import Force, ForceManager, DragForce


@pytest.fixture(scope="module")
def physics_client():
    """Set up PyBullet physics client for tests"""
    physics_id = p.connect(p.DIRECT)  # Headless mode for testing
    yield physics_id
    p.disconnect()


@pytest.fixture
def test_body(physics_client):
    """Create a test body for force application"""
    body_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
    mass = 1.0
    body = p.createMultiBody(mass, body_id, basePosition=[0, 0, 0])
    return body


def test_force_creation():
    """Test Force class initialization and vector calculation"""
    force = Force(magnitude=10.0, direction=[1, 0, 0])
    force_vector = force.get_force_vector()
    assert force_vector == [10.0, 0, 0]

    # Test normalization
    force = Force(magnitude=10.0, direction=[1, 1, 0])
    force_vector = force.get_force_vector()
    assert np.allclose(np.linalg.norm(force_vector), 10.0)


def test_force_manager_basic(test_body):
    """Test basic ForceManager functionality"""
    manager = ForceManager()

    # Add and apply a simple force
    force = Force(magnitude=10.0, direction=[1, 0, 0])
    manager.add_force(test_body, force)
    manager.apply_forces(0.0)

    # Check body velocity after force application
    vel = p.getBaseVelocity(test_body)[0]
    assert vel[0] > 0  # Should move in positive x direction


def test_drag_force():
    """Test drag force calculations"""
    drag = DragForce(
        drag_coefficient=0.5,
        reference_area=1.0,
        air_density=1.225
    )

    # Test zero velocity
    zero_force = drag.calculate([0, 0, 0])
    assert np.allclose(zero_force.get_force_vector(), [0, 0, 0])

    # Test constant velocity
    velocity = [10, 0, 0]
    drag_force = drag.calculate(velocity)
    force_vector = drag_force.get_force_vector()
    assert force_vector[0] < 0  # Force should oppose motion
    assert force_vector[1] == 0
    assert force_vector[2] == 0


def test_wind_force(test_body):
    """Test wind force application"""
    manager = ForceManager()

    # Add constant wind
    manager.add_wind(test_body, base_magnitude=1.0, variability=0.0)

    # Apply forces for multiple steps
    wind_forces = []
    for t in range(10):
        manager.apply_forces(t * 0.1)
        total_force = manager.get_total_force(test_body)
        wind_forces.append(total_force)

    # Check wind variation over time
    wind_variations = np.diff([np.linalg.norm(f) for f in wind_forces])
    assert np.any(wind_variations != 0)  # Wind should vary


def test_force_limits(test_body):
    """Test force magnitude limits"""
    manager = ForceManager()

    # Try to add force above limit
    huge_force = Force(magnitude=2000.0, direction=[1, 0, 0])
    manager.add_force(test_body, huge_force)

    # Verify force was limited
    total_force = manager.get_total_force(test_body)
    assert np.linalg.norm(total_force) <= manager.max_force_magnitude


def test_multiple_forces(test_body):
    """Test multiple force interactions"""
    manager = ForceManager()

    # Add opposing forces
    manager.add_force(test_body, Force(magnitude=10.0, direction=[1, 0, 0]))
    manager.add_force(test_body, Force(magnitude=5.0, direction=[-1, 0, 0]))

    # Apply forces
    manager.apply_forces(0.0)

    # Get total force
    total_force = manager.get_total_force(test_body)
    assert np.allclose(total_force, [5.0, 0, 0])  # Net force should be 5N in x direction