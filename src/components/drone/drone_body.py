import pybullet as p
import numpy as np
from typing import List, Tuple, Optional


class DroneBody:
    """
    Represents the physical body of the drone in the simulation
    """

    def __init__(self, config: dict):
        """
        Initialize drone body with configuration

        Args:
            config: Dictionary containing drone configuration
                mass: float - mass of the drone body
                dimensions: List[float] - [length, width, height]
        """
        self.mass = config['mass']
        self.dimensions = config['dimensions']
        self.body_id: Optional[int] = None
        self.initial_position = [0, 0, 0.5]
        self.initial_orientation = [0, 0, 0, 1]  # Quaternion [x, y, z, w]

    def create(self, physics_client: int) -> int:
        """
        Create the drone body in PyBullet

        Args:
            physics_client: PyBullet physics client ID

        Returns:
            int: Body ID of the created drone
        """
        # Create collision shape
        self.collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[dim / 2 for dim in self.dimensions]
        )

        # Create drone body
        self.body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=self.collision_shape,
            basePosition=self.initial_position,
            baseOrientation=self.initial_orientation
        )

        # Set drone color (blue)
        p.changeVisualShape(self.body_id, -1, rgbaColor=[0, 0, 1, 0.8])

        return self.body_id

    def get_state(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Get current state of the drone body

        Returns:
            Tuple containing:
            - position [x, y, z]
            - orientation [x, y, z, w]
            - linear_velocity [vx, vy, vz]
            - angular_velocity [wx, wy, wz]
        """
        if self.body_id is None:
            raise ValueError("Drone body not created yet")

        position, orientation = p.getBasePositionAndOrientation(self.body_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.body_id)

        return position, orientation, linear_vel, angular_vel

    def reset(self, position: Optional[List[float]] = None,
              orientation: Optional[List[float]] = None) -> None:
        """
        Reset the drone body to initial or specified position/orientation

        Args:
            position: Optional position [x, y, z]
            orientation: Optional orientation quaternion [x, y, z, w]
        """
        if self.body_id is None:
            raise ValueError("Drone body not created yet")

        pos = position if position is not None else self.initial_position
        orn = orientation if orientation is not None else self.initial_orientation

        p.resetBasePositionAndOrientation(self.body_id, pos, orn)
        # Reset velocities
        p.resetBaseVelocity(self.body_id, [0, 0, 0], [0, 0, 0])