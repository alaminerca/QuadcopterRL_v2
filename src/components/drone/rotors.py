import pybullet as p
import numpy as np
from typing import List, Optional, Dict


class RotorSystem:
    """
    Manages the drone's rotor system including creation, force application, and control
    """

    def __init__(self, config: dict):
        """
        Initialize rotor system with configuration

        Args:
            config: Dictionary containing rotor configuration
                count: int - number of rotors
                mass: float - mass per rotor
                max_thrust: float - maximum thrust per rotor
        """
        self.config = config
        self.rotor_count = config['count']
        self.rotor_mass = config['mass']  # Use configured mass
        self.max_thrust = config['max_thrust']
        self.previous_forces = np.zeros(self.rotor_count)  # Initialize here
        self.rotors = []
        self.constraints = []

        # Rotor configuration (X configuration)
        self.rotor_positions = [
            [0.08, 0.08, 0.51],  # Front Right
            [-0.08, 0.08, 0.51],  # Front Left
            [-0.08, -0.08, 0.51],  # Rear Left
            [0.08, -0.08, 0.51]  # Rear Right
        ]

        # Rotor colors for visualization
        self.rotor_colors = [
            [1, 0, 0, 0.8],  # Red - Front Right
            [0, 1, 0, 0.8],  # Green - Front Left
            [1, 1, 0, 0.8],  # Yellow - Rear Left
            [1, 0.5, 0, 0.8]  # Orange - Rear Right
        ]

    def create(self, drone_id: int) -> List[int]:
        """
        Create rotors and attach them to the drone body

        Args:
            drone_id: PyBullet body ID of the drone

        Returns:
            List of rotor IDs
        """
        for pos, color in zip(self.rotor_positions, self.rotor_colors):
            # Create rotor shape
            rotor = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.02,
                height=0.005
            )

            # Create rotor body
            motor = p.createMultiBody(
                baseMass=self.rotor_mass,
                baseCollisionShapeIndex=rotor,
                basePosition=pos
            )

            # Set rotor color
            p.changeVisualShape(motor, -1, rgbaColor=color)

            # Create constraint between drone and rotor
            constraint = p.createConstraint(
                drone_id, -1, motor, -1,
                p.JOINT_FIXED, [0, 0, 0],
                parentFramePosition=[pos[0], pos[1], 0.01],
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=[0, 0, 0, 1]
            )

            # Configure constraint
            p.changeConstraint(constraint, maxForce=100)

            self.rotors.append(motor)
            self.constraints.append(constraint)

        return self.rotors

    def apply_forces(self, actions: List[float]) -> None:
        """Apply forces to rotors based on action inputs"""
        if len(self.rotors) == 0:
            raise ValueError("Rotors not created yet")

        actions = np.array(actions, dtype=np.float32)

        # Physics calculations
        drone_mass = 0.5
        rotor_mass = 0.05 * 4
        total_mass = drone_mass + rotor_mass
        hover_force = total_mass * 9.81
        force_per_rotor = hover_force / 4
        max_force = force_per_rotor * 1.5

        # Calculate hover bias based on required lift
        hover_bias = force_per_rotor / max_force  # Dynamic hover bias

        # Smooth actions (reduced smoothing)
        smoothing = 0.5  # Reduced from 0.8 for faster response
        self.previous_forces = smoothing * self.previous_forces + (1 - smoothing) * actions

        total_thrust = 0
        # Apply forces
        for i, force in enumerate(self.previous_forces):
            # Combine hover bias with control action
            thrust = hover_bias * max_force + force * (max_force - hover_bias * max_force)
            thrust = np.clip(thrust, 0, max_force)
            total_thrust += thrust

            # Apply thrust
            p.applyExternalForce(
                self.rotors[i],
                -1,
                [0, 0, thrust],
                [0, 0, 0],
                p.WORLD_FRAME
            )

            # Apply torque (reduced magnitude)
            torque_magnitude = 0.0002 * thrust  # Further reduced
            torque_direction = 1 if i in [0, 2] else -1
            p.applyExternalTorque(
                self.rotors[i],
                -1,
                [0, 0, torque_magnitude * torque_direction],
                p.WORLD_FRAME
            )

        # Detailed force logging
        if hasattr(self, 'step_counter'):
            self.step_counter += 1
        else:
            self.step_counter = 0

        if self.step_counter % 100 == 0:
            print(f"\nForce analysis:")
            print(f"Hover force needed: {hover_force:.2f}N")
            print(f"Force per rotor: {force_per_rotor:.2f}N")
            print(f"Max force: {max_force:.2f}N")
            print(f"Total thrust: {total_thrust:.2f}N")
            print(f"Hover bias: {hover_bias:.2f}")
            for i, force in enumerate(self.previous_forces):
                print(f"Rotor {i}: Force={force:.2f}, Thrust={(hover_bias + force * 0.5) * max_force:.2f}N")

    def reset(self) -> None:
        """Reset rotor forces and positions"""
        self.previous_forces = np.zeros(self.rotor_count)

        # Reset rotor positions
        for rotor, pos in zip(self.rotors, self.rotor_positions):
            p.resetBasePositionAndOrientation(rotor, pos, [0, 0, 0, 1])