import pybullet as p
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import logging


@dataclass
class Force:
    """Base class for forces in the simulation"""
    magnitude: float
    direction: List[float]
    position: List[float] = None  # None means center of mass
    frame: int = p.WORLD_FRAME

    def get_force_vector(self) -> List[float]:
        """Convert magnitude and direction to force vector"""
        dir_norm = np.linalg.norm(self.direction)
        if dir_norm == 0:
            return [0.0, 0.0, 0.0]
        norm_direction = np.array(self.direction) / dir_norm
        return (norm_direction * self.magnitude).tolist()


@dataclass
class DragForce:
    """Air resistance force"""
    drag_coefficient: float
    reference_area: float
    air_density: float = 1.225  # kg/m³ at sea level

    def calculate(self, velocity: List[float]) -> Force:
        """Calculate drag force based on velocity"""
        vel_array = np.array(velocity)
        speed = np.linalg.norm(vel_array)

        if speed == 0:
            return Force(0, [0, 0, 0])

        # Drag equation: F = -0.5 * ρ * v² * Cd * A
        magnitude = 0.5 * self.air_density * speed ** 2 * self.drag_coefficient * self.reference_area
        # Direction opposite to velocity
        direction = -vel_array / speed

        return Force(magnitude, direction.tolist())


class ForceManager:
    """Manages all forces in the simulation"""

    def __init__(self, debug: bool = False):
        """Initialize force manager"""
        self.forces = {
            'constant': {},    # Forces that persist
            'temporary': {},   # Forces that only apply for one step
            'periodic': {}     # Forces that vary periodically
        }
        self.periodic_functions = {}
        self.applied_forces = {}  # Track currently applied forces
        self.drag_configs = {}  # Store drag configurations
        self.max_force_magnitude = 1000.0
        self.max_torque_magnitude = 100.0  # Maximum allowed torque in N⋅m

        # Debug visualization
        self.debug = debug
        self.force_visualizations: Dict[int, int] = {}  # Store debug line IDs

        # Setup logging
        logging.basicConfig(level=logging.INFO if debug else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def add_gravity(self, body_id: int, mass: float, g: float = -9.81) -> str:
        """Add gravitational force to a body"""
        gravity_force = Force(
            magnitude=mass * abs(g),
            direction=[0, 0, np.sign(g)]
        )
        return self.add_force(body_id, gravity_force, 'constant', 'gravity')

    def add_drag(self, body_id: int, drag_coefficient: float,
                 reference_area: float) -> None:
        """Configure drag force for a body"""
        self.drag_configs[body_id] = DragForce(
            drag_coefficient=drag_coefficient,
            reference_area=reference_area
        )

    def add_wind(self, body_id: int, base_magnitude: float,
                 variability: float = 0.2) -> str:
        """Add wind force to body"""

        def wind_function(t: float) -> Force:
            # Create time-varying wind using sinusoidal functions
            variation = 1.0 + variability * np.sin(t * 2.0)  # Ensure wind varies over time
            magnitude = base_magnitude * variation

            # Direction varies with time
            direction = [
                np.cos(t * 0.5),  # X component varies
                np.sin(t * 0.5),  # Y component varies
                0.1 * np.sin(t)  # Small Z variation
            ]

            return Force(magnitude=magnitude, direction=direction)

        self.periodic_functions[body_id] = wind_function
        return self.add_force(body_id, wind_function(0), 'periodic', 'wind')

    def add_force(self, body_id: int, force: Force,
                  force_type: str = 'temporary', name: str = None) -> str:
        """Add a force to a specific body"""
        if force_type not in self.forces:
            raise ValueError(f"Invalid force type: {force_type}")

        # Verify force magnitude is safe
        if force.magnitude > self.max_force_magnitude:
            self.logger.warning(f"Force magnitude {force.magnitude}N exceeds safety limit")
            force.magnitude = self.max_force_magnitude

        if body_id not in self.forces[force_type]:
            self.forces[force_type][body_id] = []

        force_id = name or f"{force_type}_{len(self.forces[force_type][body_id])}"
        self.forces[force_type][body_id].append(force)

        return force_id

    def apply_forces(self, t: float) -> None:
        """Apply all forces for the current timestep"""
        # Reset applied forces
        self.applied_forces = {}

        # Apply each type of force
        for force_type, forces_dict in self.forces.items():
            for body_id, forces in forces_dict.items():
                if force_type == 'periodic' and body_id in self.periodic_functions:
                    # Update periodic forces
                    force = self.periodic_functions[body_id](t)
                    self._apply_force(body_id, force)
                    self._track_force(body_id, force)
                else:
                    # Apply stored forces
                    for force in forces:
                        self._apply_force(body_id, force)
                        self._track_force(body_id, force)

        # Clear temporary forces
        self.forces['temporary'] = {}

    def _apply_force(self, body_id: int, force: Force) -> None:
        """Apply a single force to a body"""
        force_vector = force.get_force_vector()
        position = force.position if force.position is not None else [0, 0, 0]

        # Need to apply force for multiple steps to see movement in PyBullet
        for _ in range(10):  # Apply force for 10 simulation steps
            p.applyExternalForce(
                objectUniqueId=body_id,
                linkIndex=-1,  # -1 for base
                forceObj=force_vector,
                posObj=position,
                flags=force.frame
            )
            p.stepSimulation()  # Step physics simulation

        # Visualize force if debug is enabled
        if self.debug:
            self._visualize_force(body_id, position, force_vector)

    def _visualize_force(self, body_id: int, start_pos: List[float],
                         force_vector: List[float]) -> None:
        """Create debug visualization for forces"""
        # Scale force vector for visualization
        scale = 0.1  # Scale factor for visualization
        end_pos = np.array(start_pos) + np.array(force_vector) * scale

        # Remove old visualization if it exists
        if body_id in self.force_visualizations:
            p.removeUserDebugItem(self.force_visualizations[body_id])

        # Create new visualization
        color = [1, 0, 0]  # Red for forces
        self.force_visualizations[body_id] = p.addUserDebugLine(
            start_pos,
            end_pos.tolist(),
            color,
            lineWidth=2.0,
            lifeTime=0.1  # Short lifetime for dynamic updating
        )

    def clear_forces(self, body_id: Optional[int] = None,
                     force_type: Optional[str] = None) -> None:
        """Clear forces for a specific body and/or type"""
        if force_type and force_type not in self.forces:
            raise ValueError(f"Invalid force type: {force_type}")

        force_types = [force_type] if force_type else self.forces.keys()

        for ftype in force_types:
            if body_id is None:
                self.forces[ftype] = {}
                if ftype == 'periodic':
                    self.periodic_functions = {}
            elif body_id in self.forces[ftype]:
                self.forces[ftype][body_id] = []
                if ftype == 'periodic':
                    self.periodic_functions.pop(body_id, None)

    def get_total_force(self, body_id: int) -> np.ndarray:
        """Calculate total force currently acting on a body"""
        total_force = np.zeros(3)

        for forces_dict in self.forces.values():
            if body_id in forces_dict:
                for force in forces_dict[body_id]:
                    total_force += np.array(force.get_force_vector())

        return total_force

    def get_force_magnitude(self, body_id: int) -> float:
        """Get magnitude of total force on a body"""
        total_force = self.get_total_force(body_id)
        return np.linalg.norm(total_force)

    def _track_force(self, body_id: int, force: Force) -> None:
        """Track applied force for total force calculations"""
        if body_id not in self.applied_forces:
            self.applied_forces[body_id] = []
        self.applied_forces[body_id].append(np.array(force.get_force_vector()))

    def get_total_force(self, body_id: int) -> np.ndarray:
        """Calculate total force currently acting on a body"""
        if body_id not in self.applied_forces:
            return np.zeros(3)
        return np.sum(self.applied_forces[body_id], axis=0)