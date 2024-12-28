"""Obstacle Manager module for handling various obstacles in the environment"""
from typing import List, Dict, Optional, Tuple
import numpy as np
import pybullet as p
from dataclasses import dataclass


@dataclass
class ObstacleConfig:
    """Configuration for an obstacle"""
    position: List[float]
    dimensions: List[float]  # [length, width, height] for box, radius for sphere
    orientation: List[float] = None  # Quaternion [x, y, z, w]
    obstacle_type: str = "box"  # "box", "sphere", "cylinder"
    is_static: bool = True
    mass: float = 0.0  # 0 mass for static obstacles
    color: List[float] = None  # RGBA


class ObstacleManager:
    """Manages obstacles in the environment"""

    def __init__(self):
        """Initialize obstacle manager"""
        self.obstacles: Dict[int, ObstacleConfig] = {}  # body_id -> config
        self.dynamic_obstacles: Dict[int, Dict] = {}  # body_id -> movement params
        self._last_update_time = 0

    def add_wall(self, start: List[float], end: List[float], height: float,
                 thickness: float = 0.1, color: List[float] = None) -> int:
        """
        Add a wall between two points

        Args:
            start: Start point [x, y, z]
            end: End point [x, y, z]
            height: Wall height
            thickness: Wall thickness
            color: RGBA color

        Returns:
            int: Wall body ID
        """
        # Calculate wall dimensions and position
        wall_vector = np.array(end) - np.array(start)
        length = np.linalg.norm(wall_vector)
        center = (np.array(start) + np.array(end)) / 2

        # Calculate orientation
        if np.allclose(wall_vector, 0):
            orientation = [0, 0, 0, 1]  # Default orientation
        else:
            direction = wall_vector / length
            angle = np.arctan2(direction[1], direction[0])
            orientation = p.getQuaternionFromEuler([0, 0, angle])

        # Create wall shape
        wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length / 2, thickness / 2, height / 2]
        )

        # Create wall body
        wall_id = p.createMultiBody(
            baseMass=0,  # Static wall
            baseCollisionShapeIndex=wall_shape,
            basePosition=[center[0], center[1], height / 2],
            baseOrientation=orientation
        )

        # Set color if provided
        if color:
            p.changeVisualShape(wall_id, -1, rgbaColor=color)

        # Store configuration
        self.obstacles[wall_id] = ObstacleConfig(
            position=center.tolist(),
            dimensions=[length, thickness, height],
            orientation=orientation,
            obstacle_type="box",
            color=color
        )

        return wall_id

    def add_box(self, position: List[float], dimensions: List[float],
                is_static: bool = True, mass: float = 1.0,
                color: List[float] = None) -> int:
        """Add a box obstacle"""
        box_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[d / 2 for d in dimensions]
        )

        box_id = p.createMultiBody(
            baseMass=0 if is_static else mass,
            baseCollisionShapeIndex=box_shape,
            basePosition=position
        )

        if color:
            p.changeVisualShape(box_id, -1, rgbaColor=color)

        self.obstacles[box_id] = ObstacleConfig(
            position=position,
            dimensions=dimensions,
            obstacle_type="box",
            is_static=is_static,
            mass=mass,
            color=color
        )

        return box_id

    def add_sphere(self, position: List[float], radius: float,
                   is_static: bool = True, mass: float = 1.0,
                   color: List[float] = None) -> int:
        """Add a spherical obstacle"""
        sphere_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )

        sphere_id = p.createMultiBody(
            baseMass=0 if is_static else mass,
            baseCollisionShapeIndex=sphere_shape,
            basePosition=position
        )

        if color:
            p.changeVisualShape(sphere_id, -1, rgbaColor=color)

        self.obstacles[sphere_id] = ObstacleConfig(
            position=position,
            dimensions=[radius],
            obstacle_type="sphere",
            is_static=is_static,
            mass=mass,
            color=color
        )

        return sphere_id

    def add_moving_obstacle(self, config: ObstacleConfig,
                            movement_type: str = "linear",
                            movement_params: Dict = None) -> int:
        """
        Add a moving obstacle

        Args:
            config: Obstacle configuration
            movement_type: Type of movement ("linear", "circular", "sinusoidal")
            movement_params: Parameters for movement pattern

        Returns:
            int: Obstacle body ID
        """
        if config.obstacle_type == "box":
            body_id = self.add_box(
                config.position,
                config.dimensions,
                is_static=False,
                mass=config.mass,
                color=config.color
            )
        elif config.obstacle_type == "sphere":
            body_id = self.add_sphere(
                config.position,
                config.dimensions[0],
                is_static=False,
                mass=config.mass,
                color=config.color
            )
        else:
            raise ValueError(f"Unsupported obstacle type: {config.obstacle_type}")

        self.dynamic_obstacles[body_id] = {
            'type': movement_type,
            'params': movement_params or {},
            'initial_position': config.position
        }

        return body_id

    def update(self, time_step: float) -> None:
        """
        Update positions of dynamic obstacles

        Args:
            time_step: Current simulation time step
        """
        for body_id, movement in self.dynamic_obstacles.items():
            new_pos = self._calculate_position(
                movement['initial_position'],
                movement['type'],
                movement['params'],
                time_step
            )
            p.resetBasePositionAndOrientation(body_id, new_pos, [0, 0, 0, 1])

        self._last_update_time = time_step

    def _calculate_position(self, initial_pos: List[float], movement_type: str,
                            params: Dict, time: float) -> List[float]:
        """Calculate new position based on movement pattern"""
        pos = np.array(initial_pos)

        if movement_type == "linear":
            velocity = np.array(params.get('velocity', [0, 0, 0]))
            pos += velocity * time

        elif movement_type == "circular":
            center = np.array(params.get('center', initial_pos))
            radius = params.get('radius', 1.0)
            frequency = params.get('frequency', 1.0)
            pos = center + radius * np.array([
                np.cos(frequency * time),
                np.sin(frequency * time),
                0
            ])

        elif movement_type == "sinusoidal":
            amplitude = params.get('amplitude', 1.0)
            frequency = params.get('frequency', 1.0)
            pos[2] = initial_pos[2] + amplitude * np.sin(frequency * time)

        return pos.tolist()

    def remove_obstacle(self, body_id: int) -> None:
        """Remove an obstacle"""
        if body_id in self.obstacles:
            p.removeBody(body_id)
            self.obstacles.pop(body_id)
            self.dynamic_obstacles.pop(body_id, None)

    def get_obstacle_config(self, body_id: int) -> Optional[ObstacleConfig]:
        """Get configuration of an obstacle"""
        return self.obstacles.get(body_id)

    def get_all_obstacles(self) -> List[int]:
        """Get list of all obstacle IDs"""
        return list(self.obstacles.keys())

    def clear_all_obstacles(self) -> None:
        """Remove all obstacles"""
        for body_id in list(self.obstacles.keys()):
            self.remove_obstacle(body_id)