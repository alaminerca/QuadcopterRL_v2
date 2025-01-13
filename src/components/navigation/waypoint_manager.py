# src/components/navigation/waypoint_manager.py
"""Waypoint management system for drone navigation"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .collision_avoider import CollisionAvoider
from .path_optimizer import PathOptimizer
from .trajectory_generator import TrajectoryGenerator
import logging

logger = logging.getLogger(__name__)

@dataclass
class Waypoint:
    """
    Single waypoint data structure

    Args:
        position: 3D position [x, y, z]
        radius: Acceptance radius for reaching waypoint
        heading: Desired heading at waypoint (radians)
        speed: Desired speed at waypoint (m/s)
    """
    position: np.ndarray
    radius: float
    heading: float = 0.0
    speed: float = 0.5

    def __post_init__(self):
        """Convert position to numpy array if needed"""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)


class WaypointManager:
    """Manages waypoints and path following for drone navigation"""

    def __init__(self, config: dict):
        """
        Initialize waypoint manager

        Args:
            config: Configuration dictionary containing waypoint parameters
        """
        self.config = config
        self.waypoints: List[Waypoint] = []
        self.current_index: int = 0
        self.path_completed: bool = False
        self.logger = logging.getLogger(__name__)

        # Default parameters
        self.default_radius = config.get('default_radius', 0.5)
        self.default_speed = config.get('default_speed', 0.5)
        self.min_height = config.get('min_height', 0.5)

        # Path metrics
        self.total_distance = 0.0
        self.distance_to_next = 0.0

        # Waypoint reaching state
        self.stable_count = 0
        self.required_stable_steps = 3

        # Trajectory generation (if enabled)
        self.use_trajectories = config.get('use_trajectories', False)
        if self.use_trajectories:
            trajectory_config = config.get('trajectory', {
                'max_velocity': 2.0,
                'max_acceleration': 1.0,
                'curve_resolution': 20
            })
            self.trajectory_generator = TrajectoryGenerator(trajectory_config)

            # Initialize path optimizer
            optimizer_config = config.get('optimizer', {
                'min_distance': 0.5,
                'max_velocity': trajectory_config['max_velocity'],
                'max_acceleration': trajectory_config['max_acceleration'],
                'smoothing_factor': 0.1,
                'path_resolution': 0.1
            })
            self.optimizer = PathOptimizer(optimizer_config)
        else:
            self.trajectory_generator = None
            self.optimizer = None

        # Initialize collision avoidance
        collision_config = config.get('collision_avoidance', {
            'safe_distance': 1.0,
            'max_detection_distance': 3.0,
            'height_adjust': 0.5
        })
        self.collision_avoider = CollisionAvoider(collision_config)

        # Initialize visualization if enabled
        if config.get('visualization', {}).get('enabled', False):
            from ..visualization.path_visualizer import PathVisualizer
            self.visualizer = PathVisualizer()
        else:
            self.visualizer = None

        # Temporary avoidance state
        self.avoiding_collision = False
        self.avoidance_direction = None

    def add_waypoint(self, position: List[float], radius: Optional[float] = None,
                     heading: Optional[float] = None, speed: Optional[float] = None) -> None:
        """
        Add a new waypoint to the path

        Args:
            position: Waypoint position [x, y, z]
            radius: Acceptance radius (optional)
            heading: Desired heading (optional)
            speed: Desired speed (optional)
        """
        # Ensure minimum height for new waypoint
        position[2] = max(position[2], self.min_height)

        waypoint = Waypoint(
            position=np.array(position),
            radius=radius or self.default_radius,
            heading=heading or 0.0,
            speed=speed or self.default_speed
        )

        self.waypoints.append(waypoint)
        self._update_path_metrics()

        # Generate trajectory between waypoints directly
        if self.use_trajectories and len(self.waypoints) > 1:
            # Get positions of actual waypoints only
            positions = [wp.position for wp in self.waypoints]

            # Generate trajectory directly between waypoints
            self.trajectory_generator.generate_trajectory(
                points=positions,
                velocities=[wp.speed for wp in self.waypoints]
            )

    def reset(self) -> None:
        """Reset path following state and visualization"""
        self.current_index = 0
        self.path_completed = False
        self.stable_count = 0
        self._update_path_metrics()

        # Reset trajectory if enabled
        if self.use_trajectories and self.waypoints:
            positions = [wp.position for wp in self.waypoints]
            self.trajectory_generator.generate_trajectory(positions)

        # Reset visualization
        if self.visualizer:
            positions = [wp.position for wp in self.waypoints]
            self.visualizer.update(
                waypoints=positions,
                trajectory_points=self.get_lookahead_points() if self.use_trajectories else None
            )

    def update(self, drone_position: np.ndarray, drone_id: int = None, obstacles: List[int] = None) -> Tuple[
        bool, float]:
        """
        Update navigation state based on drone position

        Args:
            drone_position: Current drone position [x, y, z]
            drone_id: PyBullet ID of drone (for collision checking)
            obstacles: List of obstacle IDs to check

        Returns:
            Tuple of (waypoint_reached, distance_to_target)
        """
        if self.path_completed or not self.waypoints:
            return False, 0.0

        current_waypoint = self.waypoints[self.current_index]
        distance = np.linalg.norm(drone_position - current_waypoint.position)
        self.distance_to_next = distance

        # Check for collisions if IDs provided
        if drone_id is not None and obstacles:
            collision_risk, avoid_direction = self.collision_avoider.check_collision_risk(
                drone_id,
                drone_position,
                current_waypoint.position,
                obstacles,
                already_avoiding=self.avoiding_collision
            )

            self.avoiding_collision = collision_risk
            self.avoidance_direction = avoid_direction if collision_risk else None

            if collision_risk:
                self.stable_count = 0  # Reset stability when avoiding
                return False, distance

        # Check position requirements
        height_diff = abs(drone_position[2] - current_waypoint.position[2])
        position_ok = (distance <= current_waypoint.radius and height_diff <= 0.1)

        if not position_ok:
            if self.stable_count > 0:
                self.logger.debug("Position lost - resetting stability")
            self.stable_count = 0
            return False, distance

        # Increment stability
        self.stable_count += 1

        # Only reach after required_stable_steps updates
        if self.stable_count > self.required_stable_steps:
            self.logger.info(f"Waypoint {self.current_index} reached after {self.stable_count} stable updates")
            self._advance_waypoint()
            return True, distance

        return False, distance

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """
        Get current target waypoint

        Returns:
            Current waypoint or None if path is completed
        """
        if self.path_completed or not self.waypoints:
            return None
        return self.waypoints[self.current_index]

    def get_path_progress(self) -> float:
        """
        Get progress along path from 0 to 1

        Returns:
            Float indicating progress (0 = start, 1 = complete)
        """
        if not self.waypoints:
            return 0.0
        if self.path_completed:
            return 1.0
        if len(self.waypoints) == 1:
            return 1.0 if self.current_index == 0 else 0.0
        return self.current_index / (len(self.waypoints) - 1)

    def get_direction_to_waypoint(self, drone_position: np.ndarray) -> np.ndarray:
        """
        Get normalized direction vector to current target, considering avoidance

        Args:
            drone_position: Current drone position [x, y, z]

        Returns:
            Normalized direction vector [dx, dy, dz]
        """
        if self.avoiding_collision and self.avoidance_direction is not None:
            return self.avoidance_direction

        if self.use_trajectories:
            target_pos, _ = self.trajectory_generator.get_current_target()
        else:
            current = self.get_current_waypoint()
            if current is None:
                return np.zeros(3)
            target_pos = current.position

        direction = target_pos - drone_position
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return np.zeros(3)

        return direction / norm

    def get_lookahead_points(self) -> List[np.ndarray]:
        """
        Get lookahead points for visualization

        Returns:
            List of future trajectory points
        """
        if not self.use_trajectories or not self.trajectory_generator:
            return []
        return self.trajectory_generator.get_lookahead_points()

    def cleanup(self) -> None:
        """Clean up resources, including visualization"""
        if self.visualizer:
            self.visualizer.clear()

    def _advance_waypoint(self) -> None:
        """
        Advance to next waypoint or complete path.
        Sets path_completed to True when last waypoint is reached.
        Updates path metrics after advancing.
        """
        if self.current_index + 1 >= len(self.waypoints):
            self.path_completed = True
        else:
            self.current_index += 1
            self._update_path_metrics()

    def _update_path_metrics(self) -> None:
        """Update path distance metrics"""
        if not self.waypoints:
            self.total_distance = 0.0
            return

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            dist = np.linalg.norm(
                self.waypoints[i + 1].position - self.waypoints[i].position
            )
            total += dist
        self.total_distance = total