# src/components/navigation/waypoint_manager.py
"""Waypoint management system for drone navigation"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


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

    def update(self, drone_position: np.ndarray) -> Tuple[bool, float]:
        """
        Update navigation state based on drone position

        Args:
            drone_position: Current drone position [x, y, z]

        Returns:
            Tuple of (waypoint_reached, distance_to_target)
        """
        if self.path_completed or not self.waypoints:
            return False, 0.0

        current_waypoint = self.waypoints[self.current_index]
        distance = np.linalg.norm(drone_position - current_waypoint.position)
        self.distance_to_next = distance

        # Check if we're close enough
        height_diff = abs(drone_position[2] - current_waypoint.position[2])
        position_ok = (distance < current_waypoint.radius and height_diff < 0.1)

        # Reset stability counter if we're outside tolerance
        if not position_ok:
            self.stable_count = 0
            return False, distance

        # Increment stability counter
        self.stable_count += 1

        # Check if we've been stable for enough steps
        if self.stable_count >= self.required_stable_steps:
            self.stable_count = 0
            self._advance_waypoint()
            return True, distance

        return False, distance

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
        waypoint = Waypoint(
            position=np.array(position),
            radius=radius or self.default_radius,
            heading=heading or 0.0,
            speed=speed or self.default_speed
        )

        # Ensure minimum height
        waypoint.position[2] = max(waypoint.position[2], self.min_height)

        self.waypoints.append(waypoint)
        self._update_path_metrics()

    def reset(self) -> None:
        """Reset path following state"""
        self.current_index = 0
        self.path_completed = False
        self.stable_count = 0
        self._update_path_metrics()

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
        Get normalized direction vector to current waypoint

        Args:
            drone_position: Current drone position [x, y, z]

        Returns:
            Normalized direction vector [dx, dy, dz]
        """
        current = self.get_current_waypoint()
        if current is None:
            return np.zeros(3)

        direction = current.position - drone_position
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return np.zeros(3)

        return direction / norm

    def _advance_waypoint(self) -> None:
        """
        Advance to next waypoint or complete path.
        Sets path_completed to True when last waypoint is reached.
        Updates path metrics after advancing.
        """
        if self.current_index + 1 >= len(self.waypoints):
            self.path_completed = True
            return

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