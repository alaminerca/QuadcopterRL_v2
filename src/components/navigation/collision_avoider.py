# src/components/navigation/collision_avoider.py
"""Simple collision avoidance for drone navigation"""
from typing import List, Tuple, Optional
import numpy as np
import pybullet as p


class CollisionAvoider:
    """Basic collision avoidance system"""

    def __init__(self, config: dict):
        """
        Initialize collision avoider

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Safety parameters
        self.safe_distance = config.get('safe_distance', 1.0)
        self.resume_distance = self.safe_distance * 0.8  # Distance to resume normal navigation
        self.max_detection_distance = config.get('max_detection_distance', 3.0)
        self.height_adjust = config.get('height_adjust', 0.5)

    def check_collision_risk(self, drone_id: int, drone_position: np.ndarray,
                             target_position: np.ndarray,
                             obstacles: List[int],
                             already_avoiding: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check for potential collisions and get avoidance direction

        Args:
            drone_id: PyBullet ID of drone
            drone_position: Current drone position
            target_position: Desired target position
            obstacles: List of obstacle IDs to check
            already_avoiding: Whether currently in avoidance mode

        Returns:
            Tuple of (collision_risk, avoidance_direction)
        """
        if not obstacles:
            return False, None

        min_distance = float('inf')
        closest_obstacle_pos = None

        # Check each obstacle
        for obstacle_id in obstacles:
            closest_points = p.getClosestPoints(
                bodyA=drone_id,
                bodyB=obstacle_id,
                distance=self.max_detection_distance
            )

            if closest_points:
                point = closest_points[0]  # Get first (closest) point
                contact_distance = point[8]  # distance between objects

                print(f"Debug - Distance to obstacle: {contact_distance}, Safe distance: {self.safe_distance}")

                if contact_distance < min_distance:
                    min_distance = contact_distance
                    closest_obstacle_pos = np.array(point[6])

        # Use different thresholds for activation and deactivation
        threshold = self.safe_distance if not already_avoiding else self.resume_distance

        if min_distance < threshold:
            avoidance = self._get_avoidance_direction(
                drone_position,
                closest_obstacle_pos,
                target_position
            )
            print(f"Debug - Avoidance activated, direction: {avoidance}")
            return True, avoidance

        return False, None

    def _get_avoidance_direction(self, drone_pos: np.ndarray,
                                 obstacle_pos: np.ndarray,
                                 target_pos: np.ndarray) -> np.ndarray:
        """
        Get simple avoidance direction

        Args:
            drone_pos: Current drone position
            obstacle_pos: Obstacle position
            target_pos: Target position

        Returns:
            Avoidance direction vector
        """
        # Get vector from obstacle to drone
        away_vector = drone_pos - obstacle_pos

        # Project onto horizontal plane for horizontal avoidance
        horizontal_avoid = np.array([
            away_vector[0],
            away_vector[1],
            0.0
        ])

        # Normalize horizontal component
        horizontal_norm = np.linalg.norm(horizontal_avoid)
        if horizontal_norm > 0:
            horizontal_avoid = horizontal_avoid / horizontal_norm
        else:
            horizontal_avoid = np.array([1.0, 0.0, 0.0])  # Default direction if directly above/below

        # Add upward component for vertical avoidance
        vertical_component = np.array([0.0, 0.0, self.height_adjust])

        # Combine horizontal and vertical (ensuring non-zero)
        avoidance = horizontal_avoid + vertical_component

        # Normalize final vector
        avoidance_norm = np.linalg.norm(avoidance)
        if avoidance_norm > 0:
            avoidance = avoidance / avoidance_norm
        else:
            avoidance = np.array([0.0, 0.0, 1.0])  # Default to upward if no clear direction

        return avoidance