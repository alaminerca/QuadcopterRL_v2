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
        self.max_detection_distance = config.get('max_detection_distance', 3.0)
        self.height_adjust = config.get('height_adjust', 0.5)

    def check_collision_risk(self, drone_id: int, drone_position: np.ndarray,
                             target_position: np.ndarray,
                             obstacles: List[int]) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check for potential collisions and get avoidance direction

        Args:
            drone_id: PyBullet ID of drone
            drone_position: Current drone position
            target_position: Desired target position
            obstacles: List of obstacle IDs to check

        Returns:
            Tuple of (collision_risk, avoidance_direction)
        """
        # Get direction to target
        direction = target_position - drone_position
        distance_to_target = np.linalg.norm(direction)

        if distance_to_target > 0:
            direction = direction / distance_to_target

        # Check each obstacle
        for obstacle_id in obstacles:
            # Get closest points between drone and obstacle
            closest_points = p.getClosestPoints(
                bodyA=drone_id,
                bodyB=obstacle_id,
                distance=self.max_detection_distance
            )

            if not closest_points:
                continue

            # Get closest point data
            for point in closest_points:
                contact_distance = point[8]  # distance between objects

                # Check if within safe distance
                if contact_distance < self.safe_distance:
                    # Get positions from contact points
                    drone_contact = np.array(point[5])  # position on drone
                    obstacle_contact = np.array(point[6])  # position on obstacle

                    # Calculate avoidance direction
                    avoidance = self._get_avoidance_direction(
                        drone_contact,
                        obstacle_contact,
                        target_position
                    )
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