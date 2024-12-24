from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set, Callable
import numpy as np
import pybullet as p


@dataclass
class CollisionData:
    """Data structure for collision information"""
    body_a: int
    body_b: int
    link_a: int
    link_b: int
    position: List[float]
    normal: List[float]
    distance: float
    normal_force: float
    lateral_friction_force_1: float
    lateral_friction_force_2: float


class CollisionManager:
    """Manages collision detection and response in the simulation"""

    def __init__(self, margin: float = 0.02):
        """
        Initialize collision manager

        Args:
            margin: Collision margin (buffer distance)
        """
        self.margin = margin
        self.collision_pairs = set()  # Store pairs to check for collisions
        self.collision_callbacks = {}  # Callbacks for collision events
        self.collision_data = []  # Store current collision data

    def add_collision_pair(self, body1: int, body2: int) -> None:
        """
        Add a pair of bodies to check for collisions

        Args:
            body1: First body ID
            body2: Second body ID
        """
        pair = tuple(sorted([body1, body2]))
        self.collision_pairs.add(pair)

    def remove_collision_pair(self, body1: int, body2: int) -> None:
        """Remove a collision pair"""
        pair = tuple(sorted([body1, body2]))
        self.collision_pairs.discard(pair)
        if pair in self.collision_callbacks:
            del self.collision_callbacks[pair]

    def register_collision_callback(self, body1: int, body2: int,
                                    callback: Callable) -> None:
        """Register a callback for collision events between two bodies"""
        pair = tuple(sorted([body1, body2]))
        if pair not in self.collision_pairs:
            self.collision_pairs.add(pair)
        self.collision_callbacks[pair] = callback

    def check_collisions(self) -> List[CollisionData]:
        """
        Check for collisions between registered pairs

        Returns:
            List of CollisionData for all detected collisions
        """
        self.collision_data = []

        for body1, body2 in self.collision_pairs:
            points = p.getContactPoints(bodyA=body1, bodyB=body2)

            for point in points:
                collision = CollisionData(
                    body_a=point[1],
                    body_b=point[2],
                    link_a=point[3],
                    link_b=point[4],
                    position=list(point[5]),
                    normal=list(point[7]),
                    distance=point[8],
                    normal_force=point[9],
                    lateral_friction_force_1=point[10],
                    lateral_friction_force_2=point[11]
                )

                self.collision_data.append(collision)

                pair_key = tuple(sorted([body1, body2]))
                if pair_key in self.collision_callbacks:
                    self.collision_callbacks[pair_key](collision)

        return self.collision_data

    def check_raycast_collision(self, start_pos: List[float],
                                end_pos: List[float]) -> Optional[Dict]:
        """
        Check for collision using raycast

        Args:
            start_pos: Start position of ray
            end_pos: End position of ray

        Returns:
            Dictionary with collision data if hit, None otherwise
        """
        results = p.rayTest(start_pos, end_pos)

        if results[0][0] != -1:  # If we hit something
            return {
                'body_id': results[0][0],
                'position': list(results[0][3]),
                'normal': list(results[0][4]),
                'distance': np.linalg.norm(
                    np.array(end_pos) - np.array(start_pos)
                ) * results[0][2]
            }
        return None

    def check_box_collision(self, body_id: int, half_extents: List[float],
                            position: List[float],
                            orientation: List[float]) -> List[int]:
        """
        Check for collisions within a box volume

        Args:
            body_id: Body ID to check collisions with
            half_extents: Half-sizes of the box [x, y, z]
            position: Position of box center
            orientation: Orientation quaternion [x, y, z, w]

        Returns:
            List of body IDs that collide with the box
        """
        aabb_min = np.array(position) - np.array(half_extents)
        aabb_max = np.array(position) + np.array(half_extents)

        # Get overlapping objects
        overlapping_objects = p.getOverlappingObjects(aabb_min.tolist(), aabb_max.tolist())

        if overlapping_objects is None:
            return []

        # Include the querying body in the list
        object_ids = [obj[0] for obj in overlapping_objects]
        if body_id not in object_ids:
            object_ids.append(body_id)

        return object_ids

    def get_closest_points(self, body1: int, body2: int,
                           max_distance: float) -> List[Tuple]:
        """
        Get closest points between two bodies

        Args:
            body1: First body ID
            body2: Second body ID
            max_distance: Maximum distance to check

        Returns:
            List of closest points data
        """
        return p.getClosestPoints(body1, body2, max_distance)