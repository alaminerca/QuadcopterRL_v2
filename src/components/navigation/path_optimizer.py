# src/components/navigation/path_optimizer.py
"""Path optimization for drone navigation"""
from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import CubicSpline


class PathOptimizer:
    """
    Optimizes paths for drone navigation considering dynamics and obstacles
    """

    def __init__(self, config: dict):
        """
        Initialize path optimizer

        Args:
            config: Configuration dictionary containing optimization parameters
        """
        self.config = config

        # Optimization parameters
        self.min_distance = config.get('min_distance', 0.5)
        self.max_velocity = config.get('max_velocity', 2.0)
        self.max_acceleration = config.get('max_acceleration', 1.0)
        self.smoothing_factor = config.get('smoothing_factor', 0.1)

        # Path resolution
        self.path_resolution = config.get('path_resolution', 0.1)

    def optimize_path(self, waypoints: List[np.ndarray],
                      obstacles: Optional[List[np.ndarray]] = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate optimized path through waypoints

        Args:
            waypoints: List of waypoint positions
            obstacles: Optional list of obstacle positions

        Returns:
            Tuple of (optimized path points, velocity profile)
        """
        if not waypoints:
            return [], []
        if len(waypoints) == 1:
            return waypoints, [0.0]

        # 1. Path smoothing
        smooth_path = self._smooth_path(waypoints)

        # 2. Velocity optimization
        velocities = self._optimize_velocity(smooth_path)

        # 3. Obstacle avoidance (if obstacles present)
        if obstacles:
            smooth_path = self._avoid_obstacles(smooth_path, obstacles)
            # Recompute velocities after obstacle avoidance
            velocities = self._optimize_velocity(smooth_path)

        return smooth_path, velocities

    def _smooth_path(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """
        Create smooth path through waypoints using cubic splines

        Args:
            waypoints: List of waypoint positions

        Returns:
            List of smoothed path points
        """
        waypoints_array = np.array(waypoints)
        num_points = len(waypoints)

        # Create parameter for interpolation (cumulative distance)
        t = np.zeros(num_points)
        for i in range(1, num_points):
            t[i] = t[i - 1] + np.linalg.norm(waypoints_array[i] - waypoints_array[i - 1])

        # Normalize parameter to [0, 1]
        if t[-1] > 0:
            t = t / t[-1]

        # Create splines for each dimension
        splines = [
            CubicSpline(t, waypoints_array[:, i], bc_type='natural')
            for i in range(3)  # x, y, z dimensions
        ]

        # Generate smooth path points
        num_samples = max(int(t[-1] / self.path_resolution), 100)
        t_smooth = np.linspace(0, 1, num_samples)

        smooth_path = []
        min_z = min(waypoints_array[:, 2])  # Get minimum z-height from waypoints

        for t_val in t_smooth:
            point = np.array([spline(t_val) for spline in splines])
            # Ensure z coordinate is not less than minimum height
            point[2] = max(point[2], min_z)
            smooth_path.append(point)

        return smooth_path

    def _optimize_velocity(self, path_points: List[np.ndarray]) -> List[float]:
        """
        Generate velocity profile for path

        Args:
            path_points: List of path points

        Returns:
            List of velocities for each path point
        """
        velocities = []
        prev_point = None

        for point in path_points:
            if prev_point is None:
                velocities.append(0.0)  # Start from rest
            else:
                # Calculate distance and desired velocity
                distance = np.linalg.norm(point - prev_point)

                # Basic trapezoidal velocity profile
                v_desired = min(
                    self.max_velocity,
                    np.sqrt(2 * self.max_acceleration * distance)
                )

                # Smooth velocity transition
                if velocities:
                    v_prev = velocities[-1]
                    max_change = self.max_acceleration * self.path_resolution
                    v_desired = np.clip(
                        v_desired,
                        v_prev - max_change,
                        v_prev + max_change
                    )

                velocities.append(v_desired)

            prev_point = point

        return velocities

    def _avoid_obstacles(self, path: List[np.ndarray],
                         obstacles: List[np.ndarray]) -> List[np.ndarray]:
        """
        Adjust path to avoid obstacles

        Args:
            path: List of path points
            obstacles: List of obstacle positions

        Returns:
            Modified path avoiding obstacles
        """
        safe_path = []

        for point in path:
            # Check distances to all obstacles
            min_distance = float('inf')
            nearest_obstacle = None

            for obstacle in obstacles:
                distance = np.linalg.norm(point - obstacle)
                if distance < min_distance:
                    min_distance = distance
                    nearest_obstacle = obstacle

            if min_distance < self.min_distance and nearest_obstacle is not None:
                # Move point away from obstacle
                direction = point - nearest_obstacle
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    point = nearest_obstacle + direction * self.min_distance

            safe_path.append(point)

        return safe_path

    def check_path_safety(self, path: List[np.ndarray],
                          obstacles: List[np.ndarray]) -> bool:
        """
        Check if path maintains safe distance from obstacles

        Args:
            path: List of path points
            obstacles: List of obstacle positions

        Returns:
            True if path is safe, False otherwise
        """
        for point in path:
            for obstacle in obstacles:
                if np.linalg.norm(point - obstacle) < self.min_distance:
                    return False
        return True