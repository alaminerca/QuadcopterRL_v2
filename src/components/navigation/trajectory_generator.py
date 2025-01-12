# src/components/navigation/trajectory_generator.py
"""Trajectory generation system for smooth path planning"""
from typing import List, Optional, Tuple
import numpy as np
from scipy.special import comb


class BezierCurve:
    """Bezier curve implementation for smooth trajectories"""

    def __init__(self, control_points: np.ndarray):
        """
        Initialize Bezier curve with control points

        Args:
            control_points: Array of shape (n, 3) containing control points
        """
        self.control_points = control_points
        self.degree = len(control_points) - 1

    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate curve at parameter t

        Args:
            t: Parameter in range [0, 1]

        Returns:
            Point on curve at parameter t
        """
        t = np.clip(t, 0.0, 1.0)
        point = np.zeros(3)

        for i in range(len(self.control_points)):
            coeff = comb(self.degree, i) * (t ** i) * ((1 - t) ** (self.degree - i))
            point += coeff * self.control_points[i]

        return point

    def evaluate_derivative(self, t: float) -> np.ndarray:
        """
        Evaluate curve derivative (velocity) at parameter t

        Args:
            t: Parameter in range [0, 1]

        Returns:
            Velocity vector at parameter t
        """
        t = np.clip(t, 0.0, 1.0)
        velocity = np.zeros(3)

        for i in range(self.degree):
            coeff = self.degree * comb(self.degree - 1, i) * (t ** i) * ((1 - t) ** (self.degree - 1 - i))
            velocity += coeff * (self.control_points[i + 1] - self.control_points[i])

        return velocity


class TrajectoryGenerator:
    """Generates smooth trajectories between waypoints"""

    def __init__(self, config: dict):
        """
        Initialize trajectory generator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.max_velocity = config.get('max_velocity', 2.0)
        self.max_acceleration = config.get('max_acceleration', 1.0)
        self.curve_resolution = config.get('curve_resolution', 20)

        # Current trajectory state
        self.current_curve: Optional[BezierCurve] = None
        self.current_param: float = 0.0
        self.curves: List[BezierCurve] = []

    def generate_trajectory(self, waypoints: List[np.ndarray]) -> None:
        """
        Generate smooth trajectory through waypoints

        Args:
            waypoints: List of waypoint positions
        """
        if len(waypoints) < 2:
            return

        self.curves = []

        # Generate control points for each segment
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]

            # Generate intermediate control points
            direction = end - start
            distance = np.linalg.norm(direction)
            unit_direction = direction / (distance + 1e-6)

            # Create control points for cubic Bezier curve
            control_points = np.array([
                start,
                start + unit_direction * (distance / 3),
                end - unit_direction * (distance / 3),
                end
            ])

            self.curves.append(BezierCurve(control_points))

        self.current_curve = self.curves[0]
        self.current_param = 0.0

    def get_current_target(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current target position and velocity

        Returns:
            Tuple of (position, velocity) for current target
        """
        if self.current_curve is None:
            return np.zeros(3), np.zeros(3)

        position = self.current_curve.evaluate(self.current_param)
        velocity = self.current_curve.evaluate_derivative(self.current_param)

        # Scale velocity to respect max_velocity
        velocity_mag = np.linalg.norm(velocity)
        if velocity_mag > self.max_velocity:
            velocity = velocity * self.max_velocity / velocity_mag

        return position, velocity

    def update(self, dt: float, progress: float) -> None:
        """
        Update trajectory state

        Args:
            dt: Time step
            progress: Overall path progress (0-1)
        """
        if not self.curves:
            return

        # Update curve index based on progress
        curve_index = int(progress * len(self.curves))
        curve_index = min(curve_index, len(self.curves) - 1)

        # Update current curve if needed
        if self.current_curve is not self.curves[curve_index]:
            self.current_curve = self.curves[curve_index]
            self.current_param = 0.0

        # Update parameter along curve
        param_velocity = 1.0 / self.curve_resolution
        self.current_param = min(self.current_param + param_velocity * dt, 1.0)

    def get_lookahead_points(self, num_points: int = 5) -> List[np.ndarray]:
        """
        Get future points along trajectory for visualization

        Args:
            num_points: Number of lookahead points

        Returns:
            List of positions along future trajectory
        """
        if self.current_curve is None:
            return []

        points = []
        param_step = (1.0 - self.current_param) / num_points

        for i in range(num_points):
            t = self.current_param + i * param_step
            t = min(t, 1.0)
            points.append(self.current_curve.evaluate(t))

        return points