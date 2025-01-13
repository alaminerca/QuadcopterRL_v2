# src/components/navigation/trajectory_generator.py
"""Trajectory generation system for smooth path planning"""
from typing import List, Optional, Tuple
import numpy as np
from scipy.special import comb


class BezierCurve:
    """Bezier curve for smooth trajectory generation with height constraints"""

    def __init__(self, control_points: np.ndarray, min_height: float = 0.5):
        """
        Initialize Bezier curve

        Args:
            control_points: Array of control points
            min_height: Minimum allowed height
        """
        self.control_points = control_points
        self.degree = len(control_points) - 1
        self.min_height = min_height

    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate curve at parameter t with height constraint

        Args:
            t: Parameter in range [0, 1]

        Returns:
            Point on curve with enforced minimum height
        """
        t = np.clip(t, 0.0, 1.0)
        point = np.zeros(3)

        for i in range(len(self.control_points)):
            coeff = comb(self.degree, i) * (t ** i) * ((1 - t) ** (self.degree - i))
            point += coeff * self.control_points[i]

        # Enforce minimum height
        point[2] = max(point[2], self.min_height)
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
        derivative = np.zeros(3)

        # For cubic Bezier curves, derivative uses three control point differences
        for i in range(self.degree):
            coeff = self.degree * comb(self.degree - 1, i) * (t ** i) * ((1 - t) ** (self.degree - 1 - i))
            derivative += coeff * (self.control_points[i + 1] - self.control_points[i])

        return derivative

class TrajectoryGenerator:
    """Generates smooth trajectories between waypoints"""

    def __init__(self, config: dict):
        """
        Initialize trajectory generator

        Args:
            config: Configuration dictionary containing trajectory parameters
        """
        self.config = config
        self.max_velocity = config.get('max_velocity', 2.0)
        self.max_acceleration = config.get('max_acceleration', 1.0)
        self.curve_resolution = config.get('curve_resolution', 20)

        # Current trajectory state
        self.current_curve: Optional[BezierCurve] = None
        self.current_param: float = 0.0
        self.curves: List[BezierCurve] = []
        self.velocities: List[float] = []

    def generate_trajectory(self, points: List[np.ndarray],
                            velocities: Optional[List[float]] = None) -> None:
        """
        Generate smooth trajectory through points with velocity profile

        Args:
            points: List of path points to follow
            velocities: Optional list of velocities for each point
        """
        if len(points) < 2:
            return

        self.curves = []
        self.velocities = velocities if velocities is not None else []

        # Only use original waypoints for curves
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            # Generate intermediate control points
            direction = end - start
            distance = np.linalg.norm(direction)
            unit_direction = direction / (distance + 1e-6)

            # Adjust control point positions based on velocity if available
            if velocities and i < len(velocities):
                v_start = velocities[i]
                v_end = velocities[min(i + 1, len(velocities) - 1)]
                start_influence = v_start * distance / 3
                end_influence = v_end * distance / 3
            else:
                start_influence = distance / 3
                end_influence = distance / 3

            # Create control points for cubic Bezier curve
            control_points = np.array([
                start,
                start + unit_direction * start_influence,
                end - unit_direction * end_influence,
                end
            ])

            self.curves.append(BezierCurve(control_points))

        self.current_curve = self.curves[0] if self.curves else None
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