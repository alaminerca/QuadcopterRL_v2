# src/components/visualization/path_visualizer.py
"""Path visualization utilities for drone navigation"""
from typing import List, Dict, Optional
import numpy as np
import pybullet as p


class PathVisualizer:
    """Handles visualization of paths and trajectories in PyBullet"""

    def __init__(self):
        """Initialize path visualizer"""
        self.debug_items = {
            'waypoints': [],  # Waypoint spheres
            'path': [],  # Path lines
            'trajectory': [],  # Trajectory points
            'current': None  # Current target indicator
        }

        # Visual parameters
        self.colors = {
            'waypoint': [0.2, 0.8, 0.2, 0.8],  # Green
            'path': [0.5, 0.5, 0.5, 0.5],  # Gray
            'trajectory': [0.2, 0.2, 0.8, 0.8],  # Blue
            'current': [0.8, 0.2, 0.2, 0.8]  # Red
        }

        self.sizes = {
            'waypoint': 0.1,  # Waypoint sphere size
            'path': 3.0,  # Line width
            'trajectory': 0.05,  # Trajectory point size
            'current': 0.15  # Current target size
        }

    def update(self, waypoints: List[np.ndarray],
               trajectory_points: Optional[List[np.ndarray]] = None,
               current_target: Optional[np.ndarray] = None) -> None:
        """
        Update visualization with new path data

        Args:
            waypoints: List of waypoint positions
            trajectory_points: Optional list of trajectory points
            current_target: Optional current target position
        """
        self.clear()

        # Draw path between waypoints
        for i in range(len(waypoints) - 1):
            line_id = p.addUserDebugLine(
                listify_point(waypoints[i]),
                listify_point(waypoints[i + 1]),
                self.colors['path'],
                lineWidth=self.sizes['path']
            )
            self.debug_items['path'].append(line_id)

        # Draw waypoint spheres (one at a time to avoid color mismatch)
        for pos in waypoints:
            point_id = p.addUserDebugPoints(
                pointPositions=[listify_point(pos)],
                pointColorsRGB=[self.colors['waypoint'][:3]],  # RGB only
                pointSize=self.sizes['waypoint']
            )
            self.debug_items['waypoints'].append(point_id)

        # Draw trajectory points if available
        if trajectory_points:
            for pos in trajectory_points:
                point_id = p.addUserDebugPoints(
                    pointPositions=[listify_point(pos)],
                    pointColorsRGB=[self.colors['trajectory'][:3]],  # RGB only
                    pointSize=self.sizes['trajectory']
                )
                self.debug_items['trajectory'].append(point_id)

        # Draw current target if available
        if current_target is not None:
            self.debug_items['current'] = p.addUserDebugPoints(
                pointPositions=[listify_point(current_target)],
                pointColorsRGB=[self.colors['current'][:3]],  # RGB only
                pointSize=self.sizes['current']
            )

    def clear(self) -> None:
        """Clear all visualizations"""
        # Clear path lines
        for line_id in self.debug_items['path']:
            p.removeUserDebugItem(line_id)

        # Clear waypoint markers
        for point_id in self.debug_items['waypoints']:
            p.removeUserDebugItem(point_id)

        # Clear trajectory points
        for point_id in self.debug_items['trajectory']:
            p.removeUserDebugItem(point_id)

        # Clear current target
        if self.debug_items['current'] is not None:
            p.removeUserDebugItem(self.debug_items['current'])

        # Reset debug items
        self.debug_items = {
            'waypoints': [],
            'path': [],
            'trajectory': [],
            'current': None
        }


def listify_point(point: np.ndarray) -> List[float]:
    """Convert numpy array to list of floats"""
    return [float(x) for x in point]