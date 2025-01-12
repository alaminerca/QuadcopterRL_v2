# src/environments/drone_env.py
"""DroneEnv module that integrates all drone simulation components"""
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from ..components.drone.drone_body import DroneBody
from ..components.drone.rotors import RotorSystem
from ..components.obstacles.obstacle_manager import ObstacleManager, ObstacleConfig
from ..components.physics.forces import ForceManager
from ..components.physics.collisions import CollisionManager
from ..components.navigation.waypoint_manager import WaypointManager
from .base_env import BaseEnv
import logging


class DroneEnv(BaseEnv):
    """
    Drone environment that integrates all components for reinforcement learning
    """

    def __init__(self, config_path: str = "configs/default_env.yaml"):
        """Initialize drone environment"""
        super().__init__(config_path)
        self.logger = logging.getLogger(__name__)

        # Create action and observation spaces
        self._create_spaces()

        # Initialize components
        self._init_components()

        # Initialize state variables
        self.current_step = 0
        self.target_height = self.config['drone']['target_height']

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment"""
        # Apply actions
        self.rotor_system.apply_forces(action)

        # Update physics
        self.force_manager.apply_forces(self.current_step * self.config['simulation']['time_step'])
        self.obstacle_manager.update(self.current_step * self.config['simulation']['time_step'])

        p.stepSimulation()

        # Get state and check collisions
        state = self.get_state()
        has_collision = self._check_collisions()

        # Calculate reward
        reward = self.compute_reward(state, action, has_collision)

        # Check termination
        terminated = self.is_terminated(state) or has_collision
        truncated = self.current_step >= self.config['simulation']['max_steps']

        self.current_step += 1

        info = {}
        if self.waypoint_manager:
            info['waypoint_reached'] = self.waypoint_manager.update(state[0:3])[0]
            info['path_progress'] = self.waypoint_manager.get_path_progress()

        return state, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.current_step = 0

        # Reset components
        self.drone_body.reset()
        self.rotor_system.reset()
        self.force_manager.clear_forces()

        if self.waypoint_manager:
            self.waypoint_manager.reset()

        # Get initial state
        state = self.get_state()

        info = {
            'reset_position': state[0:3].tolist(),
            'reset_orientation': state[3:6].tolist()
        }

        if self.waypoint_manager:
            info['path_progress'] = 0.0
            info['waypoint_reached'] = False

        return state, info

    def get_state(self) -> np.ndarray:
        """Get current state vector including navigation data"""
        # Get base state
        position, orientation, linear_vel, angular_vel = self.drone_body.get_state()
        orientation_euler = p.getEulerFromQuaternion(orientation)

        # Initialize navigation state
        waypoint_dir = np.zeros(3)
        waypoint_dist = 0.0
        progress = 0.0

        # Add navigation state if enabled
        if self.waypoint_manager:
            waypoint_dir = self.waypoint_manager.get_direction_to_waypoint(np.array(position))
            _, waypoint_dist = self.waypoint_manager.update(np.array(position))
            progress = self.waypoint_manager.get_path_progress()

        # Combine into state vector
        state = np.concatenate([
            position,
            orientation_euler,
            linear_vel,
            angular_vel,
            waypoint_dir,
            [waypoint_dist],
            [progress]
        ])

        return state.astype(np.float32)

    def compute_reward(self, state: np.ndarray, action: np.ndarray, has_collision: bool) -> float:
        """
        Compute reward based on state, action, and collision status

        Args:
            state: Current state vector
            action: Current action vector
            has_collision: Whether collision occurred

        Returns:
            float: Computed reward
        """
        position = state[0:3]
        waypoint_dist = state[12]  # Index after position, orientation, velocities
        path_progress = state[13]

        # Base hover reward
        height_diff = abs(position[2] - self.target_height)
        height_reward = 2.0 / (1.0 + height_diff * 5)

        # Navigation rewards
        nav_reward = 0.0
        if self.waypoint_manager:
            # Waypoint reached reward
            reached, _ = self.waypoint_manager.update(position)
            if reached:
                nav_reward += self.config['navigation']['rewards']['waypoint_reached']

            # Progress reward
            nav_reward += path_progress * self.config['navigation']['rewards']['progress']

            # Distance penalty (transformed to decay from -1 to 0)
            dist_penalty = -1.0 / (1.0 + np.exp(-0.5 * waypoint_dist))
            nav_reward += dist_penalty

        # Obstacle penalty
        obstacle_penalty = 0.0
        if has_collision:
            obstacle_penalty = -10.0
        else:
            for obstacle_id in self.obstacle_manager.get_all_obstacles():
                closest_points = self.collision_manager.get_closest_points(
                    self.drone_id, obstacle_id, max_distance=1.0
                )
                if closest_points:
                    distance = closest_points[0][8]
                    obstacle_penalty = -5.0 * (1.0 - min(distance, 1.0))

        # Combine rewards
        reward = height_reward + nav_reward + obstacle_penalty

        return float(np.clip(reward, -8, 8))

    def is_terminated(self, state: np.ndarray) -> bool:
        """
        Check if episode should terminate

        Args:
            state: Current state vector

        Returns:
            bool: Whether episode should terminate
        """
        position = state[0:3]
        orientation = state[3:6]

        # Log termination checks
        self.logger.info(f"Height: {position[2]:.3f}m")
        self.logger.info(f"Roll: {np.degrees(orientation[0]):.1f}°, "
                         f"Pitch: {np.degrees(orientation[1]):.1f}°")

        if position[2] < 0.02:
            self.logger.info("Terminated: Too low")
            return True

        if np.abs(orientation[0]) > np.pi / 2 or np.abs(orientation[1]) > np.pi / 2:
            self.logger.info("Terminated: Extreme tilt")
            return True

        return False

    def _create_spaces(self):
        """Create action and observation spaces"""
        # Action space: [0,1] for each rotor
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

        # Observation space limits
        obs_ranges = {
            'position': 10.0,  # x, y, z position
            'orientation': np.pi,  # roll, pitch, yaw
            'linear_vel': 10.0,  # x, y, z velocity
            'angular_vel': 5.0,  # roll, pitch, yaw rates
            'waypoint_dir': 1.0,  # normalized direction to waypoint
            'waypoint_dist': 20.0,  # distance to waypoint
            'progress': 1.0  # path progress
        }

        obs_low = np.array([
                               -obs_ranges['position']] * 3 +  # position
                           [-obs_ranges['orientation']] * 3 +  # orientation
                           [-obs_ranges['linear_vel']] * 3 +  # linear velocity
                           [-obs_ranges['angular_vel']] * 3 +  # angular velocity
                           [-obs_ranges['waypoint_dir']] * 3 +  # waypoint direction
                           [0.0] +  # waypoint distance
                           [0.0]  # path progress
                           )

        obs_high = np.array([
                                obs_ranges['position']] * 3 +
                            [obs_ranges['orientation']] * 3 +
                            [obs_ranges['linear_vel']] * 3 +
                            [obs_ranges['angular_vel']] * 3 +
                            [obs_ranges['waypoint_dir']] * 3 +
                            [obs_ranges['waypoint_dist']] +
                            [obs_ranges['progress']]
                            )

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

    def _init_components(self):
        """Initialize all environment components"""
        # Create drone components
        self.drone_body = DroneBody(self.config['drone'])
        self.rotor_system = RotorSystem(self.config['drone']['rotors'])

        # Create physics components
        self.force_manager = ForceManager(debug=self.config['physics'].get('debug', False))
        self.collision_manager = CollisionManager()
        self.obstacle_manager = ObstacleManager()

        # Create navigation component if enabled
        if self.config.get('navigation', {}).get('enabled', False):
            self.waypoint_manager = WaypointManager(self.config['navigation']['waypoints'])
        else:
            self.waypoint_manager = None

        # Create drone in PyBullet
        self.drone_id = self.drone_body.create(self.physics_client)
        self.rotor_ids = self.rotor_system.create(self.drone_id)

        # Setup initial obstacles
        self._setup_obstacles()

        # Setup initial waypoints
        self._setup_waypoints()

        # Add physics effects
        self._setup_physics()

    def _setup_obstacles(self):
        """Setup initial obstacles from configuration"""
        if not self.config.get('obstacles', {}).get('enabled', False):
            return

        # Setup static obstacles
        if 'static' in self.config['obstacles']:
            # Add walls
            for wall in self.config['obstacles']['static'].get('walls', []):
                self.obstacle_manager.add_wall(
                    start=wall['start'],
                    end=wall['end'],
                    height=wall['height'],
                    thickness=wall.get('thickness', 0.1),
                    color=wall.get('color')
                )

            # Add boxes
            for box in self.config['obstacles']['static'].get('boxes', []):
                self.obstacle_manager.add_box(
                    position=box['position'],
                    dimensions=box['dimensions'],
                    color=box.get('color')
                )

        # Setup dynamic obstacles
        if self.config['obstacles'].get('dynamic', {}).get('enabled', False):
            for obs in self.config['obstacles']['dynamic'].get('moving_obstacles', []):
                config = ObstacleConfig(
                    position=obs.get('position', [0, 0, 1]),
                    dimensions=[obs['radius']] if obs['type'] == 'sphere' else obs['dimensions'],
                    obstacle_type=obs['type'],
                    mass=obs.get('mass', 1.0),
                    is_static=False,
                    color=obs.get('color')
                )

                self.obstacle_manager.add_moving_obstacle(
                    config=config,
                    movement_type=obs['movement']['type'],
                    movement_params=obs['movement']['params']
                )
    def _setup_waypoints(self):
        """Setup initial waypoints if navigation is enabled"""
        if not self.waypoint_manager:
            return

        # Add default waypoints from config if specified
        waypoints = self.config.get('navigation', {}).get('initial_waypoints', [])
        for wp in waypoints:
            self.waypoint_manager.add_waypoint(
                position=wp['position'],
                radius=wp.get('radius'),
                heading=wp.get('heading'),
                speed=wp.get('speed')
            )

    def _setup_physics(self):
        """Setup physics effects from configuration"""
        if self.config['physics'].get('wind', {}).get('enabled', False):
            self.force_manager.add_wind(
                self.drone_id,
                base_magnitude=self.config['physics']['wind']['base_magnitude'],
                variability=self.config['physics']['wind'].get('variability', 0.0)
            )

    def _check_collisions(self) -> bool:
        """
        Check for collisions with obstacles

        Returns:
            bool: Whether collision was detected
        """
        for obstacle_id in self.obstacle_manager.get_all_obstacles():
            if self.collision_manager.are_objects_colliding(self.drone_id, obstacle_id):
                return True
        return False









