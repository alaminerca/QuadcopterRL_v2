from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from typing import List
from ..components.drone.drone_body import DroneBody
from ..components.drone.rotors import RotorSystem
from ..components.obstacles.obstacle_manager import ObstacleManager, ObstacleConfig
from ..components.physics.forces import ForceManager
from ..components.physics.collisions import CollisionManager
from .base_env import BaseEnv
import logging


class DroneEnv(BaseEnv):
    """
    Drone environment that integrates all components for reinforcement learning
    """

    def __init__(self, config_path: str = "configs/default_env.yaml"):
        """
        Initialize drone environment

        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        self.logger = logging.getLogger(__name__)

        # Create action and observation spaces
        self._create_spaces()

        # Initialize components
        self._init_components()

        # Initialize state variables
        self.current_step = 0
        self.target_height = self.config['drone']['target_height']

    def _setup_obstacles(self):
            """Setup initial obstacles from configuration"""
            if not self.config['obstacles'].get('enabled', False):
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

    def _init_components(self):
        """Initialize all environment components"""
        # Create drone components
        self.drone_body = DroneBody(self.config['drone'])
        self.rotor_system = RotorSystem(self.config['drone']['rotors'])

        # Create physics components
        self.force_manager = ForceManager(debug=self.config['physics'].get('debug', False))
        self.collision_manager = CollisionManager()
        self.obstacle_manager = ObstacleManager()

        # Create drone in PyBullet
        self.drone_id = self.drone_body.create(self.physics_client)
        self.rotor_ids = self.rotor_system.create(self.drone_id)

        # Setup initial obstacles if configured
        self._setup_obstacles()

        # Add physics effects
        if self.config['physics'].get('enable_wind', False):
            self.force_manager.add_wind(
                self.drone_id,
                base_magnitude=self.config['physics']['wind_magnitude'],
                variability=self.config['physics']['wind_variability']
            )

    def _create_spaces(self):
        """Create action and observation spaces"""
        # Action space: [0,1] for each rotor
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(4,),
            dtype=np.float32
        )

        # State space limits
        high = np.array([
            10.0,  # x pos
            10.0,  # y pos
            10.0,  # z pos
            np.pi,  # roll
            np.pi,  # pitch
            np.pi,  # yaw
            10.0,  # x vel
            10.0,  # y vel
            10.0,  # z vel
            5.0,  # roll rate
            5.0,  # pitch rate
            5.0  # yaw rate
        ])

        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def get_state(self) -> np.ndarray:
        """Get current state of the environment"""
        position, orientation, linear_vel, angular_vel = self.drone_body.get_state()

        # Convert quaternion to euler angles
        orientation_euler = p.getEulerFromQuaternion(orientation)

        # Combine into state vector
        state = np.array(
            position +
            orientation_euler +
            linear_vel +
            angular_vel
        )

        return state.astype(np.float32)

    def compute_reward(self, state: np.ndarray, action: np.ndarray, has_collision: bool) -> float:
        action = np.array(action, dtype=np.float32)
        position = state[0:3]

        # Base hover reward
        height_diff = abs(position[2] - self.target_height)
        height_reward = 2.0 / (1.0 + height_diff * 5)

        # Obstacle penalty - reduce magnitude
        # Obstacle penalty
        obstacle_penalty = 0
        if not has_collision:
            for obstacle_id in self.obstacle_manager.get_all_obstacles():
                closest_points = self.collision_manager.get_closest_points(
                    self.drone_id, obstacle_id, max_distance=1.0
                )
                if closest_points:
                    distance = closest_points[0][8]  # Distance from point data
                    # Make penalty negative and stronger
                    obstacle_penalty = -5.0 * (1.0 - min(distance, 1.0))
        else:
            obstacle_penalty = -10.0  # Strong collision penalty

        # Final reward combines hover and obstacle penalties
        reward = height_reward + obstacle_penalty
        return float(np.clip(reward, -8, 8))  # Less aggressive clipping

    def is_terminated(self, state: np.ndarray) -> bool:
        position = state[0:3]
        orientation = state[3:6]

        # Log termination checks
        self.logger.info(f"Height: {position[2]:.3f}m")
        self.logger.info(f"Roll: {np.degrees(orientation[0]):.1f}°, Pitch: {np.degrees(orientation[1]):.1f}°")

        if position[2] < 0.02:
            self.logger.info("Terminated: Too low")
            return True

        if np.abs(orientation[0]) > np.pi / 2 or np.abs(orientation[1]) > np.pi / 2:
            self.logger.info("Terminated: Extreme tilt")
            return True

        return False

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

        return state, reward, terminated, truncated, {}

    def _check_collisions(self) -> bool:
        """Check for collisions with obstacles"""
        for obstacle_id in self.obstacle_manager.get_all_obstacles():
            if self.collision_manager.are_objects_colliding(self.drone_id, obstacle_id):
                return True
        return False

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.current_step = 0

        # Reset drone and components
        self.drone_body.reset()
        self.rotor_system.reset()
        self.force_manager.clear_forces()

        # Get initial state
        state = self.get_state()

        info = {
            'reset_position': state[0:3].tolist(),
            'reset_orientation': state[3:6].tolist()
        }

        return state, info