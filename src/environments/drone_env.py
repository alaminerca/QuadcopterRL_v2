from typing import Tuple, Dict, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from typing import List
from ..components.drone.drone_body import DroneBody
from ..components.drone.rotors import RotorSystem
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

    def _init_components(self):
        """Initialize all environment components"""
        # Create drone components
        self.drone_body = DroneBody(self.config['drone'])
        self.rotor_system = RotorSystem(self.config['drone']['rotors'])

        # Create physics components
        self.force_manager = ForceManager(debug=self.config['physics'].get('debug', False))
        self.collision_manager = CollisionManager()

        # Create drone in PyBullet
        self.drone_id = self.drone_body.create(self.physics_client)
        self.rotor_ids = self.rotor_system.create(self.drone_id)

        # Add physics effects
        if self.config['physics'].get('enable_wind', False):
            self.force_manager.add_wind(
                self.drone_id,
                base_magnitude=self.config['physics']['wind_magnitude'],
                variability=self.config['physics']['wind_variability']
            )

        # Add drag forces
        self.force_manager.add_drag(
            self.drone_id,
            drag_coefficient=0.5,
            reference_area=0.1
        )

        # Set up collision detection
        for rotor_id in self.rotor_ids:
            self.collision_manager.add_collision_pair(self.drone_id, rotor_id)

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

    def compute_reward(self, state: np.ndarray, action: List[float]) -> float:
        """
        Compute reward based on current state and action

        Args:
            state: Current state vector
            action: Current action vector

        Returns:
            float: Calculated reward
        """
        action = np.array(action, dtype=np.float32)
        position = state[0:3]
        orientation = state[3:6]
        velocities = state[6:9]
        ang_velocities = state[9:12]

        # Height control (more precise)
        height_diff = abs(position[2] - self.target_height)
        height_reward = 2.0 / (1.0 + height_diff * 5)  # Sharper dropoff

        # Bonus for very stable height
        height_bonus = 3.0 if height_diff < 0.05 else 0.0

        # Stronger penalties for instability
        tilt = abs(orientation[0]) + abs(orientation[1])  # roll + pitch
        tilt_penalty = -tilt * 3.0

        # Penalize rapid movements
        velocity_penalty = -np.sum(np.square(velocities)) * 0.3
        ang_velocity_penalty = -np.sum(np.square(ang_velocities)) * 0.3

        # Encourage smooth, balanced control
        action_smoothness = -np.sum(np.square(action - 0.5)) * 0.5
        action_balance = -np.std(action) * 0.5  # Penalize uneven rotor usage

        # Combine rewards
        reward = (
                height_reward +  # Base height control
                height_bonus +  # Precision bonus
                tilt_penalty +  # Stability
                velocity_penalty +  # Smooth motion
                ang_velocity_penalty +  # Rotation stability
                action_smoothness +  # Smooth control
                action_balance  # Balanced rotors
        )

        # Log detailed rewards if debugging
        if self.current_step % 100 == 0:
            self.logger.debug(f"\nStep {self.current_step} Rewards:")
            self.logger.debug(f"Height: {position[2]:.2f}m (target: {self.target_height}m)")
            self.logger.debug(f"Height Reward: {height_reward:.2f}")
            self.logger.debug(f"Height Bonus: {height_bonus:.2f}")
            self.logger.debug(f"Tilt Penalty: {tilt_penalty:.2f}")
            self.logger.debug(f"Velocity Penalty: {velocity_penalty:.2f}")
            self.logger.debug(f"Action Smoothness: {action_smoothness:.2f}")

        return float(np.clip(reward, -10, 10))

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
        """
        Take a step in the environment

        Args:
            action: Array of normalized [0,1] rotor thrusts

        Returns:
            state: New state
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Apply rotor forces
        self.rotor_system.apply_forces(action)

        # Apply environmental forces
        self.force_manager.apply_forces(self.current_step * self.config['simulation']['time_step'])

        # Step simulation
        p.stepSimulation()

        # Get new state
        state = self.get_state()

        # Calculate reward
        reward = self.compute_reward(state, action)

        # Check termination
        terminated = self.is_terminated(state)
        truncated = self.current_step >= self.config['simulation']['max_steps']

        # Increment step counter
        self.current_step += 1

        # Get additional info
        info = {
            'total_force': self.force_manager.get_force_magnitude(self.drone_id),
            'height': state[2],
            'step': self.current_step
        }

        return state, reward, terminated, truncated, info

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