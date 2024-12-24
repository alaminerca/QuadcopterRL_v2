import os
import gymnasium as gym
import pybullet as p
import pybullet_data
import yaml
from abc import ABC, abstractmethod

class BaseEnv(gym.Env, ABC):
    """Abstract base class for PyBullet-based environments"""
    
    def __init__(self, config_path="configs/default_env.yaml"):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize PyBullet
        self._setup_pybullet()
        
        # Initialize environment components
        self.components = {}
        
    def _load_config(self, config_path):
        """Load environment configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['environment']

    def _setup_pybullet(self):
        """Setup PyBullet simulation"""
        try:
            p.disconnect()
        except:
            pass

        # Connect and configure PyBullet
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # Set camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )

        # Set gravity and load ground plane
        p.setGravity(0, 0, self.config['physics']['gravity'])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load and configure ground plane
        self.plane = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
    
    @abstractmethod
    def step(self, action):
        """Take a step in the environment"""
        pass
    
    @abstractmethod
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        pass
    
    def render(self):
        """
        Render the environment
        PyBullet environments are automatically rendered if using GUI connection
        """
        pass
    
    def close(self):
        """Clean up environment"""
        try:
            p.disconnect()
        except:
            pass
    
    def add_component(self, name, component):
        """Add a component to the environment"""
        self.components[name] = component
    
    def get_component(self, name):
        """Get a component by name"""
        return self.components.get(name)
    
    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
