import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path


@dataclass
class ConfigPaths:
    """Stores paths to configuration files"""
    env_config: str = "configs/default_env.yaml"
    training_config: str = "configs/default_training.yaml"
    custom_config: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, str]) -> 'ConfigPaths':
        """Create ConfigPaths from dictionary"""
        return cls(
            env_config=config_dict.get('env_config', cls.env_config),
            training_config=config_dict.get('training_config', cls.training_config),
            custom_config=config_dict.get('custom_config')
        )


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors"""
    pass


class ConfigManager:
    """Manages loading and validation of configuration files"""

    REQUIRED_ENV_KEYS = {
        'environment.simulation.time_step',
        'environment.simulation.max_steps',
        'environment.drone.mass',
        'environment.drone.dimensions',
        'environment.physics.gravity'
    }

    REQUIRED_TRAINING_KEYS = {
        'training.algorithm',
        'training.total_timesteps',
        'training.hyperparameters.learning_rate',
        'training.hyperparameters.gamma'
    }

    def __init__(self, config_paths: Union[ConfigPaths, Dict[str, str]] = None):
        """
        Initialize configuration manager

        Args:
            config_paths: ConfigPaths object or dictionary with paths
        """
        if isinstance(config_paths, dict):
            self.config_paths = ConfigPaths.from_dict(config_paths)
        else:
            self.config_paths = config_paths or ConfigPaths()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Initialize configurations
        self.env_config: Dict = {}
        self.training_config: Dict = {}
        self.custom_config: Dict = {}
        self.config: Dict = {}

        # Load and validate configurations
        self._initialize_configs()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _initialize_configs(self) -> None:
        """Initialize all configurations"""
        try:
            # Load base configurations
            self.env_config = self._load_yaml(self.config_paths.env_config)
            self.training_config = self._load_yaml(self.config_paths.training_config)

            # Load custom config if provided
            if self.config_paths.custom_config:
                self.custom_config = self._load_yaml(self.config_paths.custom_config)

            # Merge configurations
            self.config = self._merge_configs()

            # Validate configurations
            self._validate_config()

        except Exception as e:
            self.logger.error(f"Error initializing configurations: {e}")
            raise

    def _load_yaml(self, path: str) -> Dict:
        """
        Load YAML configuration file

        Args:
            path: Path to YAML file

        Returns:
            Dict: Loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with path.open('r') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    raise ValueError(f"Invalid config format in {path}")
                return config
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML from {path}: {e}")
            raise

    def _merge_configs(self) -> Dict:
        """
        Merge all configurations with custom overrides

        Returns:
            Dict: Merged configuration
        """
        config = deepcopy(self.env_config)

        # Merge training config
        if 'training' not in config:
            config['training'] = {}
        config['training'].update(self.training_config.get('training', {}))

        # Apply custom overrides
        self._deep_update(config, self.custom_config)

        return config

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """
        Recursively update dictionary

        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = deepcopy(value)

    def _validate_config(self) -> None:
        """
        Validate configuration values

        Raises:
            ConfigValidationError: If validation fails
        """
        # Check required keys
        self._check_required_keys()

        # Validate numeric ranges
        self._validate_numeric_ranges()

        # Validate specific components
        self._validate_drone_config()
        self._validate_physics_config()
        self._validate_training_config()

    def _check_required_keys(self) -> None:
        """Check if all required keys are present"""

        def get_value(d: Dict, path: str) -> Any:
            try:
                current = d
                for key in path.split('.'):
                    current = current[key]
                return current
            except KeyError:
                return None

        missing_keys = []

        # Check environment keys
        for key in self.REQUIRED_ENV_KEYS:
            if get_value(self.config, key) is None:
                missing_keys.append(key)

        # Check training keys
        for key in self.REQUIRED_TRAINING_KEYS:
            if get_value(self.config, key) is None:
                missing_keys.append(key)

        if missing_keys:
            raise ConfigValidationError(f"Missing required keys: {missing_keys}")

    def _validate_numeric_ranges(self) -> None:
        """Validate numeric values are within acceptable ranges"""
        checks = [
            (self.config['environment']['simulation']['time_step'] > 0,
             "time_step must be positive"),
            (self.config['environment']['simulation']['max_steps'] > 0,
             "max_steps must be positive"),
            (self.config['environment']['drone']['mass'] > 0,
             "drone mass must be positive"),
            (0 < self.config['training']['hyperparameters']['learning_rate'] < 1,
             "learning_rate must be between 0 and 1"),
            (0 <= self.config['training']['hyperparameters']['gamma'] <= 1,
             "gamma must be between 0 and 1")
        ]

        for condition, message in checks:
            if not condition:
                raise ConfigValidationError(message)

    def _validate_drone_config(self) -> None:
        """Validate drone-specific configuration"""
        drone_config = self.config['environment']['drone']

        # Validate dimensions
        if len(drone_config['dimensions']) != 3:
            raise ConfigValidationError("drone dimensions must have 3 components")

        # Validate rotor configuration
        rotors = drone_config['rotors']
        if rotors['count'] != len(rotors['positions']):
            raise ConfigValidationError(
                "rotor count must match number of rotor positions"
            )

    def _validate_physics_config(self) -> None:
        """Validate physics configuration"""
        physics = self.config['environment']['physics']

        # Check wind settings if enabled
        if physics.get('wind', {}).get('enabled', False):
            wind = physics['wind']
            if not (0 <= wind['variability'] <= 1):
                raise ConfigValidationError("wind variability must be between 0 and 1")

    def _validate_training_config(self) -> None:
        """Validate training configuration"""
        curriculum = self.config['training'].get('curriculum', {})
        if curriculum.get('enabled', False):
            if not curriculum.get('stages'):
                raise ConfigValidationError("Curriculum enabled but no stages defined")

    def get_config(self, section: Optional[str] = None) -> Dict:
        """
        Get configuration or section

        Args:
            section: Optional dot-notation path to config section

        Returns:
            Dict: Requested configuration section or full config
        """
        if not section:
            return deepcopy(self.config)

        try:
            current = self.config
            for key in section.split('.'):
                current = current[key]
            return deepcopy(current)
        except KeyError:
            raise KeyError(f"Configuration section not found: {section}")

    def set_config(self, section: str, value: Any) -> None:
        """
        Set configuration value

        Args:
            section: Dot-notation path to config section
            value: Value to set
        """
        keys = section.split('.')
        current = self.config

        for key in keys[:-1]:
            current = current.setdefault(key, {})

        current[keys[-1]] = deepcopy(value)
        self._validate_config()

    def save_config(self, path: str, format: str = 'yaml') -> None:
        """
        Save current configuration to file

        Args:
            path: Output file path
            format: Output format ('yaml' or 'json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open('w') as f:
                if format.lower() == 'yaml':
                    yaml.safe_dump(self.config, f, default_flow_style=False)
                elif format.lower() == 'json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            self.logger.error(f"Error saving config to {path}: {e}")
            raise

    @staticmethod
    def get_config_template(config_type: str = 'env') -> Dict:
        """
        Get configuration template

        Args:
            config_type: Type of config template ('env' or 'training')

        Returns:
            Dict: Template configuration
        """
        if config_type == 'env':
            return {
                'environment': {
                    'simulation': {
                        'time_step': 0.02,
                        'max_steps': 1000
                    },
                    'drone': {
                        'mass': 0.7,
                        'dimensions': [0.2, 0.2, 0.1]
                    },
                    'physics': {
                        'gravity': -9.81
                    }
                }
            }
        elif config_type == 'training':
            return {
                'training': {
                    'algorithm': 'PPO',
                    'total_timesteps': 1000000,
                    'hyperparameters': {
                        'learning_rate': 3e-4,
                        'gamma': 0.99
                    }
                }
            }
        else:
            raise ValueError(f"Unsupported config type: {config_type}")

    def get_flattened_config(self) -> Dict[str, Any]:
        """
        Get flattened configuration with dot notation keys

        Returns:
            Dict: Flattened configuration
        """

        def flatten_dict(d: Dict, parent_key: str = '') -> Dict:
            items: List[Tuple[str, Any]] = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return flatten_dict(self.config)

    def diff_configs(self, other_config: Union[Dict, 'ConfigManager']) -> Dict[str, Tuple[Any, Any]]:
        """
        Compare current config with another config

        Args:
            other_config: Configuration to compare with

        Returns:
            Dict: Differences between configurations {key: (current_value, other_value)}
        """
        if isinstance(other_config, ConfigManager):
            other_flat = other_config.get_flattened_config()
        elif isinstance(other_config, dict):
            temp_manager = ConfigManager(config_paths=self.config_paths)
            temp_manager.config = deepcopy(other_config)
            other_flat = temp_manager.get_flattened_config()
        else:
            other_flat = {}

        current_flat = self.get_flattened_config()

        diffs = {}
        all_keys = set(current_flat.keys()) | set(other_flat.keys())
        for key in all_keys:
            current_value = current_flat.get(key)
            other_value = other_flat.get(key)
            if current_value != other_value:
                diffs[key] = (current_value, other_value)

        return diffs