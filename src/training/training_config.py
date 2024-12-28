from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
import yaml


@dataclass
class TrainingConfig:
    """Configuration for RL training"""

    # Algorithm parameters
    algorithm: str = "PPO"
    total_timesteps: int = 2_500_000

    # Network architecture
    policy_network: List[int] = None
    value_network: List[int] = None
    activation: str = "ReLU"

    # PPO specific
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_freq: int = 10000
    log_freq: int = 1000
    eval_freq: int = 10000
    n_eval_episodes: int = 5

    def __post_init__(self):
        if self.policy_network is None:
            self.policy_network = [64, 64]
        if self.value_network is None:
            self.value_network = [64, 64]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainingConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)['training']
            return cls(**config_dict)

    def save(self, path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'training': {
                key: value for key, value in self.__dict__.items()
                if not key.startswith('_')
            }
        }
        with open(path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    def get_sb3_params(self) -> Dict:
        """Get parameters formatted for Stable-Baselines3"""
        return {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'policy_kwargs': {
                'net_arch': {
                    'pi': self.policy_network,
                    'vf': self.value_network
                },
                'activation_fn': getattr(torch.nn, self.activation)
            }
        }