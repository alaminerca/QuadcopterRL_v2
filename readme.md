# QuadcopterRL_v2: Advanced Drone Control Using Reinforcement Learning

## Project Overview
Advanced implementation of a quadcopter drone simulation and control system using PyBullet physics engine and Reinforcement Learning.

## Project Structure
```
QuadcopterRL_v2/
├── src/                           # Source code directory
│   ├── environments/              # Environment-related modules
│   │   ├── base_env.py           # Base environment class
│   │   └── drone_env.py          # Main drone environment
│   │
│   ├── components/               # Modular components
│   │   ├── drone/               # Drone-specific components
│   │   │   ├── drone_body.py    # Drone physical structure
│   │   │   └── rotors.py        # Rotor management
│   │   │
│   │   └── physics/             # Physics-related components
│   │       ├── forces.py        # Force calculations
│   │       └── collisions.py    # Collision detection
│   │
│   └── utils/                   # Utility functions
│       └── config_utils.py      # Configuration utilities
│
├── configs/                     # Configuration files
│   ├── default_env.yaml        # Default environment config
│   └── default_training.yaml   # Default training config
│
├── tests/                      # Test files
│   ├── test_environments/
│   ├── test_physics/
│   └── test_utils/
│
└── requirements/
    ├── base.txt               # Base requirements
    └── test.txt              # Testing requirements

```

## Completed Milestones

### Milestone 1: Core Systems (✓)
- Configuration Management System
  - YAML-based configuration
  - Dynamic config loading
  - Validation system

### Milestone 2: Physics Engine (✓)
- Force System
  - Accurate force calculations
  - Wind simulation
  - Drag forces
  - Force visualization

### Milestone 3: Collision System (✓)
- Collision Detection
  - Box collision detection
  - Raycast collision
  - Collision callbacks
  - Contact point detection

### Milestone 4: Basic Drone Control (✓)
- Hover Stabilization
  - Height maintenance
  - Force distribution
  - Rotor management

## Setup Instructions

1. Create environment:
```bash
conda create -n drone_rl_new python=3.9
conda activate drone_rl_new
```

2. Install requirements:
```bash
pip install -r requirements/base.txt
pip install -r requirements/test.txt
```

## Running Tests
Run all tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_physics/test_forces.py -v
```

## Dependencies
- Python 3.9
- PyBullet 3.2.5
- Gymnasium 0.29.1
- PyYAML
- NumPy 1.24.3
- Pytest (for testing)

## Next Steps
- Training environment setup
- PPO implementation
- Training visualization
- Performance metrics

## License
MIT License

## Contributors
- Al-Amine MOUHAMAD | @alaminerca