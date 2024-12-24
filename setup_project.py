#!/usr/bin/env python3
import os
import shutil

def create_directory_structure():
    """Create the new project directory structure"""
    
    # Define the directory structure
    directories = [
        # Source code directories
        "src/environments",
        "src/components/drone",
        "src/components/obstacles",
        "src/components/physics",
        "src/training",
        "src/visualization",
        "src/utils",
        
        # Configuration directories
        "configs/custom_configs",
        
        # Test directories
        "tests/test_environments",
        "tests/test_components",
        "tests/test_physics",
        "tests/test_integration",
        
        # Other directories
        "models/trained",
        "logs/training_logs",
        "logs/debug_logs",
        "docs/api",
        "docs/tutorials",
        "docs/examples",
        "scripts",
        "requirements"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py for Python packages
        if directory.startswith(("src/", "tests/")):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                open(init_file, 'a').close()

def backup_current_code():
    """Backup current working code"""
    backup_dir = "backup_original"
    os.makedirs(backup_dir, exist_ok=True)
    
    # List of files to backup
    files_to_backup = [
        "drone_env.py",
        "train.py",
        "run_model.py",
        "requirements.txt",
        "README.md"
    ]
    
    # Copy each file to backup directory
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(backup_dir, file))

def create_initial_files():
    """Create initial configuration and setup files"""
    
    # Create base requirements files
    requirements = {
        "requirements/base.txt": "\n".join([
            "numpy==1.24.3",
            "torch==2.1.0",
            "gymnasium==0.29.1",
            "pybullet==3.2.5",
            "stable-baselines3==2.1.0",
            "tensorboard==2.14.0",
            "PyYAML>=6.0.1"  # Added for config management
        ]),
        "requirements/dev.txt": "\n".join([
            "-r base.txt",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0"
        ]),
        "requirements/test.txt": "\n".join([
            "-r base.txt",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0"
        ])
    }
    
    # Create requirements files
    for file_path, content in requirements.items():
        with open(file_path, 'w') as f:
            f.write(content)
    
    # Create default configuration file
    default_config = """# Default Environment Configuration
environment:
  simulation:
    time_step: 0.02
    max_steps: 1000
  
  drone:
    mass: 0.7
    dimensions: [0.2, 0.2, 0.1]
    rotors:
      count: 4
      max_thrust: 2.58
  
  physics:
    gravity: -9.81
    wind_enabled: false
    rain_enabled: false
  
  obstacles:
    static:
      enabled: false
      count: 0
    dynamic:
      enabled: false
      bird_count: 0
"""
    
    with open('configs/default_env.yaml', 'w') as f:
        f.write(default_config)

def main():
    """Main setup function"""
    print("Starting project setup...")
    
    # Backup current code
    print("Backing up current code...")
    backup_current_code()
    
    # Create new directory structure
    print("Creating directory structure...")
    create_directory_structure()
    
    # Create initial files
    print("Creating initial configuration files...")
    create_initial_files()
    
    print("\nProject setup completed successfully!")
    print("\nNext steps:")
    print("1. Review the backup_original directory for original code")
    print("2. Start migrating components to the new structure")
    print("3. Update import statements in existing code")
    print("4. Run tests to verify structure")

if __name__ == "__main__":
    main()