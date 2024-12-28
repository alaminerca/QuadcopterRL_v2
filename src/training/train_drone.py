"""Main script for training the drone"""
import argparse
from pathlib import Path

from training_config import TrainingConfig
from training_manager import TrainingManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train drone using PPO")
    parser.add_argument("--config", type=str, default="configs/default_training.yaml",
                        help="Path to training configuration file")
    parser.add_argument("--env-config", type=str, default="configs/default_env.yaml",
                        help="Path to environment configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume training from")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load training configuration
    config = TrainingConfig.from_yaml(args.config)

    # Create training manager
    trainer = TrainingManager(config, args.env_config)

    # Save configuration for reference
    trainer.save_config()

    # Start training
    trainer.train(resume_from=args.resume)

    # Run evaluation
    if Path(trainer.model_dir / "final_model.zip").exists():
        metrics = trainer.evaluate(
            str(trainer.model_dir / "final_model.zip"),
            n_episodes=10
        )
        print("\nFinal Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()