"""
LoRA Training Script for FairSkin Diffusion.

Trains Low-Rank Adaptation (LoRA) on Stable Diffusion v1.5 using
HAM10000 dermoscopy dataset with FST-balanced prompting.

Usage:
    python experiments/augmentation/train_lora.py --config configs/fairskin_config.yaml

Expected Training Time (RTX 3090):
    - 10,000 steps: ~10-20 GPU hours
    - 5,000 steps: ~5-10 GPU hours

Framework: MENDICANT_BIAS - Phase 2
Agent: HOLLOWED_EYES
Version: 0.3.0
Date: 2025-10-13
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

from src.data.ham10000_dataset import HAM10000Dataset, load_splits
from src.augmentation.lora_trainer import LoRATrainer, LoRATrainingConfig


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train LoRA for FairSkin diffusion model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/fairskin_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory from config"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Override number of training steps"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate"
    )

    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="Override LoRA rank"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda or cpu)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test mode (100 steps only)"
    )

    return parser.parse_args()


def setup_training_config(config: dict, args: argparse.Namespace) -> LoRATrainingConfig:
    """
    Create LoRA training configuration from YAML and CLI args.

    Args:
        config: Loaded YAML configuration
        args: Parsed command-line arguments

    Returns:
        LoRATrainingConfig instance
    """
    # Extract config sections
    lora_cfg = config.get('lora', {})
    training_cfg = config.get('training', {})
    sd_cfg = config.get('stable_diffusion', {})

    # Build training config
    training_config = LoRATrainingConfig(
        # Model
        model_id=sd_cfg.get('model_id', 'runwayml/stable-diffusion-v1-5'),
        lora_rank=args.lora_rank or lora_cfg.get('rank', 16),
        lora_alpha=lora_cfg.get('alpha', 16),
        lora_dropout=lora_cfg.get('dropout', 0.1),
        target_modules=lora_cfg.get('target_modules', ["to_q", "to_k", "to_v", "to_out.0"]),

        # Training
        num_train_steps=args.num_steps or training_cfg.get('num_train_steps', 10000),
        batch_size=args.batch_size or training_cfg.get('batch_size', 4),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 2),
        learning_rate=args.learning_rate or training_cfg.get('learning_rate', 1e-4),
        lr_scheduler=training_cfg.get('lr_scheduler', 'cosine_with_restarts'),
        lr_warmup_steps=training_cfg.get('lr_warmup_steps', 500),
        num_cycles=training_cfg.get('num_cycles', 3),
        weight_decay=training_cfg.get('weight_decay', 0.01),
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),

        # Diffusion
        num_train_timesteps=training_cfg.get('num_train_timesteps', 1000),
        beta_schedule=training_cfg.get('beta_schedule', 'scaled_linear'),
        prediction_type=training_cfg.get('prediction_type', 'epsilon'),
        snr_gamma=training_cfg.get('snr_gamma', 5.0),

        # Data
        resolution=training_cfg.get('resolution', 512),
        center_crop=training_cfg.get('center_crop', True),
        random_flip=training_cfg.get('random_flip', True),

        # Optimization
        use_8bit_adam=training_cfg.get('use_8bit_adam', False),
        mixed_precision=training_cfg.get('mixed_precision', True),
        gradient_checkpointing=training_cfg.get('gradient_checkpointing', True),

        # Logging & Checkpointing
        logging_steps=training_cfg.get('logging_steps', 100),
        checkpoint_steps=training_cfg.get('checkpoint_steps', 1000),
        validation_steps=training_cfg.get('validation_steps', 500),
        output_dir=args.output_dir or training_cfg.get('output_dir', 'checkpoints/fairskin_lora'),

        # Hardware
        device=args.device or sd_cfg.get('device', 'cuda'),
        dataloader_num_workers=training_cfg.get('dataloader_num_workers', 4),
    )

    # Quick test mode
    if args.quick_test:
        print("\n" + "=" * 70)
        print("QUICK TEST MODE - Training for 100 steps only")
        print("=" * 70 + "\n")
        training_config.num_train_steps = 100
        training_config.checkpoint_steps = 50
        training_config.logging_steps = 10

    return training_config


def load_datasets(config: dict, args: argparse.Namespace):
    """
    Load HAM10000 training and validation datasets.

    Args:
        config: Configuration dictionary
        args: Command-line arguments

    Returns:
        (train_dataset, val_dataset) tuple
    """
    # Get data paths
    data_cfg = config.get('data', {})
    data_dir = args.data_dir or data_cfg.get('real_data_dir', 'data/raw/ham10000')
    metadata_path = data_cfg.get('real_metadata', None)
    splits_file = data_cfg.get('splits_file', None)

    print("\nLoading HAM10000 dataset...")
    print(f"  Data directory: {data_dir}")

    # Load splits if available
    if splits_file and Path(splits_file).exists():
        print(f"  Using splits from: {splits_file}")
        splits = load_splits(splits_file)
        train_indices = splits['train']
        val_indices = splits.get('val', None)
    else:
        print(f"  No splits file found, using all data for training")
        train_indices = None
        val_indices = None

    # Load training dataset
    train_dataset = HAM10000Dataset(
        root_dir=data_dir,
        metadata_path=metadata_path,
        split="train",
        split_indices=train_indices,
        transform=None,  # Transforms handled by LoRATrainer
        use_fst_annotations=True,
        estimate_fst_if_missing=True,
    )

    # Load validation dataset
    val_dataset = None
    if val_indices is not None:
        val_dataset = HAM10000Dataset(
            root_dir=data_dir,
            metadata_path=metadata_path,
            split="val",
            split_indices=val_indices,
            transform=None,
            use_fst_annotations=True,
            estimate_fst_if_missing=True,
        )

    return train_dataset, val_dataset


def main():
    """Main training script."""
    # Parse arguments
    args = parse_args()

    print("=" * 70)
    print("FairSkin LoRA Training")
    print("=" * 70)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Setup training config
    training_config = setup_training_config(config, args)

    print("\nTraining Configuration:")
    print(f"  Model: {training_config.model_id}")
    print(f"  LoRA rank: {training_config.lora_rank}")
    print(f"  LoRA alpha: {training_config.lora_alpha}")
    print(f"  Training steps: {training_config.num_train_steps}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Mixed precision: {training_config.mixed_precision}")
    print(f"  Device: {training_config.device}")
    print(f"  Output: {training_config.output_dir}")

    # Load datasets
    train_dataset, val_dataset = load_datasets(config, args)

    print(f"\nDatasets loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation samples: {len(val_dataset)}")

    # Set random seed
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"\nRandom seed set to: {seed}")

    # Initialize trainer
    print("\nInitializing LoRA trainer...")
    trainer = LoRATrainer(
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "=" * 70)
    print("Starting LoRA training...")
    print("=" * 70 + "\n")

    try:
        trainer.train()

        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print("=" * 70)

        print(f"\nFinal checkpoint saved to:")
        print(f"  {training_config.output_dir}/lora_weights_final.pt")

        print(f"\nTo generate synthetic images, run:")
        print(f"  python experiments/augmentation/generate_fairskin.py \\")
        print(f"    --lora_weights {training_config.output_dir}/lora_weights_final.pt")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved.")

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
