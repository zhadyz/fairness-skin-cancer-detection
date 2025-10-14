"""
CIRCLe Training Script

Command-line interface for training CIRCLe color-invariant model on HAM10000 dataset.

Usage:
    # Train with default configuration
    python experiments/fairness/train_circle.py

    # Train with custom config
    python experiments/fairness/train_circle.py --config configs/circle_config.yaml

    # Train with specific hyperparameters
    python experiments/fairness/train_circle.py --lambda_reg 0.3 --epochs 50

    # Initialize from FairDisCo checkpoint
    python experiments/fairness/train_circle.py --fairdisco_checkpoint experiments/fairdisco/checkpoints/fairdisco_best.pth

Framework: MENDICANT_BIAS - Phase 2, Week 7-8
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.circle_model import create_circle_model
from src.training.circle_trainer import CIRCLe_Trainer, CIRCLe_TrainerConfig
# Placeholder imports - adjust based on actual dataset implementation
# from src.data.ham10000_dataset import HAM10000Dataset
# from src.data.augmentation import get_train_transforms, get_val_transforms


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        # Load first document only (ignore alternative configs)
        config = yaml.safe_load(f)
    return config


def create_trainer_config(config_dict: dict, args: argparse.Namespace) -> CIRCLe_TrainerConfig:
    """Create trainer configuration from YAML and CLI arguments."""

    # Override config with CLI arguments if provided
    if args.lambda_reg is not None:
        config_dict['training']['lambda_reg'] = args.lambda_reg
    if args.lambda_adv is not None:
        config_dict['training']['lambda_adv'] = args.lambda_adv
    if args.lambda_con is not None:
        config_dict['training']['lambda_con'] = args.lambda_con
    if args.epochs is not None:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config_dict['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config_dict['training']['learning_rate'] = args.learning_rate

    # Extract configuration values
    model_cfg = config_dict['model']
    train_cfg = config_dict['training']
    data_cfg = config_dict['data']
    checkpoint_cfg = config_dict['checkpointing']
    logging_cfg = config_dict['logging']
    hardware_cfg = config_dict['hardware']
    repro_cfg = config_dict['reproducibility']

    # Create trainer config
    trainer_config = CIRCLe_TrainerConfig(
        # Model name
        model_name=model_cfg['name'],

        # Training hyperparameters
        epochs=train_cfg['epochs'],
        batch_size=train_cfg['batch_size'],
        learning_rate=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],

        # Loss weights
        lambda_adv=train_cfg['lambda_adv'],
        lambda_con=train_cfg['lambda_con'],
        lambda_reg=train_cfg['lambda_reg'],
        temperature=train_cfg['temperature'],

        # Lambda scheduling (FairDisCo)
        use_lambda_schedule=train_cfg['use_lambda_schedule'],
        lambda_schedule_start_epoch=train_cfg['lambda_schedule_start_epoch'],
        lambda_schedule_end_epoch=train_cfg['lambda_schedule_end_epoch'],
        lambda_schedule_start_value=train_cfg['lambda_schedule_start_value'],
        lambda_schedule_end_value=train_cfg['lambda_schedule_end_value'],

        # Lambda scheduling (CIRCLe)
        use_lambda_reg_schedule=train_cfg['use_lambda_reg_schedule'],
        lambda_reg_schedule_start_epoch=train_cfg['lambda_reg_schedule_start_epoch'],
        lambda_reg_schedule_end_epoch=train_cfg['lambda_reg_schedule_end_epoch'],
        lambda_reg_schedule_start_value=train_cfg['lambda_reg_schedule_start_value'],
        lambda_reg_schedule_end_value=train_cfg['lambda_reg_schedule_end_value'],

        # CIRCLe parameters
        target_fsts=model_cfg['target_fsts'],
        use_multi_target=model_cfg['use_multi_target'],
        distance_metric=model_cfg['distance_metric'],

        # Monitoring
        monitor_discriminator=train_cfg['monitor_discriminator'],
        discriminator_target_acc=train_cfg['discriminator_target_acc'],
        discriminator_acc_tolerance=train_cfg['discriminator_acc_tolerance'],
        monitor_tone_invariance=train_cfg['monitor_tone_invariance'],
        tone_invariance_metric=train_cfg['tone_invariance_metric'],

        # Regularization
        label_smoothing=train_cfg['label_smoothing'],
        gradient_clip_norm=train_cfg['gradient_clip_norm'],

        # Early stopping
        early_stopping=train_cfg['early_stopping'],
        patience=train_cfg['patience'],

        # Optimization
        use_amp=train_cfg['use_amp'],

        # Scheduler
        scheduler_t0=train_cfg['scheduler_t0'],
        scheduler_t_mult=train_cfg['scheduler_t_mult'],
        scheduler_eta_min=train_cfg['scheduler_eta_min'],

        # Checkpointing
        checkpoint_dir=checkpoint_cfg['checkpoint_dir'],
        save_best_only=checkpoint_cfg['save_best_only'],
        save_frequency=checkpoint_cfg['save_frequency'],

        # Logging
        log_dir=logging_cfg['log_dir'],

        # Hardware
        device=hardware_cfg['device']
    )

    return trainer_config


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train CIRCLe color-invariant model")

    # Configuration
    parser.add_argument('--config', type=str, default='configs/circle_config.yaml',
                       help='Path to configuration file')

    # Model parameters
    parser.add_argument('--fairdisco_checkpoint', type=str, default=None,
                       help='Path to FairDisCo checkpoint to initialize from')

    # Training hyperparameters (override config)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--lambda_reg', type=float, default=None,
                       help='CIRCLe regularization weight')
    parser.add_argument('--lambda_adv', type=float, default=None,
                       help='Adversarial loss weight')
    parser.add_argument('--lambda_con', type=float, default=None,
                       help='Contrastive loss weight')

    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda, cpu, mps)')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable mixed precision training')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load configuration
    print(f"\n{'='*80}")
    print(f"CIRCLe Training Script")
    print(f"{'='*80}\n")
    print(f"Loading configuration from: {args.config}")

    config_dict = load_config(args.config)

    # Set random seed
    seed = args.seed if args.seed is not None else config_dict['reproducibility']['seed']
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Create trainer configuration
    trainer_config = create_trainer_config(config_dict, args)

    if args.no_amp:
        trainer_config.use_amp = False

    if args.device is not None:
        trainer_config.device = args.device

    print(f"\nTrainer Configuration:")
    print(f"  Epochs: {trainer_config.epochs}")
    print(f"  Batch size: {trainer_config.batch_size}")
    print(f"  Learning rate: {trainer_config.learning_rate}")
    print(f"  Lambda (adv/con/reg): {trainer_config.lambda_adv}/{trainer_config.lambda_con}/{trainer_config.lambda_reg}")
    print(f"  Target FSTs: {trainer_config.target_fsts}")
    print(f"  Device: {trainer_config.device}")
    print(f"  Mixed precision: {trainer_config.use_amp}")

    # Create model
    print(f"\nCreating CIRCLe model...")
    model = create_circle_model(
        num_classes=config_dict['model']['num_classes'],
        num_fst_classes=config_dict['model']['num_fst_classes'],
        backbone=config_dict['model']['backbone'],
        pretrained=config_dict['model']['pretrained'],
        contrastive_dim=config_dict['model']['contrastive_dim'],
        lambda_adv=trainer_config.lambda_adv,
        lambda_con=trainer_config.lambda_con,
        lambda_reg=trainer_config.lambda_reg,
        target_fsts=trainer_config.target_fsts,
        distance_metric=config_dict['model']['distance_metric'],
        fairdisco_checkpoint=args.fairdisco_checkpoint or config_dict['checkpointing'].get('load_fairdisco_checkpoint')
    )

    print(f"  Architecture: {model.get_model_info()['architecture']}")
    print(f"  Backbone: {model.get_model_info()['backbone']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders
    print(f"\nPreparing data loaders...")
    print(f"  Dataset: {config_dict['data']['dataset']}")
    print(f"  Data root: {config_dict['data']['data_root']}")

    # NOTE: This is a placeholder. Actual dataset implementation needed.
    # For now, create dummy data loaders for testing
    print(f"  WARNING: Using dummy data loaders (implement actual dataset)")

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, 7, (1,)).item()
            fst = torch.randint(1, 7, (1,)).item()
            return image, label, fst

    train_dataset = DummyDataset(size=1000)
    val_dataset = DummyDataset(size=200)

    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=config_dict['data']['num_workers'],
        pin_memory=config_dict['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=config_dict['data']['num_workers'],
        pin_memory=config_dict['data']['pin_memory']
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = CIRCLe_Trainer(model, trainer_config)

    # Train model
    print(f"\nStarting training...")
    history = trainer.fit(train_loader, val_loader)

    # Print final results
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")
    print(f"Best validation AUROC: {max(history['val_auroc']):.4f}")
    print(f"Final AUROC gap: {history['val_auroc_gap'][-1]:.4f}")
    print(f"Final EOD: {history['val_eod'][-1]:.4f}")
    print(f"Final tone-invariance: {history['tone_invariance_score'][-1]:.4f}")
    print(f"\nCheckpoints saved to: {trainer_config.checkpoint_dir}")
    print(f"Logs saved to: {trainer_config.log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {trainer_config.log_dir}")


if __name__ == "__main__":
    main()
