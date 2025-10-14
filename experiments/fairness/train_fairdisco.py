"""
FairDisCo Training Script for HAM10000 Dataset

Complete training pipeline for FairDisCo adversarial debiasing model.
Implements three-loss training with lambda scheduling and comprehensive
fairness evaluation.

Usage:
    python experiments/fairness/train_fairdisco.py --config configs/fairdisco_config.yaml

Framework: MENDICANT_BIAS - Phase 2, Week 5-6
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.fairdisco_model import create_fairdisco_model
from src.training.fairdisco_trainer import FairDisCoTrainer, FairDisCoTrainerConfig
from src.data.ham10000_dataset import HAM10000Dataset, create_fst_stratified_splits
from src.evaluation.fairness_metrics import FairnessMetrics


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_transforms(config: dict) -> tuple:
    """
    Create training and validation data transforms.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_transform, val_transform)
    """
    img_size = config['data']['img_size']
    mean = config['data']['mean']
    std = config['data']['std']

    # Training transforms with augmentation
    if config['data']['augmentation']['enabled']:
        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=config['data']['augmentation']['horizontal_flip']),
            A.VerticalFlip(p=config['data']['augmentation']['vertical_flip']),
            A.Rotate(limit=config['data']['augmentation']['rotation_limit'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config['data']['augmentation']['brightness_contrast'],
                contrast_limit=config['data']['augmentation']['brightness_contrast'],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(config['data']['augmentation']['hue_saturation'] * 255),
                sat_shift_limit=int(config['data']['augmentation']['hue_saturation'] * 255),
                val_shift_limit=int(config['data']['augmentation']['hue_saturation'] * 255),
                p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return train_transform, val_transform


def create_dataloaders(config: dict, train_transform, val_transform) -> tuple:
    """
    Create train and validation data loaders.

    Args:
        config: Configuration dictionary
        train_transform: Training data transform
        val_transform: Validation data transform

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_root = config['data']['data_root']

    # Check if splits exist, otherwise create them
    splits_path = Path(data_root) / 'ham10000_splits.json'

    if not splits_path.exists():
        print(f"\nCreating stratified splits...")
        metadata_path = Path(data_root) / 'HAM10000_metadata.csv'

        splits = create_fst_stratified_splits(
            metadata_path=str(metadata_path),
            output_path=str(splits_path),
            train_ratio=config['data']['train_split'],
            val_ratio=config['data']['val_split'],
            test_ratio=config['data']['test_split'],
            random_seed=config['reproducibility']['seed'],
            stratify_by_fst=True
        )
    else:
        print(f"\nLoading existing splits from {splits_path}")
        import json
        with open(splits_path, 'r') as f:
            splits = json.load(f)

    # Create datasets
    print("\nLoading HAM10000 training dataset...")
    train_dataset = HAM10000Dataset(
        root_dir=data_root,
        split='train',
        split_indices=splits['train'],
        transform=train_transform,
        use_fst_annotations=config['data']['use_fst_annotations'],
        estimate_fst_if_missing=config['data']['estimate_fst_if_missing'],
        fst_csv_path=config['data'].get('fst_csv_path')
    )

    print("\nLoading HAM10000 validation dataset...")
    val_dataset = HAM10000Dataset(
        root_dir=data_root,
        split='val',
        split_indices=splits['val'],
        transform=val_transform,
        use_fst_annotations=config['data']['use_fst_annotations'],
        estimate_fst_if_missing=config['data']['estimate_fst_if_missing'],
        fst_csv_path=config['data'].get('fst_csv_path')
    )

    # Create custom collate function to handle dictionary outputs
    def collate_fn(batch):
        """Custom collate function for HAM10000Dataset."""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        fst_labels = torch.tensor([item['fst'] for item in batch])
        return images, labels, fst_labels

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['data']['shuffle_train'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def create_trainer_config(config: dict) -> FairDisCoTrainerConfig:
    """
    Create FairDisCoTrainerConfig from YAML configuration.

    Args:
        config: Configuration dictionary

    Returns:
        FairDisCoTrainerConfig instance
    """
    return FairDisCoTrainerConfig(
        # Model settings
        model_name=config['experiment']['name'],
        num_classes=config['model']['num_classes'],
        device=config['hardware']['device'],

        # Training hyperparameters
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        optimizer=config['training']['optimizer'],

        # Learning rate scheduling
        scheduler=config['training']['scheduler'],
        scheduler_t0=config['training']['scheduler_t0'],
        scheduler_t_mult=config['training']['scheduler_t_mult'],
        scheduler_eta_min=config['training']['scheduler_eta_min'],

        # FairDisCo-specific
        lambda_adv=config['training']['lambda_adv'],
        lambda_con=config['training']['lambda_con'],
        temperature=config['training']['temperature'],

        # Lambda scheduling
        use_lambda_schedule=config['training']['use_lambda_schedule'],
        lambda_schedule_start_epoch=config['training']['lambda_schedule_start_epoch'],
        lambda_schedule_end_epoch=config['training']['lambda_schedule_end_epoch'],
        lambda_schedule_start_value=config['training']['lambda_schedule_start_value'],
        lambda_schedule_end_value=config['training']['lambda_schedule_end_value'],

        # Discriminator monitoring
        monitor_discriminator=config['training']['monitor_discriminator'],
        discriminator_target_acc=config['training']['discriminator_target_acc'],
        discriminator_acc_tolerance=config['training']['discriminator_acc_tolerance'],

        # Loss configuration
        label_smoothing=config['training']['label_smoothing'],
        gradient_clip_norm=config['training']['gradient_clip_norm'],

        # Early stopping
        early_stopping=config['training']['early_stopping'],
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],

        # Checkpointing
        checkpoint_dir=config['checkpointing']['checkpoint_dir'],
        save_best_only=config['checkpointing']['save_best_only'],
        save_frequency=config['checkpointing']['save_frequency'],

        # Logging
        log_dir=config['logging']['log_dir'],
        log_frequency=config['logging']['log_frequency'],

        # Mixed precision
        use_amp=config['training']['use_amp'],

        # Validation metric
        validation_metric=config['training']['validation_metric']
    )


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # For full reproducibility (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main(args):
    """Main training function."""
    print("=" * 80)
    print("FairDisCo Training Pipeline")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)

    # Set random seeds
    seed = config['reproducibility']['seed']
    set_random_seeds(seed)
    print(f"Random seed set to {seed}")

    # Create data transforms
    print("\nCreating data transforms...")
    train_transform, val_transform = create_data_transforms(config)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(config, train_transform, val_transform)

    # Create model
    print("\nCreating FairDisCo model...")
    model = create_fairdisco_model(
        num_classes=config['model']['num_classes'],
        num_fst_classes=config['model']['num_fst_classes'],
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        contrastive_dim=config['model']['contrastive_dim'],
        lambda_adv=config['training']['lambda_adv']
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer config
    trainer_config = create_trainer_config(config)

    # Create trainer
    print("\nInitializing FairDisCo trainer...")
    trainer = FairDisCoTrainer(model, trainer_config)

    # Train model
    print("\nStarting training...")
    start_time = datetime.now()

    history = trainer.fit(train_loader, val_loader)

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 3600  # hours

    # Print final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nTraining duration: {training_duration:.2f} hours")
    print(f"Best validation AUROC: {max(history['val_auroc']):.4f}")
    print(f"Final AUROC gap: {history['val_auroc_gap'][-1]:.4f}")
    print(f"Final EOD: {history['val_eod'][-1]:.4f}")
    print(f"Final discriminator accuracy: {history['discriminator_acc'][-1]:.4f}")

    # Print expected vs actual performance
    print("\n" + "-" * 80)
    print("Expected vs Actual Performance:")
    print("-" * 80)
    expected_eod = config['expected_performance']['target_eod']
    actual_eod = history['val_eod'][-1]
    print(f"Expected EOD: {expected_eod:.4f}")
    print(f"Actual EOD:   {actual_eod:.4f}")
    if actual_eod <= expected_eod:
        print("SUCCESS: Target EOD achieved!")
    else:
        print("WARNING: Target EOD not yet achieved. Consider:")
        print("  - Increasing lambda_adv (current: {:.3f})".format(config['training']['lambda_adv']))
        print("  - Training for more epochs")
        print("  - Using larger batch size for contrastive loss")

    print("\nCheckpoint saved to:", trainer_config.checkpoint_dir)
    print("TensorBoard logs:", trainer_config.log_dir)
    print("\nTo view training logs, run:")
    print(f"  tensorboard --logdir {trainer_config.log_dir}")

    print("\n" + "=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train FairDisCo model for fair skin cancer detection'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/fairdisco_config.yaml',
        help='Path to configuration file (default: configs/fairdisco_config.yaml)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
