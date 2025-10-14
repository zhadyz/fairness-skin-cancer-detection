"""
ResNet50 Baseline Training Script

Trains ResNet50 model on HAM10000/ISIC dataset with comprehensive fairness evaluation.

Usage:
    python experiments/baseline/train_resnet50.py --config configs/baseline_config.yaml

Author: HOLLOWED_EYES
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models.resnet_baseline import create_resnet50_model
from src.training.trainer import Trainer, TrainerConfig
from src.evaluation.fairness_metrics import FairnessMetrics, evaluate_fairness
from src.evaluation.visualizations import FairnessVisualizer
from src.data.ham10000_dataset import HAM10000Dataset, load_splits
from src.data.preprocessing import get_training_augmentation, get_validation_transform


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """Custom collate function to handle HAM10000Dataset dictionary output."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    fst_labels = torch.tensor([item['fst'] for item in batch])
    return images, labels, fst_labels


def create_ham10000_dataloaders(config: dict):
    """
    Create HAM10000 dataloaders for training, validation, and testing.

    Uses real HAM10000 dataset with FST annotations.
    """
    data_config = config['data']

    # Check if HAM10000 data exists
    data_dir = Path(data_config.get('data_dir', 'data/raw/ham10000'))
    metadata_path = Path(data_config.get('metadata_path', 'data/metadata/ham10000_fst_estimated.csv'))
    splits_path = Path(data_config.get('splits_path', 'data/metadata/ham10000_splits.json'))

    if not data_dir.exists():
        print("\n" + "="*80)
        print("ERROR: HAM10000 data directory not found!")
        print(f"Expected location: {data_dir.absolute()}")
        print("\nTo download HAM10000 dataset:")
        print("  1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("  2. Download HAM10000_images_part_1.zip")
        print("  3. Download HAM10000_images_part_2.zip")
        print("  4. Download HAM10000_metadata")
        print(f"  5. Extract to: {data_dir.absolute()}")
        print("\nFalling back to dummy data for testing...")
        print("="*80 + "\n")
        return create_dummy_dataloaders(config)

    # Load splits
    if not splits_path.exists():
        print(f"\nWARNING: Splits file not found at {splits_path}")
        print("Please generate splits first:")
        print("  python scripts/generate_ham10000_fst.py")
        print("  python scripts/create_ham10000_splits.py")
        print("\nFalling back to dummy data...")
        return create_dummy_dataloaders(config)

    print(f"\nLoading HAM10000 dataset from {data_dir}")
    splits = load_splits(str(splits_path))

    # Create transforms
    img_size = data_config['img_size']
    train_transform = get_training_augmentation(
        image_size=img_size,
        augmentation_strength=data_config.get('augmentation_strength', 'medium')
    )
    val_transform = get_validation_transform(image_size=img_size)

    # Create datasets
    train_dataset = HAM10000Dataset(
        root_dir=str(data_dir),
        metadata_path=str(metadata_path),
        split="train",
        split_indices=splits['train'],
        transform=train_transform,
        use_fst_annotations=True,
        estimate_fst_if_missing=True,
    )

    val_dataset = HAM10000Dataset(
        root_dir=str(data_dir),
        metadata_path=str(metadata_path),
        split="val",
        split_indices=splits['val'],
        transform=val_transform,
        use_fst_annotations=True,
        estimate_fst_if_missing=True,
    )

    test_dataset = HAM10000Dataset(
        root_dir=str(data_dir),
        metadata_path=str(metadata_path),
        split="test",
        split_indices=splits['test'],
        transform=val_transform,
        use_fst_annotations=True,
        estimate_fst_if_missing=True,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=data_config['shuffle_train'],
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn,
    )

    print(f"\nDataset loaded successfully!")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def create_dummy_dataloaders(config: dict):
    """
    Create dummy dataloaders for testing (fallback when HAM10000 not available).
    """
    from torch.utils.data import TensorDataset

    print("\n" + "="*80)
    print("WARNING: Using dummy data for testing")
    print("Real HAM10000 dataset not found - using synthetic data")
    print("="*80 + "\n")

    # Dummy data parameters
    num_train = 1000
    num_val = 200
    num_test = 200
    img_size = config['data']['img_size']
    num_classes = config['model']['num_classes']

    # Generate dummy images and labels
    train_images = torch.randn(num_train, 3, img_size, img_size)
    train_labels = torch.randint(0, num_classes, (num_train,))
    train_fst = torch.randint(1, 7, (num_train,))

    val_images = torch.randn(num_val, 3, img_size, img_size)
    val_labels = torch.randint(0, num_classes, (num_val,))
    val_fst = torch.randint(1, 7, (num_val,))

    test_images = torch.randn(num_test, 3, img_size, img_size)
    test_labels = torch.randint(0, num_classes, (num_test,))
    test_fst = torch.randint(1, 7, (num_test,))

    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels, train_fst)
    val_dataset = TensorDataset(val_images, val_labels, val_fst)
    test_dataset = TensorDataset(test_images, test_labels, test_fst)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=config['data']['shuffle_train'],
        num_workers=0,  # Set to 0 for dummy data
        pin_memory=config['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=config['data']['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader, test_loader


def evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    num_classes: int
) -> dict:
    """
    Evaluate model on test set with fairness metrics.

    Args:
        model: Trained model
        test_loader: Test data loader (with FST labels)
        device: Device to run evaluation on
        num_classes: Number of classes

    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.to(device)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_fst = []

    print("\nEvaluating on test set...")

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, fst_labels = batch
            else:
                images, labels = batch
                fst_labels = torch.ones(len(labels))  # Placeholder

            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())
            all_fst.extend(fst_labels.cpu().numpy())

    # Convert to numpy
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)
    fst_labels = np.array(all_fst)

    # Compute fairness metrics
    fairness_metrics = FairnessMetrics(num_classes=num_classes)
    results = fairness_metrics.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fst_labels=fst_labels,
        target_class=0  # Melanoma class (adjust based on dataset)
    )

    return {
        'fairness_results': results,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fst_labels': fst_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 baseline model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/baseline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run evaluation on test set'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from {args.config}")
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Device: {args.device}\n")

    # Create model
    print("Creating ResNet50 model...")
    model = create_resnet50_model(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        dropout=config['model']['dropout']
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_ham10000_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # Training phase
    if not args.test_only:
        # Create trainer configuration
        trainer_config = TrainerConfig(
            model_name='resnet50',
            num_classes=config['model']['num_classes'],
            device=args.device,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            optimizer=config['training']['optimizer'],
            scheduler=config['training']['scheduler'],
            scheduler_t0=config['training']['scheduler_t0'],
            scheduler_t_mult=config['training']['scheduler_t_mult'],
            scheduler_eta_min=config['training']['scheduler_eta_min'],
            early_stopping=config['training']['early_stopping'],
            patience=config['training']['patience'],
            use_amp=config['training']['use_amp'],
            checkpoint_dir=config['checkpointing']['checkpoint_dir'],
            log_dir=config['logging']['log_dir'],
            validation_metric=config['training']['validation_metric']
        )

        # Create trainer
        trainer = Trainer(model, trainer_config)

        # Train model
        print("Starting training...")
        history = trainer.fit(train_loader, val_loader)

        print("\nTraining completed!")
        print(f"Best {trainer_config.validation_metric}: {trainer.best_metric:.4f}")

        # Save training history visualization
        visualizer = FairnessVisualizer(save_dir='experiments/baseline/figures')
        visualizer.plot_training_history(history)

    # Evaluation phase
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")

    # Evaluate on test set
    test_results = evaluate_model_on_test(
        model=model,
        test_loader=test_loader,
        device=args.device,
        num_classes=config['model']['num_classes']
    )

    fairness_results = test_results['fairness_results']

    # Print fairness summary
    print("\n" + fairness_results.summary())

    # Save results
    results_dir = Path('experiments/baseline/results')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'resnet50_results.json'
    with open(results_path, 'w') as f:
        json.dump(fairness_results.to_dict(), f, indent=4)
    print(f"\nResults saved to {results_path}")

    # Generate visualizations
    print("\nGenerating fairness visualizations...")
    visualizer = FairnessVisualizer(save_dir='experiments/baseline/figures')

    visualizer.plot_roc_curves_per_fst(
        y_true=test_results['y_true'],
        y_prob=test_results['y_prob'],
        fst_labels=test_results['fst_labels'],
        target_class=0,
        save_name='resnet50_roc_curves_per_fst.png'
    )

    visualizer.plot_calibration_curves_per_fst(
        y_true=test_results['y_true'],
        y_prob=test_results['y_prob'],
        fst_labels=test_results['fst_labels'],
        save_name='resnet50_calibration_curves_per_fst.png'
    )

    visualizer.plot_confusion_matrices_per_fst(
        y_true=test_results['y_true'],
        y_pred=test_results['y_pred'],
        fst_labels=test_results['fst_labels'],
        save_name='resnet50_confusion_matrices_per_fst.png'
    )

    visualizer.plot_fairness_gaps(
        auroc_per_fst=fairness_results.auroc_per_fst,
        save_name='resnet50_fairness_gaps.png'
    )

    print("\nExperiment complete!")
    print(f"Checkpoints saved to: {config['checkpointing']['checkpoint_dir']}")
    print(f"Logs saved to: {config['logging']['log_dir']}")
    print(f"Figures saved to: experiments/baseline/figures/")


if __name__ == "__main__":
    main()
