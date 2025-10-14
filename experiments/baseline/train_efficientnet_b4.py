"""
EfficientNet B4 Baseline Training Script

Trains EfficientNet B4 model on HAM10000/ISIC dataset with comprehensive fairness evaluation.

Usage:
    python experiments/baseline/train_efficientnet_b4.py --config configs/baseline_config.yaml

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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.models.efficientnet_baseline import create_efficientnet_model
from src.training.trainer import Trainer, TrainerConfig
from src.evaluation.fairness_metrics import FairnessMetrics
from src.evaluation.visualizations import FairnessVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dummy_dataloaders(config: dict):
    """Create dummy dataloaders (replace with actual dataset loading)."""
    print("\n" + "="*80)
    print("WARNING: Using dummy data for testing")
    print("Replace with actual dataset loader when data is available")
    print("="*80 + "\n")

    num_train = 1000
    num_val = 200
    num_test = 200
    img_size = config['data']['img_size']
    num_classes = config['model']['num_classes']

    train_images = torch.randn(num_train, 3, img_size, img_size)
    train_labels = torch.randint(0, num_classes, (num_train,))

    val_images = torch.randn(num_val, 3, img_size, img_size)
    val_labels = torch.randint(0, num_classes, (num_val,))

    test_images = torch.randn(num_test, 3, img_size, img_size)
    test_labels = torch.randint(0, num_classes, (num_test,))
    test_fst = torch.randint(1, 7, (num_test,))

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels, test_fst)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


def evaluate_model_on_test(model, test_loader, device, num_classes):
    """Evaluate model on test set with fairness metrics."""
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
                fst_labels = torch.ones(len(labels))

            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())
            all_fst.extend(fst_labels.cpu().numpy())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)
    fst_labels = np.array(all_fst)

    fairness_metrics = FairnessMetrics(num_classes=num_classes)
    results = fairness_metrics.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fst_labels=fst_labels,
        target_class=0
    )

    return {
        'fairness_results': results,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fst_labels': fst_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet B4 baseline model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/baseline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='b4',
        choices=['b3', 'b4'],
        help='EfficientNet variant to use'
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
    print(f"Model: EfficientNet-{args.variant.upper()}")
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Device: {args.device}\n")

    # Create model
    print(f"Creating EfficientNet-{args.variant.upper()} model...")
    model = create_efficientnet_model(
        variant=args.variant,
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone'],
        img_size=config['data']['img_size']
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dummy_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # Training phase
    if not args.test_only:
        trainer_config = TrainerConfig(
            model_name=f'efficientnet_{args.variant}',
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

        trainer = Trainer(model, trainer_config)

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

    test_results = evaluate_model_on_test(
        model=model,
        test_loader=test_loader,
        device=args.device,
        num_classes=config['model']['num_classes']
    )

    fairness_results = test_results['fairness_results']

    print("\n" + fairness_results.summary())

    # Save results
    results_dir = Path('experiments/baseline/results')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f'efficientnet_{args.variant}_results.json'
    with open(results_path, 'w') as f:
        json.dump(fairness_results.to_dict(), f, indent=4)
    print(f"\nResults saved to {results_path}")

    # Generate visualizations
    print("\nGenerating fairness visualizations...")
    visualizer = FairnessVisualizer(save_dir='experiments/baseline/figures')

    model_name = f'efficientnet_{args.variant}'

    visualizer.plot_roc_curves_per_fst(
        y_true=test_results['y_true'],
        y_prob=test_results['y_prob'],
        fst_labels=test_results['fst_labels'],
        target_class=0,
        save_name=f'{model_name}_roc_curves_per_fst.png'
    )

    visualizer.plot_calibration_curves_per_fst(
        y_true=test_results['y_true'],
        y_prob=test_results['y_prob'],
        fst_labels=test_results['fst_labels'],
        save_name=f'{model_name}_calibration_curves_per_fst.png'
    )

    visualizer.plot_confusion_matrices_per_fst(
        y_true=test_results['y_true'],
        y_pred=test_results['y_pred'],
        fst_labels=test_results['fst_labels'],
        save_name=f'{model_name}_confusion_matrices_per_fst.png'
    )

    visualizer.plot_fairness_gaps(
        auroc_per_fst=fairness_results.auroc_per_fst,
        save_name=f'{model_name}_fairness_gaps.png'
    )

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
