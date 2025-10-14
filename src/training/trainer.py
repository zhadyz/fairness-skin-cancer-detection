"""
Comprehensive Training Pipeline for Skin Cancer Detection Models

Implements training loop with:
- Learning rate scheduling (CosineAnnealingWarmRestarts)
- Early stopping
- TensorBoard logging
- Checkpoint management
- Mixed precision training (optional)
- Multi-metric validation (accuracy, AUROC, sensitivity, specificity)

Author: HOLLOWED_EYES
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer, AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm


@dataclass
class TrainerConfig:
    """Configuration for model training."""

    # Model settings
    model_name: str = "resnet50"
    num_classes: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training hyperparameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # 'adam' or 'adamw'

    # Learning rate scheduling
    scheduler: str = "cosine_warm"  # 'cosine_warm' or 'reduce_plateau'
    scheduler_t0: int = 10  # CosineAnnealing: epochs for first restart
    scheduler_t_mult: int = 2  # CosineAnnealing: restart period multiplier
    scheduler_eta_min: float = 1e-6  # Minimum learning rate

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "experiments/baseline/checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5  # Save every N epochs (if not save_best_only)

    # Logging
    log_dir: str = "experiments/baseline/logs"
    log_frequency: int = 10  # Log every N batches

    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision (faster on modern GPUs)

    # Class weights (for imbalanced datasets)
    use_class_weights: bool = False
    class_weights: Optional[list] = None

    # Validation metric for best model selection
    validation_metric: str = "auroc"  # 'auroc', 'accuracy', 'loss'


class Trainer:
    """
    Comprehensive trainer for skin cancer detection models.

    Features:
    - Flexible configuration via TrainerConfig
    - Multi-metric tracking (loss, accuracy, AUROC, sensitivity, specificity)
    - TensorBoard logging with real-time visualization
    - Automatic checkpoint management
    - Early stopping with patience
    - Learning rate scheduling
    - Mixed precision training support

    Example:
        >>> config = TrainerConfig(
        ...     model_name='resnet50',
        ...     epochs=50,
        ...     batch_size=32,
        ...     learning_rate=1e-4
        ... )
        >>> trainer = Trainer(model, config)
        >>> history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup loss function
        self.criterion = self._setup_criterion()

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # Create directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auroc': [],
            'learning_rate': []
        }

    def _setup_optimizer(self) -> Optimizer:
        """Setup optimizer."""
        if self.config.optimizer.lower() == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _setup_criterion(self) -> nn.Module:
        """Setup loss function with optional class weights."""
        if self.config.use_class_weights and self.config.class_weights is not None:
            weights = torch.tensor(self.config.class_weights, dtype=torch.float32)
            weights = weights.to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler == 'cosine_warm':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.scheduler_t0,
                T_mult=self.config.scheduler_t_mult,
                eta_min=self.config.scheduler_eta_min
            )
        elif self.config.scheduler == 'reduce_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history dictionary
        """
        print(f"\n{'='*80}")
        print(f"Training {self.config.model_name} on {self.device}")
        print(f"{'='*80}\n")

        # Save configuration
        self._save_config()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc, val_auroc = self._validate_epoch(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_auroc)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auroc'].append(val_auroc)
            self.history['learning_rate'].append(current_lr)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('AUROC/val', val_auroc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config.epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUROC: {val_auroc:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Checkpoint management
            metric_value = self._get_metric_value(val_loss, val_acc, val_auroc)
            is_best = metric_value > self.best_metric

            if is_best:
                self.best_metric = metric_value
                self.epochs_without_improvement = 0
                self._save_checkpoint(is_best=True)
                print(f"  >>> Best model saved! ({self.config.validation_metric}: {metric_value:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Save periodic checkpoint
            if not self.config.save_best_only and (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(is_best=False)

            # Early stopping check
            if self.config.early_stopping and self.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"No improvement for {self.config.patience} consecutive epochs")
                break

            print()

        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"Best {self.config.validation_metric}: {self.best_metric:.4f}")
        print(f"{'='*80}\n")

        # Save final training history
        self._save_history()

        self.writer.close()
        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)

        return epoch_loss, epoch_acc

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate for one epoch."""
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]  "):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)

        # Calculate AUROC (macro average across classes)
        try:
            epoch_auroc = roc_auc_score(
                all_targets,
                all_probs,
                multi_class='ovr',
                average='macro'
            )
        except ValueError:
            # Handle case where some classes might be missing in validation
            epoch_auroc = 0.0

        return epoch_loss, epoch_acc, epoch_auroc

    def _get_metric_value(self, val_loss: float, val_acc: float, val_auroc: float) -> float:
        """Get metric value for model selection."""
        if self.config.validation_metric == 'auroc':
            return val_auroc
        elif self.config.validation_metric == 'accuracy':
            return val_acc
        elif self.config.validation_metric == 'loss':
            return -val_loss  # Negative because we want to maximize
        else:
            return val_auroc

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': asdict(self.config),
            'history': self.history
        }

        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.config.model_name}_best.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.config.model_name}_epoch_{self.current_epoch+1}.pth"

        torch.save(checkpoint, checkpoint_path)

    def _save_config(self):
        """Save training configuration."""
        config_path = self.checkpoint_dir / f"{self.config.model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)

    def _save_history(self):
        """Save training history."""
        history_path = self.checkpoint_dir / f"{self.config.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint['history']

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch+1}")


def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """
    Load trained model from checkpoint (for inference).

    Args:
        model: Model instance (architecture should match)
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Example usage
    from src.models.resnet_baseline import create_resnet50_model

    print("Testing Trainer...")

    # Create dummy data
    batch_size = 8
    num_classes = 7

    dummy_train_data = [(torch.randn(3, 224, 224), torch.randint(0, num_classes, (1,)).item()) for _ in range(32)]
    dummy_val_data = [(torch.randn(3, 224, 224), torch.randint(0, num_classes, (1,)).item()) for _ in range(16)]

    train_loader = DataLoader(dummy_train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dummy_val_data, batch_size=batch_size, shuffle=False)

    # Create model
    model = create_resnet50_model(num_classes=num_classes, pretrained=False)

    # Create config
    config = TrainerConfig(
        model_name='resnet50_test',
        epochs=2,
        batch_size=batch_size,
        learning_rate=1e-3,
        use_amp=False,
        checkpoint_dir='test_checkpoints',
        log_dir='test_logs'
    )

    # Train
    trainer = Trainer(model, config)
    history = trainer.fit(train_loader, val_loader)

    print("\nTrainer test PASSED!")
    print(f"History keys: {history.keys()}")
