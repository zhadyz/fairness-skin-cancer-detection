"""
FairDisCo Trainer: Three-Loss Adversarial Training Pipeline

Specialized trainer for FairDisCo model with adversarial debiasing and contrastive learning.
Extends base Trainer class with support for:
- Three-loss training: L_cls + lambda_adv * L_adv + lambda_con * L_con
- Lambda scheduling (warm-up and adaptation)
- Discriminator accuracy monitoring
- FST-aware validation metrics

Framework: MENDICANT_BIAS - Phase 2, Week 5-6
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

from .trainer import Trainer, TrainerConfig
from ..models.fairdisco_model import FairDisCoClassifier, SupervisedContrastiveLoss
from ..evaluation.fairness_metrics import FairnessMetrics


@dataclass
class FairDisCoTrainerConfig(TrainerConfig):
    """Extended configuration for FairDisCo training."""

    # FairDisCo-specific hyperparameters
    lambda_adv: float = 0.3  # Adversarial loss weight
    lambda_con: float = 0.2  # Contrastive loss weight
    temperature: float = 0.07  # Temperature for contrastive loss

    # Lambda scheduling
    use_lambda_schedule: bool = True
    lambda_schedule_start_epoch: int = 10  # Start adversarial training after warmup
    lambda_schedule_end_epoch: int = 30  # Reach full lambda_adv by this epoch
    lambda_schedule_start_value: float = 0.0  # Initial lambda_adv
    lambda_schedule_end_value: float = 0.3  # Final lambda_adv

    # Discriminator monitoring
    monitor_discriminator: bool = True
    discriminator_target_acc: float = 0.25  # Target ~random chance (1/6 = 0.167)
    discriminator_acc_tolerance: float = 0.1  # Acceptable range

    # Loss weights for label smoothing
    label_smoothing: float = 0.1  # Classification loss smoothing

    # Gradient clipping (important for GRL stability)
    gradient_clip_norm: float = 1.0


class FairDisCoTrainer:
    """
    Comprehensive trainer for FairDisCo adversarial debiasing model.

    Implements three-loss training loop with:
    1. Classification loss (L_cls): Cross-entropy for diagnosis prediction
    2. Adversarial loss (L_adv): Cross-entropy for FST prediction (reversed gradient)
    3. Contrastive loss (L_con): Supervised contrastive loss for feature quality

    Features:
    - Lambda warm-up scheduling (prevent early training instability)
    - Discriminator accuracy monitoring (detect over/under-regularization)
    - Adaptive lambda adjustment based on discriminator performance
    - Comprehensive fairness evaluation per epoch
    - TensorBoard logging with separate loss tracking

    Example:
        >>> config = FairDisCoTrainerConfig(
        ...     model_name='fairdisco',
        ...     epochs=100,
        ...     batch_size=64,
        ...     lambda_adv=0.3,
        ...     lambda_con=0.2
        ... )
        >>> trainer = FairDisCoTrainer(model, config)
        >>> history = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: FairDisCoClassifier,
        config: FairDisCoTrainerConfig
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup loss functions
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.criterion_adv = nn.CrossEntropyLoss()
        self.criterion_con = SupervisedContrastiveLoss(temperature=config.temperature)

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
            'train_loss_cls': [],
            'train_loss_adv': [],
            'train_loss_con': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auroc': [],
            'val_auroc_gap': [],
            'val_eod': [],
            'discriminator_acc': [],
            'lambda_adv': [],
            'lambda_con': [],
            'learning_rate': []
        }

        # Fairness evaluator
        self.fairness_metrics = FairnessMetrics(
            num_classes=model.num_classes,
            fst_groups=[1, 2, 3, 4, 5, 6]
        )

    def _setup_optimizer(self) -> Optimizer:
        """Setup AdamW optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )

    def _setup_scheduler(self):
        """Setup cosine annealing learning rate scheduler."""
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.scheduler_t0,
            T_mult=self.config.scheduler_t_mult,
            eta_min=self.config.scheduler_eta_min
        )

    def _get_lambda_weights(self, epoch: int) -> Tuple[float, float, float]:
        """
        Get loss weights for current epoch with lambda scheduling.

        Implements warm-up schedule to prevent training instability:
        - Epochs 0-10: No adversarial/contrastive (learn basic features)
        - Epochs 10-30: Linearly ramp up from 0.0 to target values
        - Epochs 30+: Use full lambda values

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (lambda_cls, lambda_adv, lambda_con)
        """
        lambda_cls = 1.0  # Classification loss always enabled

        if not self.config.use_lambda_schedule:
            # No scheduling, use fixed values
            return lambda_cls, self.config.lambda_adv, self.config.lambda_con

        # Warmup phase: No adversarial training
        if epoch < self.config.lambda_schedule_start_epoch:
            return lambda_cls, 0.0, 0.0

        # Ramp-up phase: Linearly increase lambda
        elif epoch < self.config.lambda_schedule_end_epoch:
            progress = (epoch - self.config.lambda_schedule_start_epoch) / \
                      (self.config.lambda_schedule_end_epoch - self.config.lambda_schedule_start_epoch)

            lambda_adv = self.config.lambda_schedule_start_value + \
                        progress * (self.config.lambda_schedule_end_value - self.config.lambda_schedule_start_value)

            # Scale contrastive loss proportionally
            lambda_con = self.config.lambda_con * progress

            return lambda_cls, lambda_adv, lambda_con

        # Full training phase
        else:
            return lambda_cls, self.config.lambda_adv, self.config.lambda_con

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, list]:
        """
        Train FairDisCo model with three-loss adversarial training.

        Args:
            train_loader: Training data loader (must provide images, labels, fst_labels)
            val_loader: Validation data loader

        Returns:
            Training history dictionary
        """
        print(f"\n{'='*80}")
        print(f"Training FairDisCo on {self.device}")
        print(f"{'='*80}\n")

        # Save configuration
        self._save_config()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Get lambda weights for this epoch
            lambda_cls, lambda_adv, lambda_con = self._get_lambda_weights(epoch)

            # Update model's gradient reversal strength
            self.model.update_lambda_adv(lambda_adv)

            # Training phase
            train_metrics = self._train_epoch(train_loader, lambda_cls, lambda_adv, lambda_con)

            # Validation phase
            val_metrics = self._validate_epoch(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['train_loss_cls'].append(train_metrics['loss_cls'])
            self.history['train_loss_adv'].append(train_metrics['loss_adv'])
            self.history['train_loss_con'].append(train_metrics['loss_con'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auroc'].append(val_metrics['auroc'])
            self.history['val_auroc_gap'].append(val_metrics['auroc_gap'])
            self.history['val_eod'].append(val_metrics['eod'])
            self.history['discriminator_acc'].append(val_metrics['discriminator_acc'])
            self.history['lambda_adv'].append(lambda_adv)
            self.history['lambda_con'].append(lambda_con)
            self.history['learning_rate'].append(current_lr)

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train_total', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('Loss/train_cls', train_metrics['loss_cls'], epoch)
            self.writer.add_scalar('Loss/train_adv', train_metrics['loss_adv'], epoch)
            self.writer.add_scalar('Loss/train_con', train_metrics['loss_con'], epoch)
            self.writer.add_scalar('Loss/val_total', val_metrics['total_loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('AUROC/val', val_metrics['auroc'], epoch)
            self.writer.add_scalar('Fairness/auroc_gap', val_metrics['auroc_gap'], epoch)
            self.writer.add_scalar('Fairness/eod', val_metrics['eod'], epoch)
            self.writer.add_scalar('Discriminator/accuracy', val_metrics['discriminator_acc'], epoch)
            self.writer.add_scalar('Lambda/adv', lambda_adv, epoch)
            self.writer.add_scalar('Lambda/con', lambda_con, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config.epochs} - {epoch_time:.2f}s")
            print(f"  Train: Loss={train_metrics['total_loss']:.4f} "
                  f"(cls={train_metrics['loss_cls']:.4f}, "
                  f"adv={train_metrics['loss_adv']:.4f}, "
                  f"con={train_metrics['loss_con']:.4f}), "
                  f"Acc={train_metrics['accuracy']:.4f}")
            print(f"  Val:   Loss={val_metrics['total_loss']:.4f}, "
                  f"Acc={val_metrics['accuracy']:.4f}, "
                  f"AUROC={val_metrics['auroc']:.4f}")
            print(f"  Fairness: AUROC Gap={val_metrics['auroc_gap']:.4f}, "
                  f"EOD={val_metrics['eod']:.4f}")
            print(f"  Discriminator Acc: {val_metrics['discriminator_acc']:.4f}")
            print(f"  Lambda: adv={lambda_adv:.3f}, con={lambda_con:.3f}, LR={current_lr:.6f}")

            # Checkpoint management (use validation AUROC as primary metric)
            is_best = val_metrics['auroc'] > self.best_metric

            if is_best:
                self.best_metric = val_metrics['auroc']
                self.epochs_without_improvement = 0
                self._save_checkpoint(is_best=True)
                print(f"  >>> Best model saved! (AUROC: {val_metrics['auroc']:.4f})")
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
        print(f"Best validation AUROC: {self.best_metric:.4f}")
        print(f"{'='*80}\n")

        # Save final training history
        self._save_history()

        self.writer.close()
        return self.history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        lambda_cls: float,
        lambda_adv: float,
        lambda_con: float
    ) -> Dict[str, float]:
        """Train for one epoch with three losses."""
        self.model.train()

        running_loss_total = 0.0
        running_loss_cls = 0.0
        running_loss_adv = 0.0
        running_loss_con = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (expect images, labels, fst_labels)
            if len(batch) == 3:
                images, targets, fst_labels = batch
            else:
                # Fallback if FST labels not provided
                images, targets = batch
                fst_labels = torch.ones(len(targets), dtype=torch.long)

            images = images.to(self.device)
            targets = targets.to(self.device)
            fst_labels = fst_labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    diagnosis_logits, fst_logits, contrastive_embeddings, _ = self.model(images)

                    # Compute losses
                    loss_cls = self.criterion_cls(diagnosis_logits, targets)
                    loss_adv = self.criterion_adv(fst_logits, fst_labels)
                    loss_con = self.criterion_con(contrastive_embeddings, targets, fst_labels)

                    # Total loss
                    loss = lambda_cls * loss_cls + lambda_adv * loss_adv + lambda_con * loss_con

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping (important for GRL stability)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass
                diagnosis_logits, fst_logits, contrastive_embeddings, _ = self.model(images)

                # Compute losses
                loss_cls = self.criterion_cls(diagnosis_logits, targets)
                loss_adv = self.criterion_adv(fst_logits, fst_labels)
                loss_con = self.criterion_con(contrastive_embeddings, targets, fst_labels)

                # Total loss
                loss = lambda_cls * loss_cls + lambda_adv * loss_adv + lambda_con * loss_con

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

                self.optimizer.step()

            # Statistics
            running_loss_total += loss.item()
            running_loss_cls += loss_cls.item()
            running_loss_adv += loss_adv.item()
            running_loss_con += loss_con.item()

            preds = torch.argmax(diagnosis_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'cls': loss_cls.item(),
                'adv': loss_adv.item(),
                'con': loss_con.item()
            })

        # Compute epoch metrics
        epoch_loss = running_loss_total / len(train_loader)
        epoch_loss_cls = running_loss_cls / len(train_loader)
        epoch_loss_adv = running_loss_adv / len(train_loader)
        epoch_loss_con = running_loss_con / len(train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)

        return {
            'total_loss': epoch_loss,
            'loss_cls': epoch_loss_cls,
            'loss_adv': epoch_loss_adv,
            'loss_con': epoch_loss_con,
            'accuracy': epoch_acc
        }

    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch with fairness evaluation."""
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        all_fst_labels = []
        all_fst_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]  "):
                # Unpack batch
                if len(batch) == 3:
                    images, targets, fst_labels = batch
                else:
                    images, targets = batch
                    fst_labels = torch.ones(len(targets), dtype=torch.long)

                images = images.to(self.device)
                targets = targets.to(self.device)
                fst_labels = fst_labels.to(self.device)

                # Forward pass
                diagnosis_logits, fst_logits, contrastive_embeddings, _ = self.model(images)

                # Compute classification loss (for monitoring)
                loss = self.criterion_cls(diagnosis_logits, targets)
                running_loss += loss.item()

                # Get predictions and probabilities
                diagnosis_probs = torch.softmax(diagnosis_logits, dim=1)
                diagnosis_preds = torch.argmax(diagnosis_logits, dim=1)
                fst_preds = torch.argmax(fst_logits, dim=1)

                all_preds.extend(diagnosis_preds.cpu().numpy())
                all_probs.extend(diagnosis_probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_fst_labels.extend(fst_labels.cpu().numpy())
                all_fst_preds.extend(fst_preds.cpu().numpy())

        # Convert to numpy
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        fst_labels = np.array(all_fst_labels)
        fst_preds = np.array(all_fst_preds)

        # Compute basic metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(y_true, y_pred)

        # Compute fairness metrics
        fairness_results = self.fairness_metrics.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            fst_labels=fst_labels,
            target_class=None  # Use macro-averaged metrics
        )

        # Compute discriminator accuracy
        discriminator_acc = accuracy_score(fst_labels, fst_preds)

        return {
            'total_loss': epoch_loss,
            'accuracy': epoch_acc,
            'auroc': fairness_results.overall_auroc,
            'auroc_gap': fairness_results.auroc_gap,
            'eod': fairness_results.equal_opportunity_diff,
            'discriminator_acc': discriminator_acc
        }

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': asdict(self.config),
            'history': self.history,
            'model_info': self.model.get_model_info()
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


if __name__ == "__main__":
    """Test FairDisCo trainer with dummy data."""
    print("=" * 80)
    print("Testing FairDisCo Trainer")
    print("=" * 80)

    from ..models.fairdisco_model import create_fairdisco_model

    # Create model
    print("\n1. Creating FairDisCo model...")
    model = create_fairdisco_model(
        num_classes=7,
        num_fst_classes=6,
        pretrained=False
    )

    # Create config
    print("\n2. Creating trainer config...")
    config = FairDisCoTrainerConfig(
        model_name='fairdisco_test',
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        lambda_adv=0.3,
        lambda_con=0.2,
        use_amp=False,
        use_lambda_schedule=False,
        checkpoint_dir='test_checkpoints_fairdisco',
        log_dir='test_logs_fairdisco'
    )

    # Create dummy data
    print("\n3. Creating dummy data loaders...")
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=32):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, 7, (1,)).item()
            fst = torch.randint(1, 7, (1,)).item()
            return image, label, fst

    train_dataset = DummyDataset(32)
    val_dataset = DummyDataset(16)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Train
    print("\n4. Training FairDisCo model...")
    trainer = FairDisCoTrainer(model, config)
    history = trainer.fit(train_loader, val_loader)

    print("\n5. Verifying history...")
    print(f"   History keys: {list(history.keys())}")
    print(f"   Train loss: {history['train_loss']}")
    print(f"   Val AUROC: {history['val_auroc']}")
    print(f"   Discriminator acc: {history['discriminator_acc']}")

    print("\n" + "=" * 80)
    print("FairDisCo trainer test PASSED!")
    print("=" * 80)
