"""
Pruning Trainer: Training loop for pruned models with fine-tuning

Implements training strategies for pruned models:
- Gradual magnitude pruning with fine-tuning
- Knowledge distillation from full model
- Fairness metric monitoring during pruning
- Learning rate scheduling for recovery

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
import numpy as np

from .fairprune import FairnessPruner, PruningConfig

logger = logging.getLogger(__name__)


class PruningTrainer:
    """
    Trainer for pruned models with fairness-aware fine-tuning.

    Features:
    - Fine-tuning after each pruning iteration
    - Knowledge distillation from teacher model
    - Fairness-aware loss function
    - Learning rate scheduling
    - Early stopping on fairness degradation

    Args:
        model: Student model to be pruned
        pruner: FairnessPruner instance
        teacher_model: Teacher model for distillation (optional)
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        pruner: FairnessPruner,
        teacher_model: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.pruner = pruner
        self.teacher_model = teacher_model
        self.device = device

        # Move models to device
        self.model = self.model.to(device)
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(device)
            self.teacher_model.eval()

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.best_model_state = None
        self.best_metric = 0.0

        # Loss weights
        self.task_loss_weight = 1.0
        self.distill_loss_weight = 0.5
        self.fairness_loss_weight = 0.3

    def configure_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        """Configure optimizer for fine-tuning."""
        if optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=kwargs.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def configure_scheduler(
        self,
        scheduler_type: str = "cosine",
        num_epochs: int = 10,
        **kwargs
    ):
        """Configure learning rate scheduler."""
        if scheduler_type.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                **kwargs
            )
        elif scheduler_type.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 3),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_type.lower() == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 2)
            )
        else:
            self.scheduler = None

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        fst_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss: task + distillation + fairness.

        Args:
            outputs: Model predictions
            targets: Ground truth labels
            inputs: Input images (for distillation)
            fst_labels: FST labels (for fairness loss)

        Returns:
            Total loss and loss components dictionary
        """
        losses = {}

        # 1. Task loss (classification)
        if isinstance(outputs, tuple):
            # FairDisCo model: (diag_logits, fst_logits, contrastive_emb)
            diag_logits = outputs[0]
        else:
            diag_logits = outputs

        task_loss = F.cross_entropy(diag_logits, targets)
        losses["task_loss"] = task_loss.item()

        total_loss = self.task_loss_weight * task_loss

        # 2. Knowledge distillation loss
        if self.teacher_model is not None and inputs is not None:
            teacher_outputs = self.teacher_model(inputs)
            if isinstance(teacher_outputs, tuple):
                teacher_logits = teacher_outputs[0]
            else:
                teacher_logits = teacher_outputs

            # KL divergence loss
            distill_loss = self._distillation_loss(
                diag_logits, teacher_logits, temperature=3.0
            )
            losses["distill_loss"] = distill_loss.item()

            total_loss = total_loss + self.distill_loss_weight * distill_loss

        # 3. Fairness loss (optional)
        if fst_labels is not None and isinstance(outputs, tuple) and len(outputs) >= 2:
            # For FairDisCo models with FST discriminator
            fst_logits = outputs[1]
            fairness_loss = F.cross_entropy(fst_logits, fst_labels)
            losses["fairness_loss"] = fairness_loss.item()

            total_loss = total_loss + self.fairness_loss_weight * fairness_loss

        losses["total_loss"] = total_loss.item()

        return total_loss, losses

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 3.0
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss using KL divergence.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            temperature: Temperature for softening distributions

        Returns:
            KL divergence loss
        """
        student_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (temperature ** 2)

        return kl_loss

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train model for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            log_interval: Logging frequency

        Returns:
            Dictionary of average metrics
        """
        self.model.train()

        epoch_losses = {
            "task_loss": 0.0,
            "distill_loss": 0.0,
            "fairness_loss": 0.0,
            "total_loss": 0.0
        }

        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Parse batch
            inputs, targets = self._parse_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Get FST labels if available
            fst_labels = batch[2].to(self.device) if len(batch) > 2 else None

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            loss, losses = self.compute_loss(outputs, targets, inputs, fst_labels)

            # Backward pass
            loss.backward()

            # Apply gradient masks (to prevent pruned weights from updating)
            self._apply_gradient_masks()

            # Update weights
            self.optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value
            num_batches += 1

            # Update progress bar
            if batch_idx % log_interval == 0:
                pbar.set_postfix({"loss": f"{losses['total_loss']:.4f}"})

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _apply_gradient_masks(self):
        """Apply pruning masks to gradients to prevent pruned weights from updating."""
        for layer_name, layer in self.pruner.prunable_layers.items():
            if layer_name in self.pruner.masks and layer.weight.grad is not None:
                mask = self.pruner.masks[layer_name].to(self.device)
                layer.weight.grad *= mask

    def evaluate(
        self,
        val_loader: DataLoader,
        compute_fairness: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation data loader
            compute_fairness: Whether to compute per-FST metrics

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_fst_labels = []

        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                inputs, targets = self._parse_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                fst_labels = batch[2] if len(batch) > 2 else None

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss, _ = self.compute_loss(outputs, targets, inputs, fst_labels)
                val_loss += loss.item()
                num_batches += 1

                # Get predictions
                if isinstance(outputs, tuple):
                    diag_logits = outputs[0]
                else:
                    diag_logits = outputs

                predictions = diag_logits.argmax(dim=1)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

                if fst_labels is not None:
                    all_fst_labels.append(fst_labels.cpu())

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Compute metrics
        metrics = {
            "val_loss": val_loss / num_batches,
            "val_accuracy": (all_predictions == all_targets).float().mean().item()
        }

        # Compute per-FST metrics if requested
        if compute_fairness and len(all_fst_labels) > 0:
            all_fst_labels = torch.cat(all_fst_labels)
            fst_metrics = self._compute_fst_metrics(
                all_predictions, all_targets, all_fst_labels
            )
            metrics.update(fst_metrics)

        return metrics

    def _compute_fst_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        fst_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute per-FST accuracy and fairness gaps."""
        metrics = {}

        # Per-FST accuracy
        fst_accuracies = []
        for fst in range(1, 7):
            mask = fst_labels == fst
            if mask.sum() > 0:
                fst_acc = (predictions[mask] == targets[mask]).float().mean().item()
                metrics[f"fst_{fst}_accuracy"] = fst_acc
                fst_accuracies.append(fst_acc)

        # Accuracy gap
        if len(fst_accuracies) > 0:
            metrics["accuracy_gap"] = max(fst_accuracies) - min(fst_accuracies)
            metrics["avg_accuracy"] = np.mean(fst_accuracies)

        return metrics

    def _parse_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse batch into inputs and targets."""
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
            targets = batch[1]
        else:
            inputs = batch
            targets = None

        return inputs, targets

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
        early_stopping_patience: int = 3,
        fairness_threshold: float = 0.05
    ) -> Dict[str, any]:
        """
        Fine-tune pruned model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of fine-tuning epochs
            early_stopping_patience: Patience for early stopping
            fairness_threshold: Maximum allowed fairness degradation

        Returns:
            Fine-tuning history and best metrics
        """
        logger.info(f"Fine-tuning pruned model for {num_epochs} epochs...")

        if self.optimizer is None:
            self.configure_optimizer()

        if self.scheduler is None:
            self.configure_scheduler(num_epochs=num_epochs)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "accuracy_gap": []
        }

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_accuracy"])
                else:
                    self.scheduler.step()

            # Log metrics
            logger.info(f"Epoch {epoch}/{num_epochs}:")
            logger.info(f"  Train loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val accuracy: {val_metrics['val_accuracy']:.4f}")

            if "accuracy_gap" in val_metrics:
                logger.info(f"  Accuracy gap: {val_metrics['accuracy_gap']:.4f}")

            # Record history
            history["train_loss"].append(train_metrics["total_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_accuracy"].append(val_metrics["val_accuracy"])

            if "accuracy_gap" in val_metrics:
                history["accuracy_gap"].append(val_metrics["accuracy_gap"])

                # Early stopping on fairness degradation
                if val_metrics["accuracy_gap"] > fairness_threshold:
                    logger.warning(f"Fairness degradation detected: gap = {val_metrics['accuracy_gap']:.4f}")
                    patience_counter += 1
                else:
                    patience_counter = 0

            # Early stopping on validation loss
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.best_model_state = self.model.state_dict().copy()
                self.best_metric = val_metrics["val_accuracy"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        logger.info(f"Fine-tuning complete. Best accuracy: {self.best_metric:.4f}")

        return {
            "history": history,
            "best_accuracy": self.best_metric,
            "final_metrics": val_metrics
        }

    def prune_and_fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_sparsity: float,
        fine_tune_epochs: int = 5
    ) -> Dict[str, any]:
        """
        Perform one pruning + fine-tuning iteration.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            target_sparsity: Target sparsity for this iteration
            fine_tune_epochs: Number of fine-tuning epochs

        Returns:
            Metrics after pruning and fine-tuning
        """
        logger.info(f"\n=== Prune and Fine-Tune: Target Sparsity {target_sparsity * 100:.1f}% ===")

        # Compute importance scores
        logger.info("Computing importance scores...")
        importance_scores = self.pruner.compute_importance_scores(train_loader)

        # Prune model
        logger.info(f"Pruning to {target_sparsity * 100:.1f}% sparsity...")
        self.pruner.prune_to_sparsity(target_sparsity, importance_scores)

        # Fine-tune
        logger.info("Fine-tuning pruned model...")
        fine_tune_results = self.fine_tune(
            train_loader,
            val_loader,
            num_epochs=fine_tune_epochs
        )

        # Get sparsity statistics
        sparsity_stats = self.pruner.get_sparsity_statistics()

        return {
            "sparsity": target_sparsity,
            "sparsity_stats": sparsity_stats,
            "fine_tune_results": fine_tune_results,
            "final_accuracy": fine_tune_results["best_accuracy"]
        }


if __name__ == "__main__":
    """Test pruning trainer."""
    print("=" * 80)
    print("Testing Pruning Trainer")
    print("=" * 80)

    # Create dummy model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 7 * 7, 256)
            self.fc2 = nn.Linear(256, 7)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 4)
            x = x.flatten(1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleModel()

    # Create pruner
    config = PruningConfig(
        target_sparsity=0.5,
        structured=True,
        num_iterations=3
    )
    pruner = FairnessPruner(model, config)

    # Create trainer
    trainer = PruningTrainer(model, pruner)
    trainer.configure_optimizer(lr=1e-3)

    # Create dummy data
    train_data = [(torch.randn(4, 3, 224, 224), torch.randint(0, 7, (4,))) for _ in range(20)]
    val_data = [(torch.randn(4, 3, 224, 224), torch.randint(0, 7, (4,))) for _ in range(10)]

    print("\nTraining one epoch...")
    train_metrics = trainer.train_epoch(train_data, epoch=1, log_interval=5)
    print(f"Train metrics: {train_metrics}")

    print("\nEvaluating...")
    val_metrics = trainer.evaluate(val_data, compute_fairness=False)
    print(f"Val metrics: {val_metrics}")

    print("\n" + "=" * 80)
    print("Pruning trainer test PASSED!")
    print("=" * 80)
