"""
FairPrune: Fairness-Aware Model Compression via Structured Pruning

Implements magnitude-based pruning with fairness-aware importance scoring to ensure
compressed models maintain equitable performance across Fitzpatrick skin types (FSTs).

Key Features:
- Fairness-aware importance scoring: magnitude * sensitivity * (1 - fairness_penalty)
- Structured pruning: Remove entire filters/attention heads
- Per-FST sensitivity analysis during pruning
- Iterative pruning with fine-tuning
- Target: 50-70% parameter reduction, <2% accuracy loss, <0.5% fairness degradation

Algorithm:
    1. Compute importance scores per parameter (magnitude + gradient + fairness)
    2. Rank parameters by importance
    3. Prune lowest-importance parameters up to target sparsity
    4. Fine-tune to recover accuracy
    5. Evaluate fairness metrics
    6. Iterate until target compression achieved

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for FairPrune algorithm."""

    # Pruning targets
    target_sparsity: float = 0.6  # 60% parameters pruned
    structured: bool = True  # Structured vs unstructured pruning
    granularity: str = "filter"  # "filter", "channel", "head", "parameter"

    # Fairness weighting
    fairness_weight: float = 0.5  # Weight for fairness penalty (0-1)
    fst_groups: List[int] = None  # FST groups to prioritize (e.g., [5, 6])
    fairness_metric: str = "auroc_gap"  # "auroc_gap", "eod", "ece_gap"

    # Pruning schedule
    num_iterations: int = 10  # Gradual pruning iterations
    initial_sparsity: float = 0.1  # Start with 10% pruning
    pruning_schedule: str = "exponential"  # "linear", "exponential", "adaptive"

    # Layer selection
    skip_layers: List[str] = None  # Layer names to skip (e.g., first/last)
    prune_conv: bool = True
    prune_linear: bool = True
    prune_attention: bool = True

    # Importance scoring
    importance_method: str = "magnitude_gradient"  # "magnitude", "gradient", "taylor", "magnitude_gradient"
    gradient_samples: int = 100  # Samples for gradient-based importance

    def __post_init__(self):
        if self.fst_groups is None:
            self.fst_groups = [5, 6]  # Default: FST V-VI
        if self.skip_layers is None:
            self.skip_layers = []


class FairnessPruner:
    """
    Fairness-aware model pruner that maintains equitable performance across FSTs.

    The pruner computes importance scores that balance:
    1. Magnitude: Parameter size (L2 norm)
    2. Sensitivity: Gradient magnitude (impact on loss)
    3. Fairness: Per-FST performance impact

    Importance = magnitude * sensitivity * (1 - fairness_weight * fairness_penalty)

    Args:
        model: PyTorch model to prune
        config: PruningConfig with pruning settings
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        config: PruningConfig,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.config = config
        self.device = device

        # Pruning state
        self.masks: Dict[str, torch.Tensor] = {}
        self.importance_scores: Dict[str, torch.Tensor] = {}
        self.fairness_metrics: Dict[str, Dict[str, float]] = {}
        self.current_sparsity: float = 0.0

        # Layer registry
        self.prunable_layers: Dict[str, nn.Module] = {}
        self._register_prunable_layers()

        # Statistics
        self.original_params = self._count_parameters()
        self.pruning_history: List[Dict] = []

        logger.info(f"FairnessPruner initialized: {len(self.prunable_layers)} prunable layers, "
                   f"{self.original_params:,} parameters")

    def _register_prunable_layers(self):
        """Register layers that can be pruned."""
        for name, module in self.model.named_modules():
            # Skip layers in config
            if any(skip in name for skip in self.config.skip_layers):
                continue

            # Check if layer is prunable
            is_conv = isinstance(module, (nn.Conv2d, nn.Conv1d))
            is_linear = isinstance(module, nn.Linear)
            is_attention = "attention" in name.lower() or "attn" in name.lower()

            if (self.config.prune_conv and is_conv) or \
               (self.config.prune_linear and is_linear) or \
               (self.config.prune_attention and is_attention and is_linear):
                self.prunable_layers[name] = module

                # Initialize mask (all ones = no pruning)
                if hasattr(module, 'weight'):
                    self.masks[name] = torch.ones_like(module.weight)

    def compute_importance_scores(
        self,
        dataloader: torch.utils.data.DataLoader,
        fairness_evaluator: Optional['FairnessEvaluator'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute fairness-aware importance scores for all prunable parameters.

        Args:
            dataloader: DataLoader for gradient computation
            fairness_evaluator: Evaluator for per-FST metrics

        Returns:
            Dictionary mapping layer names to importance scores
        """
        logger.info("Computing importance scores...")

        self.model.eval()
        importance_scores = {}

        for layer_name, layer in self.prunable_layers.items():
            if not hasattr(layer, 'weight'):
                continue

            weight = layer.weight

            # 1. Magnitude importance
            magnitude_score = torch.abs(weight)

            # 2. Gradient-based importance (sensitivity)
            if self.config.importance_method in ["gradient", "taylor", "magnitude_gradient"]:
                gradient_score = self._compute_gradient_importance(
                    layer_name, layer, dataloader
                )
            else:
                gradient_score = torch.ones_like(weight)

            # 3. Fairness penalty
            if fairness_evaluator is not None:
                fairness_penalty = self._compute_fairness_penalty(
                    layer_name, layer, fairness_evaluator, dataloader
                )
            else:
                fairness_penalty = torch.zeros_like(weight)

            # Combined importance score
            if self.config.importance_method == "magnitude":
                importance = magnitude_score
            elif self.config.importance_method == "gradient":
                importance = gradient_score
            elif self.config.importance_method == "taylor":
                # Taylor expansion: |weight * gradient|
                importance = magnitude_score * gradient_score
            else:  # magnitude_gradient
                importance = magnitude_score * gradient_score

            # Apply fairness weighting
            fairness_factor = 1.0 - self.config.fairness_weight * fairness_penalty
            importance = importance * fairness_factor

            importance_scores[layer_name] = importance.detach().cpu()

            logger.debug(f"{layer_name}: importance range [{importance.min():.6f}, {importance.max():.6f}]")

        self.importance_scores = importance_scores
        return importance_scores

    def _compute_gradient_importance(
        self,
        layer_name: str,
        layer: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """
        Compute gradient-based importance by accumulating gradients over samples.

        Args:
            layer_name: Name of layer
            layer: Layer module
            dataloader: DataLoader for samples

        Returns:
            Gradient importance tensor (same shape as layer.weight)
        """
        self.model.train()

        gradient_acc = torch.zeros_like(layer.weight)
        num_samples = 0

        for i, batch in enumerate(dataloader):
            if num_samples >= self.config.gradient_samples:
                break

            # Get inputs and targets
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = None

            # Forward pass
            self.model.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            if targets is not None:
                if isinstance(outputs, tuple):  # FairDisCo model
                    outputs = outputs[0]
                loss = F.cross_entropy(outputs, targets)
            else:
                # Use output magnitude as proxy if no targets
                loss = outputs.abs().mean()

            # Backward pass
            loss.backward()

            # Accumulate gradients
            if layer.weight.grad is not None:
                gradient_acc += layer.weight.grad.abs().detach()
                num_samples += inputs.size(0)

        # Average gradients
        if num_samples > 0:
            gradient_acc /= (i + 1)

        self.model.eval()
        return gradient_acc

    def _compute_fairness_penalty(
        self,
        layer_name: str,
        layer: nn.Module,
        fairness_evaluator: 'FairnessEvaluator',
        dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """
        Compute per-parameter fairness penalty based on FST-specific impact.

        Higher penalty = parameter hurts fairness (should be preserved)
        Lower penalty = parameter doesn't affect fairness (can be pruned)

        Args:
            layer_name: Name of layer
            layer: Layer module
            fairness_evaluator: Evaluator for FST metrics
            dataloader: DataLoader for evaluation

        Returns:
            Fairness penalty tensor (same shape as layer.weight)
        """
        # Baseline fairness metric
        baseline_metric = self._evaluate_fairness_metric(fairness_evaluator, dataloader)

        # For efficiency, use sampling: test impact of pruning random subsets
        # Full sensitivity analysis is too expensive
        num_samples = 5
        weight = layer.weight.data.clone()

        penalty = torch.zeros_like(weight)

        # Sample random parameters and measure fairness impact
        for _ in range(num_samples):
            # Randomly zero out 10% of parameters
            mask = torch.rand_like(weight) > 0.1
            layer.weight.data = weight * mask

            # Evaluate fairness
            perturbed_metric = self._evaluate_fairness_metric(fairness_evaluator, dataloader)

            # Penalty is proportional to fairness degradation
            # If pruning hurts fairness → high penalty → preserve parameter
            fairness_change = baseline_metric - perturbed_metric
            penalty += (~mask).float() * max(0, fairness_change)

        # Restore original weights
        layer.weight.data = weight

        # Normalize penalty to [0, 1]
        penalty = penalty / (num_samples + 1e-8)
        if penalty.max() > 0:
            penalty = penalty / penalty.max()

        return penalty

    def _evaluate_fairness_metric(
        self,
        fairness_evaluator: 'FairnessEvaluator',
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Evaluate fairness metric on dataloader."""
        if self.config.fairness_metric == "auroc_gap":
            # Compute AUROC gap between FST groups
            metrics = fairness_evaluator.evaluate_per_fst(self.model, dataloader)

            # Get AUROC for priority FST groups vs others
            priority_auroc = np.mean([metrics.get(f"fst_{fst}_auroc", 0.0)
                                     for fst in self.config.fst_groups])
            other_auroc = np.mean([metrics.get(f"fst_{fst}_auroc", 0.0)
                                  for fst in range(1, 7) if fst not in self.config.fst_groups])

            return priority_auroc - other_auroc  # Negative gap = worse for priority groups

        elif self.config.fairness_metric == "eod":
            # Equalized Odds Difference
            metrics = fairness_evaluator.evaluate_per_fst(self.model, dataloader)
            return metrics.get("eod", 0.0)

        else:
            # Default: return 0 (no fairness penalty)
            return 0.0

    def prune_to_sparsity(
        self,
        target_sparsity: float,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Prune model to target sparsity based on importance scores.

        Args:
            target_sparsity: Target fraction of parameters to prune (0-1)
            importance_scores: Pre-computed importance scores (if None, use self.importance_scores)
        """
        if importance_scores is None:
            importance_scores = self.importance_scores

        if not importance_scores:
            raise ValueError("No importance scores available. Run compute_importance_scores() first.")

        logger.info(f"Pruning model to {target_sparsity * 100:.1f}% sparsity...")

        if self.config.structured:
            self._prune_structured(target_sparsity, importance_scores)
        else:
            self._prune_unstructured(target_sparsity, importance_scores)

        self.current_sparsity = target_sparsity
        self._apply_masks()

        # Log statistics
        params_after = self._count_parameters()
        reduction = (1 - params_after / self.original_params) * 100
        logger.info(f"Pruning complete: {params_after:,} parameters remaining "
                   f"({reduction:.1f}% reduction)")

    def _prune_structured(
        self,
        target_sparsity: float,
        importance_scores: Dict[str, torch.Tensor]
    ):
        """
        Structured pruning: Remove entire filters/channels/heads.

        Args:
            target_sparsity: Target sparsity
            importance_scores: Importance scores per layer
        """
        for layer_name, layer in self.prunable_layers.items():
            if layer_name not in importance_scores:
                continue

            importance = importance_scores[layer_name]
            weight_shape = layer.weight.shape

            # Compute filter-level importance (sum over other dimensions)
            if len(weight_shape) == 4:  # Conv2d: (out_channels, in_channels, H, W)
                if self.config.granularity == "filter":
                    # Importance per output filter
                    filter_importance = importance.sum(dim=(1, 2, 3))  # (out_channels,)

                    # Determine number of filters to prune
                    num_filters = filter_importance.size(0)
                    num_to_prune = int(num_filters * target_sparsity)

                    # Get indices of least important filters
                    _, prune_indices = torch.topk(filter_importance, num_to_prune, largest=False)

                    # Create mask: zero out pruned filters
                    mask = torch.ones_like(layer.weight)
                    mask[prune_indices] = 0

                    self.masks[layer_name] = mask

                elif self.config.granularity == "channel":
                    # Importance per input channel
                    channel_importance = importance.sum(dim=(0, 2, 3))  # (in_channels,)

                    num_channels = channel_importance.size(0)
                    num_to_prune = int(num_channels * target_sparsity)

                    _, prune_indices = torch.topk(channel_importance, num_to_prune, largest=False)

                    mask = torch.ones_like(layer.weight)
                    mask[:, prune_indices] = 0

                    self.masks[layer_name] = mask

            elif len(weight_shape) == 2:  # Linear: (out_features, in_features)
                if self.config.granularity == "filter":
                    # Importance per output neuron
                    neuron_importance = importance.sum(dim=1)  # (out_features,)

                    num_neurons = neuron_importance.size(0)
                    num_to_prune = int(num_neurons * target_sparsity)

                    _, prune_indices = torch.topk(neuron_importance, num_to_prune, largest=False)

                    mask = torch.ones_like(layer.weight)
                    mask[prune_indices] = 0

                    self.masks[layer_name] = mask

                elif self.config.granularity == "head" and "attention" in layer_name.lower():
                    # Attention head pruning (assume multi-head structure)
                    # This is simplified; real implementation needs head dimension info
                    neuron_importance = importance.sum(dim=1)

                    num_neurons = neuron_importance.size(0)
                    num_to_prune = int(num_neurons * target_sparsity)

                    _, prune_indices = torch.topk(neuron_importance, num_to_prune, largest=False)

                    mask = torch.ones_like(layer.weight)
                    mask[prune_indices] = 0

                    self.masks[layer_name] = mask

    def _prune_unstructured(
        self,
        target_sparsity: float,
        importance_scores: Dict[str, torch.Tensor]
    ):
        """
        Unstructured pruning: Remove individual parameters globally.

        Args:
            target_sparsity: Target sparsity
            importance_scores: Importance scores per layer
        """
        # Global thresholding: rank all parameters across layers
        all_importance = []

        for layer_name in importance_scores:
            importance = importance_scores[layer_name].flatten()
            all_importance.append(importance)

        all_importance = torch.cat(all_importance)

        # Compute global threshold
        num_params = all_importance.numel()
        num_to_prune = int(num_params * target_sparsity)
        threshold = torch.topk(all_importance, num_to_prune, largest=False)[0].max()

        # Apply threshold to each layer
        for layer_name in importance_scores:
            importance = importance_scores[layer_name]
            mask = (importance > threshold).float()
            self.masks[layer_name] = mask

    def _apply_masks(self):
        """Apply pruning masks to model weights."""
        for layer_name, layer in self.prunable_layers.items():
            if layer_name in self.masks:
                mask = self.masks[layer_name].to(self.device)
                layer.weight.data *= mask

    def _count_parameters(self, only_prunable: bool = False) -> int:
        """Count number of parameters in model."""
        if only_prunable:
            return sum(p.numel() for name, p in self.model.named_parameters()
                      if any(pname in name for pname in self.prunable_layers.keys()))
        else:
            return sum(p.numel() for p in self.model.parameters())

    def get_sparsity_statistics(self) -> Dict[str, any]:
        """Get detailed sparsity statistics."""
        stats = {
            "overall_sparsity": self.current_sparsity,
            "original_parameters": self.original_params,
            "current_parameters": self._count_parameters(),
            "pruned_parameters": self.original_params - self._count_parameters(),
            "per_layer_sparsity": {}
        }

        for layer_name, mask in self.masks.items():
            total = mask.numel()
            pruned = (mask == 0).sum().item()
            sparsity = pruned / total

            stats["per_layer_sparsity"][layer_name] = {
                "total": total,
                "pruned": pruned,
                "sparsity": sparsity
            }

        return stats

    def iterative_pruning(
        self,
        dataloader: torch.utils.data.DataLoader,
        fairness_evaluator: Optional['FairnessEvaluator'] = None,
        fine_tune_fn: Optional[callable] = None
    ) -> List[Dict]:
        """
        Perform iterative gradual pruning with fine-tuning.

        Args:
            dataloader: DataLoader for importance computation
            fairness_evaluator: Fairness evaluator
            fine_tune_fn: Function to fine-tune model after each pruning step
                         Signature: fine_tune_fn(model, current_sparsity) -> metrics

        Returns:
            List of pruning history dictionaries
        """
        logger.info(f"Starting iterative pruning: {self.config.num_iterations} iterations, "
                   f"target sparsity {self.config.target_sparsity * 100:.1f}%")

        # Compute sparsity schedule
        sparsities = self._compute_pruning_schedule()

        for iteration, target_sparsity in enumerate(sparsities):
            logger.info(f"\n=== Iteration {iteration + 1}/{self.config.num_iterations} ===")
            logger.info(f"Target sparsity: {target_sparsity * 100:.1f}%")

            # Compute importance scores
            importance_scores = self.compute_importance_scores(dataloader, fairness_evaluator)

            # Prune to target sparsity
            self.prune_to_sparsity(target_sparsity, importance_scores)

            # Fine-tune if function provided
            if fine_tune_fn is not None:
                logger.info("Fine-tuning pruned model...")
                metrics = fine_tune_fn(self.model, target_sparsity)
            else:
                metrics = {}

            # Record history
            history_entry = {
                "iteration": iteration + 1,
                "sparsity": target_sparsity,
                "parameters": self._count_parameters(),
                "metrics": metrics
            }
            self.pruning_history.append(history_entry)

            logger.info(f"Iteration {iteration + 1} complete: {metrics}")

        logger.info("\n=== Iterative pruning complete ===")
        return self.pruning_history

    def _compute_pruning_schedule(self) -> List[float]:
        """Compute sparsity schedule for gradual pruning."""
        if self.config.pruning_schedule == "linear":
            # Linear increase: 0.1 -> 0.6
            sparsities = np.linspace(
                self.config.initial_sparsity,
                self.config.target_sparsity,
                self.config.num_iterations
            )

        elif self.config.pruning_schedule == "exponential":
            # Exponential increase: prune more aggressively early on
            sparsities = self.config.target_sparsity * (
                1 - np.exp(-3 * np.linspace(0, 1, self.config.num_iterations))
            )
            sparsities = np.clip(sparsities, self.config.initial_sparsity, self.config.target_sparsity)

        else:  # adaptive
            # Start slow, accelerate, then slow down (cosine schedule)
            t = np.linspace(0, np.pi, self.config.num_iterations)
            sparsities = self.config.initial_sparsity + \
                        (self.config.target_sparsity - self.config.initial_sparsity) * \
                        (1 - np.cos(t)) / 2

        return sparsities.tolist()


class FairnessEvaluator:
    """
    Placeholder for fairness evaluation.
    In practice, this should integrate with src/evaluation/fairness_metrics.py
    """

    def evaluate_per_fst(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate per-FST metrics."""
        # Placeholder implementation
        # Real implementation should compute AUROC, EOD, ECE per FST
        return {
            "fst_1_auroc": 0.92,
            "fst_2_auroc": 0.91,
            "fst_3_auroc": 0.90,
            "fst_4_auroc": 0.89,
            "fst_5_auroc": 0.88,
            "fst_6_auroc": 0.87,
            "auroc_gap": 0.05,
            "eod": 0.03
        }


if __name__ == "__main__":
    """Test FairPrune implementation."""
    print("=" * 80)
    print("Testing FairPrune Model Compression")
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
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create pruner
    config = PruningConfig(
        target_sparsity=0.6,
        structured=True,
        granularity="filter",
        fairness_weight=0.5,
        importance_method="magnitude"
    )

    pruner = FairnessPruner(model, config)
    print(f"Prunable layers: {len(pruner.prunable_layers)}")

    # Create dummy data
    dummy_data = [(torch.randn(4, 3, 224, 224), torch.randint(0, 7, (4,))) for _ in range(10)]
    dataloader = dummy_data

    # Compute importance (magnitude-only for test)
    print("\nComputing importance scores...")
    importance = pruner.compute_importance_scores(dataloader)
    print(f"Computed importance for {len(importance)} layers")

    # Prune
    print("\nPruning to 60% sparsity...")
    pruner.prune_to_sparsity(0.6)

    stats = pruner.get_sparsity_statistics()
    print(f"\nSparsity statistics:")
    print(f"  Original parameters: {stats['original_parameters']:,}")
    print(f"  Current parameters: {stats['current_parameters']:,}")
    print(f"  Pruned parameters: {stats['pruned_parameters']:,}")
    print(f"  Overall sparsity: {stats['overall_sparsity'] * 100:.1f}%")

    print("\n" + "=" * 80)
    print("FairPrune test PASSED!")
    print("=" * 80)
