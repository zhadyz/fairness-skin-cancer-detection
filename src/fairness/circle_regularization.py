"""
CIRCLe Regularization Loss: Tone-Invariant Embedding Regularization

Implements the CIRCLe regularization loss that encourages latent embeddings to be
invariant to skin tone transformations. By minimizing the distance between embeddings
of original and color-transformed images, the model learns to focus on diagnostic
features rather than spurious skin tone correlations.

Based on: Pakzad et al. (2022) "CIRCLe: Color Invariant Representation Learning
for Unbiased Classification of Skin Lesions" ECCV 2022 Workshops

Clean-room implementation from research documentation only.

Loss Formula:
    L_reg = 1/N * sum_i ||f(x_i) - f(T(x_i))||^2

Where:
    - x_i: Original image
    - T(x_i): Color-transformed image (e.g., FST III → FST I or VI)
    - f(·): Feature extractor (produces embeddings)
    - N: Batch size

Framework: MENDICANT_BIAS - Phase 2, Week 7-8
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class CIRCLe_RegularizationLoss(nn.Module):
    """
    CIRCLe Tone-Invariant Regularization Loss.

    Computes distance between embeddings of original and color-transformed images
    to enforce skin tone invariance in the feature space.

    The loss can use different distance metrics:
    - L2 (Euclidean): Standard squared L2 distance
    - Cosine: 1 - cosine_similarity (angular distance)
    - L1 (Manhattan): Sum of absolute differences

    Args:
        distance_metric: Type of distance ("l2", "cosine", "l1")
        normalize_embeddings: Whether to L2-normalize embeddings before distance
        reduction: How to reduce batch dimension ("mean", "sum")
        temperature: Temperature for cosine distance (default 1.0)

    Example:
        >>> loss_fn = CIRCLe_RegularizationLoss(distance_metric="l2")
        >>> emb_orig = torch.randn(32, 2048)  # Original image embeddings
        >>> emb_trans = torch.randn(32, 2048)  # Transformed image embeddings
        >>> loss = loss_fn(emb_orig, emb_trans)
        >>> print(loss.item())  # Scalar loss value
    """

    def __init__(
        self,
        distance_metric: Literal["l2", "cosine", "l1"] = "l2",
        normalize_embeddings: bool = False,
        reduction: Literal["mean", "sum"] = "mean",
        temperature: float = 1.0
    ):
        super(CIRCLe_RegularizationLoss, self).__init__()
        self.distance_metric = distance_metric
        self.normalize_embeddings = normalize_embeddings
        self.reduction = reduction
        self.temperature = temperature

        # Validate parameters
        if distance_metric not in ["l2", "cosine", "l1"]:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(
        self,
        embeddings_original: torch.Tensor,
        embeddings_transformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CIRCLe regularization loss.

        Args:
            embeddings_original: Embeddings from original images (batch_size, feature_dim)
            embeddings_transformed: Embeddings from transformed images (batch_size, feature_dim)

        Returns:
            Scalar loss value (minimized when embeddings are similar)
        """
        # Validate shapes
        if embeddings_original.shape != embeddings_transformed.shape:
            raise ValueError(
                f"Embedding shapes must match: "
                f"{embeddings_original.shape} vs {embeddings_transformed.shape}"
            )

        # Optional normalization
        if self.normalize_embeddings:
            embeddings_original = F.normalize(embeddings_original, p=2, dim=1)
            embeddings_transformed = F.normalize(embeddings_transformed, p=2, dim=1)

        # Compute distance based on metric
        if self.distance_metric == "l2":
            # Squared L2 distance
            distances = torch.sum((embeddings_original - embeddings_transformed) ** 2, dim=1)
        elif self.distance_metric == "cosine":
            # Cosine distance (1 - cosine_similarity)
            cosine_sim = F.cosine_similarity(embeddings_original, embeddings_transformed, dim=1)
            distances = (1 - cosine_sim) / self.temperature
        elif self.distance_metric == "l1":
            # L1 (Manhattan) distance
            distances = torch.sum(torch.abs(embeddings_original - embeddings_transformed), dim=1)

        # Reduce over batch
        if self.reduction == "mean":
            loss = torch.mean(distances)
        else:  # sum
            loss = torch.sum(distances)

        return loss

    def get_pairwise_distances(
        self,
        embeddings_original: torch.Tensor,
        embeddings_transformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample distances (useful for analysis).

        Args:
            embeddings_original: Embeddings from original images (B, D)
            embeddings_transformed: Embeddings from transformed images (B, D)

        Returns:
            Per-sample distances (B,)
        """
        if self.normalize_embeddings:
            embeddings_original = F.normalize(embeddings_original, p=2, dim=1)
            embeddings_transformed = F.normalize(embeddings_transformed, p=2, dim=1)

        if self.distance_metric == "l2":
            distances = torch.sum((embeddings_original - embeddings_transformed) ** 2, dim=1)
        elif self.distance_metric == "cosine":
            cosine_sim = F.cosine_similarity(embeddings_original, embeddings_transformed, dim=1)
            distances = (1 - cosine_sim) / self.temperature
        elif self.distance_metric == "l1":
            distances = torch.sum(torch.abs(embeddings_original - embeddings_transformed), dim=1)

        return distances


class MultiTargetCIRCLe_Loss(nn.Module):
    """
    Multi-Target CIRCLe Regularization Loss.

    Regularizes embeddings against MULTIPLE color transformations (not just one).
    For example, transform FST III → FST I and FST VI, then average regularization loss.

    This provides more robust tone-invariance across a wider range of skin tones.

    Formula:
        L_reg = 1/(N*K) * sum_i sum_k ||f(x_i) - f(T_k(x_i))||^2

    Where K is the number of target FST transformations.

    Args:
        target_fsts: List of target FST classes (e.g., [1, 6] for extreme tones)
        distance_metric: Distance metric for regularization
        normalize_embeddings: Whether to normalize embeddings
        reduction: Batch reduction method

    Example:
        >>> loss_fn = MultiTargetCIRCLe_Loss(target_fsts=[1, 6])
        >>> emb_orig = torch.randn(32, 2048)
        >>> emb_trans_dict = {
        ...     1: torch.randn(32, 2048),  # FST I transformations
        ...     6: torch.randn(32, 2048)   # FST VI transformations
        ... }
        >>> loss = loss_fn(emb_orig, emb_trans_dict)
    """

    def __init__(
        self,
        target_fsts: list = [1, 6],
        distance_metric: Literal["l2", "cosine", "l1"] = "l2",
        normalize_embeddings: bool = False,
        reduction: Literal["mean", "sum"] = "mean"
    ):
        super(MultiTargetCIRCLe_Loss, self).__init__()
        self.target_fsts = target_fsts
        self.num_targets = len(target_fsts)

        # Create base regularization loss
        self.base_loss = CIRCLe_RegularizationLoss(
            distance_metric=distance_metric,
            normalize_embeddings=normalize_embeddings,
            reduction=reduction
        )

    def forward(
        self,
        embeddings_original: torch.Tensor,
        embeddings_transformed_dict: dict
    ) -> torch.Tensor:
        """
        Compute multi-target CIRCLe regularization loss.

        Args:
            embeddings_original: Embeddings from original images (B, D)
            embeddings_transformed_dict: Dict mapping target_fst → transformed embeddings
                Example: {1: emb_fst1, 6: emb_fst6}

        Returns:
            Scalar loss (averaged over all target FSTs)
        """
        total_loss = 0.0

        for target_fst in self.target_fsts:
            if target_fst not in embeddings_transformed_dict:
                raise ValueError(f"Missing embeddings for target FST {target_fst}")

            embeddings_transformed = embeddings_transformed_dict[target_fst]
            loss_fst = self.base_loss(embeddings_original, embeddings_transformed)
            total_loss += loss_fst

        # Average over target FSTs
        return total_loss / self.num_targets


class ToneInvarianceMetric(nn.Module):
    """
    Metric to measure tone-invariance of learned embeddings.

    Computes the average distance between embeddings of same-diagnosis images
    across different FST groups. Lower values indicate better tone-invariance.

    This is useful for monitoring training progress and validating that CIRCLe
    regularization is working.

    Args:
        distance_metric: Distance metric for measurement
        normalize_embeddings: Whether to normalize embeddings

    Example:
        >>> metric = ToneInvarianceMetric()
        >>> embeddings = torch.randn(100, 2048)
        >>> labels = torch.randint(0, 7, (100,))
        >>> fst_labels = torch.randint(1, 7, (100,))
        >>> score = metric(embeddings, labels, fst_labels)
        >>> print(f"Tone-invariance score: {score:.4f}")
    """

    def __init__(
        self,
        distance_metric: Literal["l2", "cosine", "l1"] = "l2",
        normalize_embeddings: bool = True
    ):
        super(ToneInvarianceMetric, self).__init__()
        self.distance_metric = distance_metric
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        fst_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute tone-invariance metric.

        For each pair of same-diagnosis, different-FST samples, compute embedding
        distance and return the average.

        Args:
            embeddings: Feature embeddings (N, D)
            labels: Diagnosis labels (N,)
            fst_labels: FST labels (N,)

        Returns:
            Scalar tone-invariance score (lower is better)
        """
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        N = embeddings.size(0)
        device = embeddings.device

        # Create masks
        labels = labels.view(-1, 1)
        fst_labels = fst_labels.view(-1, 1)

        # Same diagnosis mask
        label_mask = torch.eq(labels, labels.T).float().to(device)

        # Different FST mask
        fst_mask = ~torch.eq(fst_labels, fst_labels.T)
        fst_mask = fst_mask.float().to(device)

        # Positive pairs: same diagnosis AND different FST
        positive_mask = label_mask * fst_mask

        # Remove self-similarity
        self_mask = torch.eye(N, dtype=torch.float32, device=device)
        positive_mask = positive_mask * (1 - self_mask)

        # Count positive pairs
        num_positives = positive_mask.sum()

        if num_positives == 0:
            # No positive pairs (batch has single FST)
            return torch.tensor(0.0, device=device)

        # Compute pairwise distances
        if self.distance_metric == "l2":
            # Squared L2 distance matrix
            distances = torch.cdist(embeddings, embeddings, p=2) ** 2
        elif self.distance_metric == "cosine":
            # Cosine distance matrix
            similarity = torch.matmul(embeddings, embeddings.T)
            distances = 1 - similarity
        elif self.distance_metric == "l1":
            # L1 distance matrix
            distances = torch.cdist(embeddings, embeddings, p=1)

        # Average distance for positive pairs
        tone_invariance_score = (distances * positive_mask).sum() / num_positives

        return tone_invariance_score


def compute_tone_invariance_per_class(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    fst_labels: torch.Tensor,
    num_classes: int = 7,
    distance_metric: str = "l2",
    normalize: bool = True
) -> dict:
    """
    Compute tone-invariance metric per diagnosis class.

    Useful for identifying which classes have better/worse tone-invariance.

    Args:
        embeddings: Feature embeddings (N, D)
        labels: Diagnosis labels (N,)
        fst_labels: FST labels (N,)
        num_classes: Number of diagnosis classes
        distance_metric: Distance metric
        normalize: Whether to normalize embeddings

    Returns:
        Dictionary mapping class_idx → tone-invariance score

    Example:
        >>> embeddings = torch.randn(100, 2048)
        >>> labels = torch.randint(0, 7, (100,))
        >>> fst_labels = torch.randint(1, 7, (100,))
        >>> scores = compute_tone_invariance_per_class(embeddings, labels, fst_labels)
        >>> for cls_idx, score in scores.items():
        ...     print(f"Class {cls_idx}: {score:.4f}")
    """
    metric = ToneInvarianceMetric(distance_metric=distance_metric, normalize_embeddings=normalize)
    results = {}

    for class_idx in range(num_classes):
        # Filter to single class
        class_mask = labels == class_idx
        if class_mask.sum() < 2:
            # Not enough samples
            results[class_idx] = float('nan')
            continue

        class_embeddings = embeddings[class_mask]
        class_labels = labels[class_mask]
        class_fst = fst_labels[class_mask]

        # Compute tone-invariance for this class
        score = metric(class_embeddings, class_labels, class_fst)
        results[class_idx] = score.item()

    return results


if __name__ == "__main__":
    """Test CIRCLe regularization loss."""
    print("=" * 80)
    print("Testing CIRCLe Regularization Loss")
    print("=" * 80)

    # Test 1: Basic L2 regularization
    print("\n1. Testing L2 regularization loss...")
    loss_fn = CIRCLe_RegularizationLoss(distance_metric="l2")

    emb_orig = torch.randn(32, 2048)
    emb_trans = torch.randn(32, 2048)

    loss = loss_fn(emb_orig, emb_trans)
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Loss is scalar: {loss.dim() == 0}")
    print(f"   Loss is positive: {loss.item() > 0}")

    # Test 2: Identity transformation (should have low loss)
    print("\n2. Testing identity transformation (original = transformed)...")
    emb_identity = emb_orig.clone()
    loss_identity = loss_fn(emb_orig, emb_identity)
    print(f"   Loss for identity: {loss_identity.item():.6f}")
    print(f"   Loss near zero: {loss_identity.item() < 1e-5}")

    # Test 3: Cosine distance
    print("\n3. Testing cosine distance metric...")
    loss_fn_cosine = CIRCLe_RegularizationLoss(distance_metric="cosine")
    loss_cosine = loss_fn_cosine(emb_orig, emb_trans)
    print(f"   Cosine loss: {loss_cosine.item():.4f}")

    # Test 4: L1 distance
    print("\n4. Testing L1 distance metric...")
    loss_fn_l1 = CIRCLe_RegularizationLoss(distance_metric="l1")
    loss_l1 = loss_fn_l1(emb_orig, emb_trans)
    print(f"   L1 loss: {loss_l1.item():.4f}")

    # Test 5: Normalized embeddings
    print("\n5. Testing normalized embeddings...")
    loss_fn_norm = CIRCLe_RegularizationLoss(distance_metric="l2", normalize_embeddings=True)
    loss_norm = loss_fn_norm(emb_orig, emb_trans)
    print(f"   Loss with normalization: {loss_norm.item():.4f}")

    # Test 6: Pairwise distances
    print("\n6. Testing pairwise distance computation...")
    distances = loss_fn.get_pairwise_distances(emb_orig, emb_trans)
    print(f"   Distances shape: {distances.shape}")
    print(f"   Mean distance: {distances.mean().item():.4f}")
    print(f"   Std distance: {distances.std().item():.4f}")

    # Test 7: Multi-target regularization
    print("\n7. Testing multi-target CIRCLe loss...")
    multi_loss_fn = MultiTargetCIRCLe_Loss(target_fsts=[1, 6])

    emb_trans_dict = {
        1: torch.randn(32, 2048),
        6: torch.randn(32, 2048)
    }

    multi_loss = multi_loss_fn(emb_orig, emb_trans_dict)
    print(f"   Multi-target loss: {multi_loss.item():.4f}")

    # Test 8: Tone-invariance metric
    print("\n8. Testing tone-invariance metric...")
    metric = ToneInvarianceMetric(distance_metric="l2", normalize_embeddings=True)

    embeddings = torch.randn(64, 2048)
    labels = torch.randint(0, 7, (64,))
    fst_labels = torch.randint(1, 7, (64,))

    score = metric(embeddings, labels, fst_labels)
    print(f"   Tone-invariance score: {score.item():.4f}")
    print(f"   Score is positive: {score.item() > 0}")

    # Test 9: Per-class tone-invariance
    print("\n9. Testing per-class tone-invariance...")
    class_scores = compute_tone_invariance_per_class(
        embeddings, labels, fst_labels, num_classes=7
    )
    print(f"   Computed scores for {len(class_scores)} classes")
    valid_scores = [s for s in class_scores.values() if not np.isnan(s)]
    print(f"   Valid scores: {len(valid_scores)}")
    if valid_scores:
        print(f"   Mean score: {np.mean(valid_scores):.4f}")

    # Test 10: Gradient flow
    print("\n10. Testing gradient flow...")
    emb_orig_grad = torch.randn(8, 512, requires_grad=True)
    emb_trans_grad = torch.randn(8, 512, requires_grad=True)

    loss_grad = loss_fn(emb_orig_grad, emb_trans_grad)
    loss_grad.backward()

    print(f"   Loss has grad_fn: {loss_grad.grad_fn is not None}")
    print(f"   Original embeddings have gradients: {emb_orig_grad.grad is not None}")
    print(f"   Transformed embeddings have gradients: {emb_trans_grad.grad is not None}")
    print(f"   Gradient magnitude (orig): {emb_orig_grad.grad.norm().item():.4f}")

    print("\n" + "=" * 80)
    print("CIRCLe regularization loss test PASSED!")
    print("=" * 80)

    # Import numpy for statistics
    import numpy as np
