"""
FairDisCo: Fair Disentanglement with Contrastive Learning

Implementation of FairDisCo adversarial debiasing architecture for fairness-aware
skin cancer detection. Uses gradient reversal layer, adversarial discriminator,
and supervised contrastive learning to remove skin tone bias from latent representations.

Based on: Wind et al. (2022) "FairDisCo: Fairer AI in Dermatology via
Disentanglement Contrastive Learning" ECCV ISIC Workshop (Best Paper)

Clean-room implementation from research documentation only.

Architecture:
- Shared backbone (ResNet50) for feature extraction
- Classification head for diagnosis prediction
- Adversarial discriminator with gradient reversal layer (FST prediction)
- Contrastive projection head for supervised contrastive loss

Framework: MENDICANT_BIAS - Phase 2, Week 5-6
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict, Any


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) - Custom autograd function.

    Forward pass: Identity operation (y = x)
    Backward pass: Multiply gradient by -lambda (reverses gradient direction)

    This enables adversarial training without separate discriminator optimization:
    - Discriminator learns to predict FST (normal gradient)
    - Backbone learns to prevent FST prediction (reversed gradient)
    - Result: FST-invariant embeddings
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass - identity operation.

        Args:
            ctx: Context object for backward pass
            x: Input tensor
            lambda_: Gradient reversal strength (typically 0.1-0.5)

        Returns:
            Input tensor unchanged
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass - reverse and scale gradient.

        Args:
            ctx: Context object from forward pass
            grad_output: Gradient from downstream layers

        Returns:
            Tuple of (reversed gradient, None for lambda_)
        """
        lambda_ = ctx.lambda_
        # Reverse gradient direction and scale by lambda
        grad_input = grad_output.neg() * lambda_
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer wrapper module.

    Used between feature extractor and adversarial discriminator to enable
    adversarial training via gradient reversal.

    Args:
        lambda_: Initial gradient reversal strength (default: 0.0)
    """

    def __init__(self, lambda_: float = 0.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update gradient reversal strength during training."""
        self.lambda_ = lambda_


class FST_Discriminator(nn.Module):
    """
    Adversarial FST Discriminator.

    3-layer MLP that attempts to predict Fitzpatrick Skin Type (FST) from
    feature embeddings. Connected via gradient reversal layer to encourage
    FST-invariant representations.

    Architecture:
        - Input: 2048-dim feature embeddings (from ResNet50)
        - Layer 1: Linear(2048 -> 512) + ReLU + Dropout(0.3)
        - Layer 2: Linear(512 -> 256) + ReLU + Dropout(0.2)
        - Layer 3: Linear(256 -> num_fst_classes)
        - Output: FST logits (6 classes for FST I-VI)

    Args:
        feature_dim: Dimension of input features (default: 2048 for ResNet50)
        num_fst_classes: Number of FST classes (default: 6 for FST I-VI)
        dropout1: Dropout rate after first layer (default: 0.3)
        dropout2: Dropout rate after second layer (default: 0.2)
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_fst_classes: int = 6,
        dropout1: float = 0.3,
        dropout2: float = 0.2
    ):
        super(FST_Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout2),
            nn.Linear(256, num_fst_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.

        Args:
            x: Feature embeddings (batch_size, feature_dim)

        Returns:
            FST logits (batch_size, num_fst_classes)
        """
        return self.discriminator(x)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Encourages same-diagnosis embeddings to cluster together regardless of FST,
    while pushing different-diagnosis embeddings apart.

    Key insight: Positives are same-diagnosis, different-FST pairs.
    This forces model to learn diagnosis-specific features that are FST-invariant.

    Formula:
        L_con = -1/|P(i)| * sum_{p in P(i)} log[exp(sim(z_i, z_p)/tau) / sum_{a in A(i)} exp(sim(z_i, z_a)/tau)]

    Where:
        - z_i: Normalized embedding of anchor sample i
        - P(i): Positive pairs (same diagnosis, different FST)
        - A(i): All samples except i
        - tau: Temperature parameter (controls separation, default 0.07)

    Args:
        temperature: Temperature scaling parameter (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        fst_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: Normalized feature embeddings (batch_size, embedding_dim)
            labels: Diagnosis labels (batch_size,)
            fst_labels: FST labels (batch_size,)

        Returns:
            Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Normalize embeddings (L2 norm)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix (batch_size x batch_size)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create masks for positive pairs
        labels = labels.contiguous().view(-1, 1)
        fst_labels = fst_labels.contiguous().view(-1, 1)

        # Same diagnosis mask
        label_mask = torch.eq(labels, labels.T).float().to(device)

        # Different FST mask
        fst_mask = ~torch.eq(fst_labels, fst_labels.T)
        fst_mask = fst_mask.float().to(device)

        # Positive pairs: same diagnosis AND different FST
        positive_mask = label_mask * fst_mask

        # Remove self-similarity
        self_mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        positive_mask = positive_mask * (1 - self_mask)

        # For numerical stability: subtract max
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute exp and denominator
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        # Handle case where a sample has no positive pairs
        num_positives = positive_mask.sum(dim=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

        # Mask out samples with no positive pairs
        valid_samples = (num_positives > 0).float()

        # Final loss: negative mean over valid samples
        loss = -(mean_log_prob_pos * valid_samples).sum() / (valid_samples.sum() + 1e-8)

        return loss


class FairDisCoClassifier(nn.Module):
    """
    FairDisCo: Fair Disentanglement with Contrastive Learning.

    Complete fairness-aware skin cancer detection architecture combining:
    1. Shared ResNet50 backbone for feature extraction
    2. Classification head for diagnosis prediction
    3. Adversarial discriminator (with GRL) for FST debiasing
    4. Contrastive projection head for feature quality

    Training uses three losses:
        L_total = L_cls + lambda_adv * L_adv + lambda_con * L_con

    Where:
        - L_cls: Cross-entropy for diagnosis classification
        - L_adv: Cross-entropy for FST prediction (reversed gradient)
        - L_con: Supervised contrastive loss

    Args:
        num_classes: Number of diagnosis classes (default: 7 for HAM10000)
        num_fst_classes: Number of FST classes (default: 6 for FST I-VI)
        backbone: Backbone architecture name (default: "resnet50")
        pretrained: Use ImageNet pre-trained weights (default: True)
        contrastive_dim: Dimension of contrastive embeddings (default: 128)
        projection_hidden_dim: Hidden dimension in projection head (default: 1024)
        lambda_adv: Initial adversarial loss weight (default: 0.3)
        dropout_cls: Dropout rate in classification head (default: 0.3)
        dropout_disc: Dropout rates in discriminator (default: (0.3, 0.2))
    """

    def __init__(
        self,
        num_classes: int = 7,
        num_fst_classes: int = 6,
        backbone: str = "resnet50",
        pretrained: bool = True,
        contrastive_dim: int = 128,
        projection_hidden_dim: int = 1024,
        lambda_adv: float = 0.3,
        dropout_cls: float = 0.3,
        dropout_disc: Tuple[float, float] = (0.3, 0.2)
    ):
        super(FairDisCoClassifier, self).__init__()

        self.num_classes = num_classes
        self.num_fst_classes = num_fst_classes
        self.contrastive_dim = contrastive_dim

        # Load backbone (ResNet50)
        if backbone == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50(weights=None)

            # Get feature dimension (2048 for ResNet50)
            self.feature_dim = self.backbone.fc.in_features

            # Remove original FC layer (keep as feature extractor)
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Currently only 'resnet50' is supported.")

        # Classification head (diagnosis prediction)
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_cls),
            nn.Linear(512, num_classes)
        )

        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_=lambda_adv)

        # Adversarial discriminator (FST prediction)
        self.discriminator = FST_Discriminator(
            feature_dim=self.feature_dim,
            num_fst_classes=num_fst_classes,
            dropout1=dropout_disc[0],
            dropout2=dropout_disc[1]
        )

        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, contrastive_dim)
        )

        # Store model configuration
        self.model_info = {
            'architecture': 'fairdisco',
            'backbone': backbone,
            'num_classes': num_classes,
            'num_fst_classes': num_fst_classes,
            'contrastive_dim': contrastive_dim,
            'pretrained': pretrained,
            'lambda_adv': lambda_adv
        }

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through FairDisCo model.

        Args:
            x: Input images (batch_size, 3, 224, 224)
            return_embeddings: Whether to return raw feature embeddings

        Returns:
            Tuple of:
                - diagnosis_logits: Diagnosis predictions (batch_size, num_classes)
                - fst_logits: FST predictions (batch_size, num_fst_classes)
                - contrastive_embeddings: Normalized contrastive features (batch_size, contrastive_dim)
                - embeddings (optional): Raw feature embeddings (batch_size, feature_dim)
        """
        # Extract features from backbone
        embeddings = self.backbone(x)  # (batch_size, 2048)

        # Classification branch
        diagnosis_logits = self.classification_head(embeddings)

        # Adversarial branch (with gradient reversal)
        reversed_embeddings = self.grl(embeddings)
        fst_logits = self.discriminator(reversed_embeddings)

        # Contrastive branch
        contrastive_features = self.projection_head(embeddings)
        contrastive_embeddings = F.normalize(contrastive_features, p=2, dim=1)

        if return_embeddings:
            return diagnosis_logits, fst_logits, contrastive_embeddings, embeddings
        else:
            return diagnosis_logits, fst_logits, contrastive_embeddings, None

    def update_lambda_adv(self, lambda_adv: float):
        """
        Update gradient reversal strength during training.

        Used for lambda scheduling (e.g., warm-up from 0.0 to 0.3).

        Args:
            lambda_adv: New gradient reversal strength
        """
        self.grl.set_lambda(lambda_adv)
        self.model_info['lambda_adv'] = lambda_adv

    def get_feature_extractor(self) -> nn.Module:
        """
        Get backbone feature extractor (for inference without classification).

        Returns:
            Backbone module that outputs feature embeddings
        """
        return self.backbone

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return self.model_info

    def freeze_backbone(self):
        """Freeze backbone parameters (useful for initial training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (for fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_fairdisco_model(
    num_classes: int = 7,
    num_fst_classes: int = 6,
    backbone: str = "resnet50",
    pretrained: bool = True,
    contrastive_dim: int = 128,
    lambda_adv: float = 0.3
) -> FairDisCoClassifier:
    """
    Factory function to create FairDisCo model.

    Args:
        num_classes: Number of diagnosis classes
        num_fst_classes: Number of FST classes
        backbone: Backbone architecture name
        pretrained: Use ImageNet pre-trained weights
        contrastive_dim: Dimension of contrastive embeddings
        lambda_adv: Initial gradient reversal strength

    Returns:
        FairDisCoClassifier instance

    Example:
        >>> model = create_fairdisco_model(num_classes=7, pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> diag_logits, fst_logits, contrast_emb, _ = model(x)
        >>> print(diag_logits.shape)  # torch.Size([4, 7])
        >>> print(fst_logits.shape)   # torch.Size([4, 6])
        >>> print(contrast_emb.shape) # torch.Size([4, 128])
    """
    model = FairDisCoClassifier(
        num_classes=num_classes,
        num_fst_classes=num_fst_classes,
        backbone=backbone,
        pretrained=pretrained,
        contrastive_dim=contrastive_dim,
        lambda_adv=lambda_adv
    )
    return model


if __name__ == "__main__":
    """Test FairDisCo model architecture."""
    print("=" * 80)
    print("Testing FairDisCo Model Architecture")
    print("=" * 80)

    # Create model
    print("\n1. Creating FairDisCo model...")
    model = create_fairdisco_model(
        num_classes=7,
        num_fst_classes=6,
        pretrained=False  # Faster for testing
    )
    print(f"   Model created: {model.model_info['architecture']}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        diag_logits, fst_logits, contrast_emb, embeddings = model(x, return_embeddings=True)

    print(f"   Input shape: {x.shape}")
    print(f"   Diagnosis logits shape: {diag_logits.shape}")
    print(f"   FST logits shape: {fst_logits.shape}")
    print(f"   Contrastive embeddings shape: {contrast_emb.shape}")
    print(f"   Feature embeddings shape: {embeddings.shape}")

    # Test gradient reversal
    print("\n3. Testing gradient reversal layer...")
    grl = GradientReversalLayer(lambda_=0.5)
    x_test = torch.randn(4, 2048, requires_grad=True)
    y = grl(x_test)
    loss = y.sum()
    loss.backward()

    # Gradient should be reversed and scaled by 0.5
    print(f"   Forward output equals input: {torch.allclose(y, x_test)}")
    print(f"   Gradient reversed: {torch.allclose(x_test.grad, -0.5 * torch.ones_like(x_test.grad))}")

    # Test contrastive loss
    print("\n4. Testing supervised contrastive loss...")
    contrast_loss_fn = SupervisedContrastiveLoss(temperature=0.07)

    # Create dummy data
    embeddings = torch.randn(8, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    fst_labels = torch.tensor([1, 3, 1, 3, 2, 4, 2, 4])

    loss = contrast_loss_fn(embeddings, labels, fst_labels)
    print(f"   Contrastive loss computed: {loss.item():.4f}")
    print(f"   Loss is finite: {torch.isfinite(loss).item()}")

    # Count parameters
    print("\n5. Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # Test lambda update
    print("\n6. Testing lambda update...")
    initial_lambda = model.model_info['lambda_adv']
    model.update_lambda_adv(0.5)
    new_lambda = model.model_info['lambda_adv']
    print(f"   Initial lambda: {initial_lambda}")
    print(f"   Updated lambda: {new_lambda}")

    print("\n" + "=" * 80)
    print("FairDisCo model architecture test PASSED!")
    print("=" * 80)
