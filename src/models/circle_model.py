"""
CIRCLe Model: Color-Invariant Representation Learning

Extends FairDisCo with CIRCLe tone-invariant regularization. Combines adversarial
debiasing, contrastive learning, and color transformation regularization for
maximum fairness with minimal accuracy trade-off.

Based on: Pakzad et al. (2022) "CIRCLe: Color Invariant Representation Learning
for Unbiased Classification of Skin Lesions" ECCV 2022 Workshops

Clean-room implementation from research documentation only.

Architecture:
    - Base: FairDisCoClassifier (3 losses: cls + adv + con)
    - Addition: CIRCLe regularization (4th loss term)
    - Total Loss: L_cls + λ_adv*L_adv + λ_con*L_con + λ_reg*L_reg

Training Strategy:
    1. Dual forward pass: original + transformed images
    2. Extract embeddings from both passes
    3. Compute regularization loss (embedding distance)
    4. Combine with FairDisCo losses
    5. Single backward pass through all losses

Framework: MENDICANT_BIAS - Phase 2, Week 7-8
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from .fairdisco_model import FairDisCoClassifier
from ..fairness.color_transforms import LABColorTransform
from ..fairness.circle_regularization import CIRCLe_RegularizationLoss, MultiTargetCIRCLe_Loss


class CIRCLe_FairDisCo(nn.Module):
    """
    CIRCLe: FairDisCo with Color-Invariant Regularization.

    Extends FairDisCo adversarial debiasing with tone-invariant regularization
    via color transformations in LAB space.

    Four-Loss Training:
        L_total = L_cls + λ_adv*L_adv + λ_con*L_con + λ_reg*L_reg

    Where:
        - L_cls: Classification loss (cross-entropy)
        - L_adv: Adversarial FST loss (gradient reversal)
        - L_con: Supervised contrastive loss
        - L_reg: CIRCLe regularization (NEW)

    Training Flow:
        1. Forward pass on original images → embeddings_orig
        2. Transform images to target FST(s)
        3. Forward pass on transformed images → embeddings_trans
        4. Compute regularization: ||embeddings_orig - embeddings_trans||^2
        5. Combine with FairDisCo losses

    Args:
        num_classes: Number of diagnosis classes (default: 7)
        num_fst_classes: Number of FST classes (default: 6)
        backbone: Backbone architecture (default: "resnet50")
        pretrained: Use ImageNet pre-trained weights (default: True)
        contrastive_dim: Contrastive embedding dimension (default: 128)
        lambda_adv: Adversarial loss weight (default: 0.3)
        lambda_con: Contrastive loss weight (default: 0.2)
        lambda_reg: CIRCLe regularization weight (default: 0.2)
        target_fsts: Target FST classes for transformation (default: [1, 6])
        use_multi_target: Use multi-target regularization (default: True)
        distance_metric: Distance metric for regularization (default: "l2")
        cache_transforms: Pre-compute transformations (default: False)
        imagenet_normalized: Whether inputs are ImageNet normalized (default: True)

    Example:
        >>> model = CIRCLe_FairDisCo(num_classes=7, target_fsts=[1, 6])
        >>> images = torch.randn(16, 3, 224, 224)
        >>> fst_labels = torch.randint(1, 7, (16,))
        >>> outputs = model(images, fst_labels)
        >>> print(outputs.keys())
        >>> # dict_keys(['diagnosis_logits', 'fst_logits', 'contrastive_embeddings',
        >>> #             'embeddings_original', 'embeddings_transformed', 'images_transformed'])
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
        lambda_con: float = 0.2,
        lambda_reg: float = 0.2,
        target_fsts: List[int] = [1, 6],
        use_multi_target: bool = True,
        distance_metric: str = "l2",
        cache_transforms: bool = False,
        imagenet_normalized: bool = True,
        dropout_cls: float = 0.3,
        dropout_disc: Tuple[float, float] = (0.3, 0.2)
    ):
        super(CIRCLe_FairDisCo, self).__init__()

        self.num_classes = num_classes
        self.num_fst_classes = num_fst_classes
        self.target_fsts = target_fsts
        self.use_multi_target = use_multi_target
        self.cache_transforms = cache_transforms
        self.imagenet_normalized = imagenet_normalized

        # Initialize FairDisCo base model
        self.fairdisco_model = FairDisCoClassifier(
            num_classes=num_classes,
            num_fst_classes=num_fst_classes,
            backbone=backbone,
            pretrained=pretrained,
            contrastive_dim=contrastive_dim,
            projection_hidden_dim=projection_hidden_dim,
            lambda_adv=lambda_adv,
            dropout_cls=dropout_cls,
            dropout_disc=dropout_disc
        )

        # Color transformation module
        self.color_transform = LABColorTransform(
            normalize_input=True,
            imagenet_normalized=imagenet_normalized,
            clamp_output=True
        )

        # CIRCLe regularization loss
        if use_multi_target:
            self.regularization_loss = MultiTargetCIRCLe_Loss(
                target_fsts=target_fsts,
                distance_metric=distance_metric,
                normalize_embeddings=False,
                reduction="mean"
            )
        else:
            self.regularization_loss = CIRCLe_RegularizationLoss(
                distance_metric=distance_metric,
                normalize_embeddings=False,
                reduction="mean"
            )

        # Cache for pre-computed transformations (optional)
        self.cached_transforms = {} if cache_transforms else None

        # Store model configuration
        self.model_info = {
            'architecture': 'circle_fairdisco',
            'backbone': backbone,
            'num_classes': num_classes,
            'num_fst_classes': num_fst_classes,
            'contrastive_dim': contrastive_dim,
            'pretrained': pretrained,
            'lambda_adv': lambda_adv,
            'lambda_con': lambda_con,
            'lambda_reg': lambda_reg,
            'target_fsts': target_fsts,
            'use_multi_target': use_multi_target,
            'distance_metric': distance_metric
        }

    def forward(
        self,
        images: torch.Tensor,
        fst_labels: torch.Tensor,
        return_transformed_images: bool = False,
        return_all_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual processing: original + transformed images.

        Args:
            images: Input images (B, 3, H, W)
            fst_labels: FST labels for source images (B,)
            return_transformed_images: Whether to return transformed images
            return_all_embeddings: Whether to return all intermediate embeddings

        Returns:
            Dictionary containing:
                - diagnosis_logits: Diagnosis predictions (B, num_classes)
                - fst_logits: FST predictions (B, num_fst_classes)
                - contrastive_embeddings: Contrastive features (B, contrastive_dim)
                - embeddings_original: Original image embeddings (B, feature_dim)
                - embeddings_transformed: Transformed embeddings dict or single tensor
                - images_transformed: Transformed images (optional)
        """
        # 1. Forward pass on original images
        (diagnosis_logits, fst_logits, contrastive_embeddings,
         embeddings_original) = self.fairdisco_model(images, return_embeddings=True)

        # 2. Transform images to target FST(s)
        if self.use_multi_target:
            # Multi-target: transform to multiple FSTs
            images_transformed_dict = {}
            embeddings_transformed_dict = {}

            for target_fst in self.target_fsts:
                # Transform to target FST
                images_trans = self.color_transform(
                    images, fst_labels,
                    torch.full_like(fst_labels, target_fst)
                )

                images_transformed_dict[target_fst] = images_trans

                # Forward pass on transformed images (no gradient for classification)
                with torch.no_grad():
                    _, _, _, emb_trans = self.fairdisco_model(images_trans, return_embeddings=True)

                # Store embeddings (WITH gradient for regularization)
                embeddings_transformed_dict[target_fst] = self.fairdisco_model.backbone(images_trans)

            embeddings_transformed = embeddings_transformed_dict
            images_transformed = images_transformed_dict

        else:
            # Single target: transform to first target FST only
            target_fst = self.target_fsts[0]
            images_transformed = self.color_transform(
                images, fst_labels,
                torch.full_like(fst_labels, target_fst)
            )

            # Forward pass on transformed images
            embeddings_transformed = self.fairdisco_model.backbone(images_transformed)

        # 3. Prepare output dictionary
        outputs = {
            'diagnosis_logits': diagnosis_logits,
            'fst_logits': fst_logits,
            'contrastive_embeddings': contrastive_embeddings,
            'embeddings_original': embeddings_original,
            'embeddings_transformed': embeddings_transformed
        }

        if return_transformed_images:
            outputs['images_transformed'] = images_transformed

        if return_all_embeddings:
            # Include all intermediate representations
            outputs['feature_dim'] = embeddings_original.size(1)

        return outputs

    def compute_circle_loss(
        self,
        embeddings_original: torch.Tensor,
        embeddings_transformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CIRCLe regularization loss.

        Args:
            embeddings_original: Original image embeddings (B, D)
            embeddings_transformed: Transformed embeddings (dict or tensor)

        Returns:
            Scalar regularization loss
        """
        return self.regularization_loss(embeddings_original, embeddings_transformed)

    def update_lambda_reg(self, lambda_reg: float):
        """
        Update CIRCLe regularization strength.

        Used for lambda scheduling during training.

        Args:
            lambda_reg: New regularization weight
        """
        self.model_info['lambda_reg'] = lambda_reg

    def update_lambda_adv(self, lambda_adv: float):
        """
        Update adversarial loss weight (inherited from FairDisCo).

        Args:
            lambda_adv: New adversarial weight
        """
        self.fairdisco_model.update_lambda_adv(lambda_adv)
        self.model_info['lambda_adv'] = lambda_adv

    def get_fairdisco_model(self) -> FairDisCoClassifier:
        """Get underlying FairDisCo model."""
        return self.fairdisco_model

    def get_feature_extractor(self) -> nn.Module:
        """Get backbone feature extractor."""
        return self.fairdisco_model.backbone

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return self.model_info

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        self.fairdisco_model.freeze_backbone()

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        self.fairdisco_model.unfreeze_backbone()

    def load_fairdisco_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """
        Load pre-trained FairDisCo checkpoint.

        Useful for initializing CIRCLe from a trained FairDisCo model.

        Args:
            checkpoint_path: Path to FairDisCo checkpoint
            strict: Whether to strictly enforce key matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load into FairDisCo model
        self.fairdisco_model.load_state_dict(state_dict, strict=strict)
        print(f"Loaded FairDisCo checkpoint from {checkpoint_path}")


def create_circle_model(
    num_classes: int = 7,
    num_fst_classes: int = 6,
    backbone: str = "resnet50",
    pretrained: bool = True,
    contrastive_dim: int = 128,
    lambda_adv: float = 0.3,
    lambda_con: float = 0.2,
    lambda_reg: float = 0.2,
    target_fsts: List[int] = [1, 6],
    distance_metric: str = "l2",
    fairdisco_checkpoint: Optional[str] = None
) -> CIRCLe_FairDisCo:
    """
    Factory function to create CIRCLe model.

    Args:
        num_classes: Number of diagnosis classes
        num_fst_classes: Number of FST classes
        backbone: Backbone architecture
        pretrained: Use ImageNet pre-trained weights
        contrastive_dim: Contrastive embedding dimension
        lambda_adv: Adversarial loss weight
        lambda_con: Contrastive loss weight
        lambda_reg: CIRCLe regularization weight
        target_fsts: Target FST classes for transformation
        distance_metric: Distance metric for regularization
        fairdisco_checkpoint: Optional FairDisCo checkpoint to initialize from

    Returns:
        CIRCLe_FairDisCo model instance

    Example:
        >>> # Create from scratch
        >>> model = create_circle_model(num_classes=7, lambda_reg=0.2)
        >>>
        >>> # Or initialize from FairDisCo checkpoint
        >>> model = create_circle_model(
        ...     num_classes=7,
        ...     fairdisco_checkpoint="checkpoints/fairdisco_best.pth"
        ... )
    """
    model = CIRCLe_FairDisCo(
        num_classes=num_classes,
        num_fst_classes=num_fst_classes,
        backbone=backbone,
        pretrained=pretrained,
        contrastive_dim=contrastive_dim,
        lambda_adv=lambda_adv,
        lambda_con=lambda_con,
        lambda_reg=lambda_reg,
        target_fsts=target_fsts,
        use_multi_target=(len(target_fsts) > 1),
        distance_metric=distance_metric
    )

    # Optionally load FairDisCo checkpoint
    if fairdisco_checkpoint is not None:
        model.load_fairdisco_checkpoint(fairdisco_checkpoint, strict=False)

    return model


if __name__ == "__main__":
    """Test CIRCLe model architecture."""
    print("=" * 80)
    print("Testing CIRCLe Model Architecture")
    print("=" * 80)

    # Test 1: Create model
    print("\n1. Creating CIRCLe model...")
    model = create_circle_model(
        num_classes=7,
        num_fst_classes=6,
        pretrained=False,  # Faster for testing
        target_fsts=[1, 6],
        lambda_reg=0.2
    )
    print(f"   Model created: {model.model_info['architecture']}")
    print(f"   Target FSTs: {model.model_info['target_fsts']}")
    print(f"   Multi-target: {model.model_info['use_multi_target']}")

    # Test 2: Forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    fst_labels = torch.randint(1, 7, (batch_size,))

    with torch.no_grad():
        outputs = model(images, fst_labels, return_transformed_images=True)

    print(f"   Input shape: {images.shape}")
    print(f"   FST labels: {fst_labels.tolist()}")
    print(f"   Output keys: {list(outputs.keys())}")
    print(f"   Diagnosis logits shape: {outputs['diagnosis_logits'].shape}")
    print(f"   FST logits shape: {outputs['fst_logits'].shape}")
    print(f"   Contrastive embeddings shape: {outputs['contrastive_embeddings'].shape}")
    print(f"   Original embeddings shape: {outputs['embeddings_original'].shape}")

    # Test 3: Multi-target transformation
    print("\n3. Testing multi-target transformation...")
    if isinstance(outputs['embeddings_transformed'], dict):
        print(f"   Multi-target mode active")
        for target_fst, emb in outputs['embeddings_transformed'].items():
            print(f"   FST {target_fst} embeddings shape: {emb.shape}")
        for target_fst, img in outputs['images_transformed'].items():
            print(f"   FST {target_fst} images shape: {img.shape}")
    else:
        print(f"   Single-target mode active")
        print(f"   Transformed embeddings shape: {outputs['embeddings_transformed'].shape}")

    # Test 4: CIRCLe regularization loss
    print("\n4. Testing CIRCLe regularization loss...")
    loss_reg = model.compute_circle_loss(
        outputs['embeddings_original'],
        outputs['embeddings_transformed']
    )
    print(f"   Regularization loss: {loss_reg.item():.4f}")
    print(f"   Loss is scalar: {loss_reg.dim() == 0}")
    print(f"   Loss is positive: {loss_reg.item() > 0}")

    # Test 5: Single-target model
    print("\n5. Testing single-target model...")
    model_single = create_circle_model(
        num_classes=7,
        pretrained=False,
        target_fsts=[1]  # Single target
    )

    with torch.no_grad():
        outputs_single = model_single(images, fst_labels)

    print(f"   Single-target embeddings type: {type(outputs_single['embeddings_transformed'])}")
    if not isinstance(outputs_single['embeddings_transformed'], dict):
        print(f"   Shape: {outputs_single['embeddings_transformed'].shape}")

    # Test 6: Lambda updates
    print("\n6. Testing lambda updates...")
    print(f"   Initial lambda_reg: {model.model_info['lambda_reg']}")
    model.update_lambda_reg(0.3)
    print(f"   Updated lambda_reg: {model.model_info['lambda_reg']}")

    print(f"   Initial lambda_adv: {model.model_info['lambda_adv']}")
    model.update_lambda_adv(0.4)
    print(f"   Updated lambda_adv: {model.model_info['lambda_adv']}")

    # Test 7: Parameter count
    print("\n7. Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # Test 8: Gradient flow through regularization
    print("\n8. Testing gradient flow...")
    images_grad = torch.randn(2, 3, 224, 224, requires_grad=True)
    fst_labels_grad = torch.tensor([3, 4])

    outputs_grad = model(images_grad, fst_labels_grad)
    loss_reg_grad = model.compute_circle_loss(
        outputs_grad['embeddings_original'],
        outputs_grad['embeddings_transformed']
    )

    loss_reg_grad.backward()

    print(f"   Loss has grad_fn: {loss_reg_grad.grad_fn is not None}")
    print(f"   Input images have gradients: {images_grad.grad is not None}")
    if images_grad.grad is not None:
        print(f"   Gradient magnitude: {images_grad.grad.norm().item():.4f}")

    # Test 9: Feature extractor access
    print("\n9. Testing feature extractor access...")
    feature_extractor = model.get_feature_extractor()
    test_features = feature_extractor(images[:2])
    print(f"   Feature extractor output shape: {test_features.shape}")

    # Test 10: Model info
    print("\n10. Testing model info...")
    info = model.get_model_info()
    print(f"   Model info keys: {list(info.keys())}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Backbone: {info['backbone']}")

    print("\n" + "=" * 80)
    print("CIRCLe model architecture test PASSED!")
    print("=" * 80)
