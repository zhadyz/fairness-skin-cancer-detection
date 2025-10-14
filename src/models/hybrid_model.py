"""
Hybrid ConvNeXtV2-Swin Transformer Architecture with Multi-Scale Fusion

Combines local feature extraction (ConvNeXtV2) with global context modeling (Swin)
for improved skin cancer classification with fairness.

Key Features:
- ConvNeXtV2 for local texture patterns
- Swin Transformer for global structure
- Multi-scale feature fusion with attention
- FairDisCo integration for fairness
- Modular design for easy customization

Framework: MENDICANT_BIAS - Phase 3
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .convnextv2 import ConvNeXtV2Backbone, create_convnextv2_backbone
from .swin_transformer import SwinBackbone, create_swin_backbone


@dataclass
class HybridModelConfig:
    """Configuration for hybrid model."""

    # ConvNeXtV2 config
    convnext_variant: str = 'base'  # tiny, small, base, large
    convnext_drop_path: float = 0.1
    use_grn: bool = True

    # Swin config
    swin_variant: str = 'small'  # tiny, small, base
    swin_drop_path: float = 0.2

    # Fusion config
    fusion_method: str = 'pyramid'  # pyramid, concat, attention
    fusion_dim: int = 768
    use_fusion_attention: bool = True

    # Classification head
    num_classes: int = 7
    dropout: float = 0.3
    use_batch_norm: bool = True

    # FairDisCo integration
    enable_fairdisco: bool = False
    num_fst_classes: int = 6
    lambda_adv: float = 0.3


class MultiScaleFusionModule(nn.Module):
    """
    Multi-scale feature fusion module with attention mechanism.

    Fuses features from ConvNeXtV2 (local) and Swin (global) using:
    1. Feature pyramid: Align spatial dimensions
    2. Channel-wise attention: Weight feature importance
    3. Adaptive pooling: Fixed output size
    4. Projection: Unified embedding space

    Args:
        convnext_dims: List of ConvNeXtV2 feature dimensions per stage
        swin_dim: Swin Transformer output dimension
        fusion_dim: Output fusion dimension
        fusion_method: Fusion strategy ('pyramid', 'concat', 'attention')
        use_attention: Whether to use attention weighting
    """

    def __init__(
        self,
        convnext_dims: List[int],
        swin_dim: int,
        fusion_dim: int = 768,
        fusion_method: str = 'pyramid',
        use_attention: bool = True
    ):
        super().__init__()

        self.fusion_method = fusion_method
        self.use_attention = use_attention

        # Projections for ConvNeXt features to common dimension
        self.convnext_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, fusion_dim // 4, kernel_size=1),
                nn.BatchNorm2d(fusion_dim // 4),
                nn.GELU()
            )
            for dim in convnext_dims
        ])

        # Projection for Swin features
        self.swin_projection = nn.Sequential(
            nn.Conv2d(swin_dim, fusion_dim // 2, kernel_size=1),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.GELU()
        )

        # Attention mechanism for feature weighting
        if use_attention:
            total_features = len(convnext_dims) + 1  # ConvNeXt stages + Swin
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(fusion_dim, fusion_dim // 4),
                nn.GELU(),
                nn.Linear(fusion_dim // 4, total_features),
                nn.Softmax(dim=1)
            )

        # Final fusion projection
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1, groups=fusion_dim),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU(),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1),
            nn.BatchNorm2d(fusion_dim),
            nn.GELU()
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        convnext_features: List[torch.Tensor],
        swin_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-scale features from ConvNeXt and Swin.

        Args:
            convnext_features: List of ConvNeXt features [stage1, stage2, stage3]
                Each: (B, C_i, H_i, W_i)
            swin_features: Swin features (B, C_swin, H_swin, W_swin)

        Returns:
            Fused features: (B, fusion_dim)
        """
        batch_size = convnext_features[0].shape[0]

        # Project ConvNeXt features and upsample to common spatial size
        # Target size: largest ConvNeXt feature (stage 1)
        target_size = convnext_features[0].shape[2:]

        projected_convnext = []
        for i, feat in enumerate(convnext_features):
            proj_feat = self.convnext_projections[i](feat)
            if proj_feat.shape[2:] != target_size:
                proj_feat = F.interpolate(proj_feat, size=target_size, mode='bilinear', align_corners=False)
            projected_convnext.append(proj_feat)

        # Project and upsample Swin features
        projected_swin = self.swin_projection(swin_features)
        if projected_swin.shape[2:] != target_size:
            projected_swin = F.interpolate(projected_swin, size=target_size, mode='bilinear', align_corners=False)

        # Concatenate all features
        # ConvNeXt: 3 stages × (fusion_dim // 4) = 3/4 fusion_dim
        # Swin: 1/2 fusion_dim
        # Total: 5/4 fusion_dim, but we'll pad to fusion_dim
        all_features = torch.cat(projected_convnext + [projected_swin], dim=1)  # (B, C_total, H, W)

        # Pad or project to exact fusion_dim
        if all_features.shape[1] != self.fusion[0].in_channels:
            # Add 1x1 conv to match dimensions
            if not hasattr(self, 'dim_matcher'):
                self.dim_matcher = nn.Conv2d(all_features.shape[1], self.fusion[0].in_channels, 1).to(all_features.device)
            all_features = self.dim_matcher(all_features)

        # Apply attention weighting if enabled
        if self.use_attention:
            attn_weights = self.attention(all_features)  # (B, num_features)
            # Reshape to (B, num_features, 1, 1) for broadcasting
            attn_weights = attn_weights.view(batch_size, -1, 1, 1)

            # Weight each feature map group
            num_features = len(projected_convnext) + 1
            feature_dim = all_features.shape[1] // num_features

            weighted_features = []
            for i in range(num_features):
                start_idx = i * feature_dim
                end_idx = start_idx + feature_dim if i < num_features - 1 else all_features.shape[1]
                feat_slice = all_features[:, start_idx:end_idx]
                weighted_features.append(feat_slice * attn_weights[:, i:i+1])

            all_features = torch.cat(weighted_features, dim=1)

        # Apply fusion layers
        fused = self.fusion(all_features)  # (B, fusion_dim, H, W)

        # Global pooling
        output = self.global_pool(fused)  # (B, fusion_dim, 1, 1)
        output = output.flatten(1)  # (B, fusion_dim)

        return output


class HybridFairnessClassifier(nn.Module):
    """
    Hybrid ConvNeXtV2-Swin Transformer classifier with optional FairDisCo.

    Architecture:
        Input (224×224×3)
        -> ConvNeXtV2 (local features)
        -> Swin Transformer (global features)
        -> Multi-Scale Fusion
        -> Classification Head (+ optional Discriminator for FairDisCo)

    Args:
        config: HybridModelConfig dataclass with all settings
    """

    def __init__(self, config: HybridModelConfig):
        super().__init__()

        self.config = config

        # ConvNeXtV2 backbone
        self.convnext = create_convnextv2_backbone(
            variant=config.convnext_variant,
            drop_path_rate=config.convnext_drop_path,
            use_grn=config.use_grn,
            pretrained=False
        )

        # Get ConvNeXt output dimensions
        convnext_dims = self.convnext.get_feature_dims()  # [128, 256, 512] for base
        convnext_out_dim = convnext_dims[-1]  # 512 for base
        convnext_out_size = 14  # After 3 stages: 224 -> 56 -> 28 -> 14

        # Swin Transformer backbone
        self.swin = create_swin_backbone(
            variant=config.swin_variant,
            in_channels=convnext_out_dim,
            drop_path_rate=config.swin_drop_path,
            input_resolution=(convnext_out_size, convnext_out_size),
            pretrained=False
        )

        # Get Swin output dimension
        swin_embed_dim = 128 if config.swin_variant == 'small' else \
                        96 if config.swin_variant == 'tiny' else 192
        swin_out_dim = swin_embed_dim * 2  # After 1 PatchMerging

        # Multi-scale fusion module
        self.fusion = MultiScaleFusionModule(
            convnext_dims=convnext_dims,
            swin_dim=swin_out_dim,
            fusion_dim=config.fusion_dim,
            fusion_method=config.fusion_method,
            use_attention=config.use_fusion_attention
        )

        # Classification head
        self.classifier = self._build_classifier(config.fusion_dim, config.num_classes)

        # FairDisCo components (optional)
        if config.enable_fairdisco:
            self.gradient_reversal = GradientReversalLayer()
            self.discriminator = self._build_discriminator(config.fusion_dim, config.num_fst_classes)
            # Contrastive projection (for supervised contrastive loss)
            self.contrastive_projection = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim // 2),
                nn.GELU(),
                nn.Linear(config.fusion_dim // 2, 128)
            )
        else:
            self.gradient_reversal = None
            self.discriminator = None
            self.contrastive_projection = None

    def _build_classifier(self, in_features: int, num_classes: int) -> nn.Module:
        """Build classification head."""
        layers = []

        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features))

        layers.append(nn.Dropout(self.config.dropout))
        layers.append(nn.Linear(in_features, in_features // 2))
        layers.append(nn.GELU())

        if self.config.use_batch_norm:
            layers.append(nn.BatchNorm1d(in_features // 2))

        layers.append(nn.Dropout(self.config.dropout / 2))
        layers.append(nn.Linear(in_features // 2, num_classes))

        return nn.Sequential(*layers)

    def _build_discriminator(self, in_features: int, num_fst_classes: int) -> nn.Module:
        """Build FST discriminator for FairDisCo."""
        return nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_features // 2, in_features // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_features // 4, num_fst_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through hybrid model.

        Args:
            x: Input images (B, 3, 224, 224)
            return_features: If True, return intermediate features

        Returns:
            If FairDisCo disabled:
                - diagnosis_logits: (B, num_classes)
            If FairDisCo enabled:
                - diagnosis_logits: (B, num_classes)
                - fst_logits: (B, num_fst_classes)
                - contrastive_embeddings: (B, 128)
                - fused_features: (B, fusion_dim) if return_features
        """
        # ConvNeXtV2 forward
        convnext_out, convnext_features = self.convnext(x)
        # convnext_features: [(B, 128, 56, 56), (B, 256, 28, 28), (B, 512, 14, 14)]

        # Swin forward (takes ConvNeXt output)
        swin_out, _ = self.swin(convnext_out)
        # swin_out: (B, 256, 7, 7) for small variant

        # Multi-scale fusion
        fused_features = self.fusion(convnext_features, swin_out)
        # fused_features: (B, fusion_dim)

        # Classification
        diagnosis_logits = self.classifier(fused_features)

        # FairDisCo outputs
        if self.config.enable_fairdisco:
            # Gradient reversal for adversarial debiasing
            reversed_features = self.gradient_reversal(fused_features)
            fst_logits = self.discriminator(reversed_features)

            # Contrastive embeddings
            contrastive_embeddings = self.contrastive_projection(fused_features)
            contrastive_embeddings = F.normalize(contrastive_embeddings, p=2, dim=1)

            if return_features:
                return diagnosis_logits, fst_logits, contrastive_embeddings, fused_features
            else:
                return diagnosis_logits, fst_logits, contrastive_embeddings
        else:
            if return_features:
                return diagnosis_logits, fused_features
            else:
                return diagnosis_logits

    def get_model_info(self) -> Dict[str, any]:
        """Get comprehensive model information."""
        convnext_params = sum(p.numel() for p in self.convnext.parameters())
        swin_params = sum(p.numel() for p in self.swin.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        head_params = sum(p.numel() for p in self.classifier.parameters())

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'architecture': 'HybridConvNeXtV2Swin',
            'convnext_variant': self.config.convnext_variant,
            'swin_variant': self.config.swin_variant,
            'fusion_method': self.config.fusion_method,
            'fusion_dim': self.config.fusion_dim,
            'num_classes': self.config.num_classes,
            'enable_fairdisco': self.config.enable_fairdisco,
            'parameter_breakdown': {
                'convnext': convnext_params,
                'swin': swin_params,
                'fusion': fusion_params,
                'classifier': head_params,
                'total': total_params,
                'trainable': trainable_params
            }
        }

        if self.config.enable_fairdisco:
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            cont_params = sum(p.numel() for p in self.contrastive_projection.parameters())
            info['parameter_breakdown']['discriminator'] = disc_params
            info['parameter_breakdown']['contrastive'] = cont_params

        return info


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for adversarial training.

    Forward: Identity
    Backward: Reverses gradient sign with scaling factor lambda
    """

    def __init__(self, lambda_value: float = 1.0):
        super().__init__()
        self.lambda_value = lambda_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_value)

    def set_lambda(self, lambda_value: float):
        """Update lambda value for gradient reversal strength."""
        self.lambda_value = lambda_value


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal function for adversarial training.
    """

    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


def create_hybrid_model(
    convnext_variant: str = 'base',
    swin_variant: str = 'small',
    num_classes: int = 7,
    enable_fairdisco: bool = False,
    num_fst_classes: int = 6,
    fusion_dim: int = 768,
    dropout: float = 0.3
) -> HybridFairnessClassifier:
    """
    Factory function to create hybrid model with sensible defaults.

    Args:
        convnext_variant: ConvNeXtV2 variant
        swin_variant: Swin variant
        num_classes: Number of diagnosis classes
        enable_fairdisco: Enable FairDisCo integration
        num_fst_classes: Number of FST classes (if FairDisCo enabled)
        fusion_dim: Fusion module output dimension
        dropout: Dropout rate in classifier

    Returns:
        HybridFairnessClassifier instance
    """
    config = HybridModelConfig(
        convnext_variant=convnext_variant,
        swin_variant=swin_variant,
        num_classes=num_classes,
        enable_fairdisco=enable_fairdisco,
        num_fst_classes=num_fst_classes,
        fusion_dim=fusion_dim,
        dropout=dropout
    )

    return HybridFairnessClassifier(config)


if __name__ == "__main__":
    """Test hybrid model."""
    print("=" * 80)
    print("Testing Hybrid ConvNeXtV2-Swin Model")
    print("=" * 80)

    # Test without FairDisCo
    print("\n1. Creating hybrid model (without FairDisCo)...")
    model = create_hybrid_model(
        convnext_variant='base',
        swin_variant='small',
        num_classes=7,
        enable_fairdisco=False
    )

    print("\nModel info:")
    info = model.get_model_info()
    for k, v in info.items():
        if k == 'parameter_breakdown':
            print(f"  {k}:")
            for pk, pv in v.items():
                print(f"    {pk}: {pv:,}")
        else:
            print(f"  {k}: {v}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test gradient flow
    print("\n3. Testing backward pass...")
    loss = output.sum()
    loss.backward()
    print("Backward pass successful!")

    # Test with FairDisCo
    print("\n4. Creating hybrid model (with FairDisCo)...")
    model_fairdisco = create_hybrid_model(
        convnext_variant='base',
        swin_variant='small',
        num_classes=7,
        enable_fairdisco=True,
        num_fst_classes=6
    )

    print("\n5. Testing FairDisCo forward pass...")
    diag_logits, fst_logits, cont_emb = model_fairdisco(x)
    print(f"Diagnosis logits: {diag_logits.shape}")
    print(f"FST logits: {fst_logits.shape}")
    print(f"Contrastive embeddings: {cont_emb.shape}")

    # Test gradient flow with FairDisCo
    print("\n6. Testing FairDisCo backward pass...")
    loss = diag_logits.sum() + fst_logits.sum() + cont_emb.sum()
    loss.backward()
    print("FairDisCo backward pass successful!")

    print("\n" + "=" * 80)
    print("Hybrid model test PASSED!")
    print("=" * 80)
