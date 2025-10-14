"""
ConvNeXtV2: Modern CNN Architecture with Global Response Normalization

Implements ConvNeXtV2 architecture optimized for hybrid model design.
Features:
- Global Response Normalization (GRN) for improved feature quality
- LayerScale for training stability
- Stochastic depth (DropPath) for regularization
- Efficient depthwise separable convolutions

Reference:
"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
https://arxiv.org/abs/2301.00808

Framework: MENDICANT_BIAS - Phase 3
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from collections import OrderedDict


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization (GRN) layer.

    Enhances inter-channel feature competition and provides spatial context aggregation.
    Applied after the first linear layer in ConvNeXt blocks.

    Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"

    Args:
        dim: Number of channels
        eps: Epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GRN normalization.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Normalized tensor of shape (B, H, W, C)
        """
        # Compute global spatial norms
        # x shape: (B, H, W, C)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)  # (B, 1, 1, C)

        # Apply affine transformation
        return self.gamma * (x * nx) + self.beta + x


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels-first 2D tensors.

    Normalizes over the channel dimension with spatial invariance.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Normalized tensor of shape (B, C, H, W)
        """
        # Permute to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth / Drop Path regularization.

    Randomly drops entire samples during training for regularization.
    Identity mapping during inference.

    Args:
        drop_prob: Probability of dropping path
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXtV2 block with GRN and LayerScale.

    Architecture:
        Input (C channels)
        -> 7x7 depthwise conv
        -> LayerNorm
        -> 1x1 conv (expand to 4C)
        -> GELU
        -> GRN
        -> 1x1 conv (project back to C)
        -> LayerScale
        -> DropPath
        -> Residual connection

    Args:
        dim: Number of input/output channels
        drop_path: Stochastic depth rate
        layer_scale_init_value: Initial value for LayerScale
        use_grn: Whether to use Global Response Normalization
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = True
    ):
        super().__init__()

        # Depthwise convolution (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Layer normalization
        self.norm = LayerNorm2d(dim)

        # Pointwise/Inverted-bottleneck layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Expansion
        self.act = nn.GELU()

        # Global Response Normalization
        self.use_grn = use_grn
        if use_grn:
            self.grn = GlobalResponseNorm(4 * dim)

        self.pwconv2 = nn.Linear(4 * dim, dim)  # Projection

        # Layer scale (learnable scalar per channel)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)),
                requires_grad=True
            )
        else:
            self.gamma = None

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        shortcut = x

        # Depthwise conv
        x = self.dwconv(x)
        x = self.norm(x)

        # Permute for linear layers: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # Inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)

        # GRN (operates on (B, H, W, C))
        if self.use_grn:
            x = self.grn(x)

        x = self.pwconv2(x)

        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x

        # Permute back: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Residual connection
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtV2Stage(nn.Module):
    """
    ConvNeXtV2 stage: downsampling + sequence of blocks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        depth: Number of ConvNeXt blocks in stage
        drop_path_rates: List of drop path rates for each block
        layer_scale_init_value: Initial value for LayerScale
        use_grn: Whether to use GRN
        downsample: Whether to downsample at start of stage
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path_rates: List[float],
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = True,
        downsample: bool = True
    ):
        super().__init__()

        # Downsampling layer (2x2 conv with stride 2, or identity)
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
        else:
            # First stage: no downsampling, just channel projection if needed
            if in_channels != out_channels:
                self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.downsample = nn.Identity()

        # Stack of ConvNeXtV2 blocks
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(
                dim=out_channels,
                drop_path=drop_path_rates[i],
                layer_scale_init_value=layer_scale_init_value,
                use_grn=use_grn
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C_in, H, W)

        Returns:
            Output tensor of shape (B, C_out, H/2, W/2) if downsample else (B, C_out, H, W)
        """
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXtV2 backbone for hybrid architecture.

    Extracts local features through first 3 stages (replaces stage 4 with Swin Transformer).

    Architecture variants:
    - Tiny: dims=[96, 192, 384], depths=[3, 3, 9]
    - Small: dims=[96, 192, 384], depths=[3, 3, 27]
    - Base: dims=[128, 256, 512], depths=[3, 3, 27]
    - Large: dims=[192, 384, 768], depths=[3, 3, 27]

    Args:
        in_channels: Number of input channels (3 for RGB)
        stem_channels: Number of channels after stem
        depths: Number of blocks per stage [stage1, stage2, stage3]
        dims: Number of channels per stage [stage1, stage2, stage3]
        drop_path_rate: Stochastic depth rate (linearly decays)
        layer_scale_init_value: Initial value for LayerScale
        use_grn: Whether to use GRN
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 128,
        depths: List[int] = [3, 3, 9],
        dims: List[int] = [128, 256, 512],
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = True
    ):
        super().__init__()

        assert len(depths) == 3, "ConvNeXtV2 backbone expects 3 stages"
        assert len(dims) == 3, "ConvNeXtV2 backbone expects 3 dimension values"

        self.depths = depths
        self.dims = dims

        # Stem: 4x4 conv with stride 4 (aggressive downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=4, stride=4),
            LayerNorm2d(stem_channels)
        )

        # Compute drop path rates (stochastic depth)
        # Linearly increase from 0 to drop_path_rate across all blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Build stages
        self.stages = nn.ModuleList()

        # Stage 1: No downsampling (already done in stem)
        dp_rates_stage1 = dpr[0:depths[0]]
        self.stages.append(
            ConvNeXtV2Stage(
                in_channels=stem_channels,
                out_channels=dims[0],
                depth=depths[0],
                drop_path_rates=dp_rates_stage1,
                layer_scale_init_value=layer_scale_init_value,
                use_grn=use_grn,
                downsample=False
            )
        )

        # Stage 2: Downsample 2x
        dp_rates_stage2 = dpr[depths[0]:depths[0]+depths[1]]
        self.stages.append(
            ConvNeXtV2Stage(
                in_channels=dims[0],
                out_channels=dims[1],
                depth=depths[1],
                drop_path_rates=dp_rates_stage2,
                layer_scale_init_value=layer_scale_init_value,
                use_grn=use_grn,
                downsample=True
            )
        )

        # Stage 3: Downsample 2x
        dp_rates_stage3 = dpr[depths[0]+depths[1]:total_depth]
        self.stages.append(
            ConvNeXtV2Stage(
                in_channels=dims[1],
                out_channels=dims[2],
                depth=depths[2],
                drop_path_rates=dp_rates_stage3,
                layer_scale_init_value=layer_scale_init_value,
                use_grn=use_grn,
                downsample=True
            )
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights following ConvNeXt paper."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through ConvNeXtV2 backbone.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Tuple of:
                - Final output tensor of shape (B, dims[2], 28, 28)
                - List of intermediate features from each stage
        """
        # Stem: 224 -> 56
        x = self.stem(x)  # (B, stem_channels, 56, 56)

        # Collect multi-scale features
        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            # Stage 1: (B, dims[0], 56, 56)
            # Stage 2: (B, dims[1], 28, 28)
            # Stage 3: (B, dims[2], 14, 14)

        return x, features

    def get_feature_dims(self) -> List[int]:
        """Get output dimensions for each stage."""
        return self.dims

    def get_model_info(self) -> dict:
        """Get model configuration information."""
        return {
            'architecture': 'ConvNeXtV2Backbone',
            'depths': self.depths,
            'dims': self.dims,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_convnextv2_backbone(
    variant: str = 'base',
    drop_path_rate: float = 0.1,
    use_grn: bool = True,
    pretrained: bool = False
) -> ConvNeXtV2Backbone:
    """
    Create ConvNeXtV2 backbone with predefined configurations.

    Args:
        variant: Model variant ('tiny', 'small', 'base', 'large')
        drop_path_rate: Stochastic depth rate
        use_grn: Whether to use GRN
        pretrained: Whether to load pretrained weights (from timm if available)

    Returns:
        ConvNeXtV2Backbone instance
    """
    configs = {
        'tiny': {
            'stem_channels': 96,
            'depths': [3, 3, 9],
            'dims': [96, 192, 384]
        },
        'small': {
            'stem_channels': 96,
            'depths': [3, 3, 27],
            'dims': [96, 192, 384]
        },
        'base': {
            'stem_channels': 128,
            'depths': [3, 3, 27],
            'dims': [128, 256, 512]
        },
        'large': {
            'stem_channels': 192,
            'depths': [3, 3, 27],
            'dims': [192, 384, 768]
        }
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    config = configs[variant]

    model = ConvNeXtV2Backbone(
        in_channels=3,
        stem_channels=config['stem_channels'],
        depths=config['depths'],
        dims=config['dims'],
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=1e-6,
        use_grn=use_grn
    )

    if pretrained:
        # Try to load pretrained weights from timm
        try:
            import timm
            print(f"Loading pretrained ConvNeXtV2-{variant} weights from timm...")
            timm_model = timm.create_model(f'convnextv2_{variant}.fcmae_ft_in1k', pretrained=True)

            # Transfer compatible weights
            model_dict = model.state_dict()
            pretrained_dict = timm_model.state_dict()

            # Filter and transfer weights
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v

            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict)

            print(f"Loaded {len(compatible_dict)}/{len(model_dict)} parameters from pretrained model")
        except ImportError:
            print("Warning: timm not installed. Install with 'pip install timm' for pretrained weights.")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    return model


if __name__ == "__main__":
    """Test ConvNeXtV2 backbone."""
    print("=" * 80)
    print("Testing ConvNeXtV2 Backbone")
    print("=" * 80)

    # Test model creation
    print("\n1. Creating ConvNeXtV2-base backbone...")
    model = create_convnextv2_backbone(variant='base', pretrained=False)

    print(f"\nModel info:")
    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {x.shape}")

    output, features = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of feature maps: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Stage {i+1} features: {feat.shape}")

    # Test gradient flow
    print("\n3. Testing backward pass...")
    loss = output.sum()
    loss.backward()
    print("Backward pass successful!")

    # Test all variants
    print("\n4. Testing all variants...")
    for variant in ['tiny', 'small', 'base', 'large']:
        model = create_convnextv2_backbone(variant=variant)
        output, _ = model(x)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  {variant:10s}: output={output.shape}, params={num_params:,}")

    print("\n" + "=" * 80)
    print("ConvNeXtV2 backbone test PASSED!")
    print("=" * 80)
