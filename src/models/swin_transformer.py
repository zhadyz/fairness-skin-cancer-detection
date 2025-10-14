"""
Swin Transformer: Hierarchical Vision Transformer with Shifted Windows

Implements Swin Transformer optimized for hybrid architecture.
Features:
- Shifted window multi-head self-attention
- Hierarchical design with patch merging
- Linear complexity relative to image size
- Efficient global receptive field

Reference:
"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
https://arxiv.org/abs/2103.14030

Framework: MENDICANT_BIAS - Phase 3
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition tensor into non-overlapping windows.

    Args:
        x: Input tensor of shape (B, H, W, C)
        window_size: Window size (M)

    Returns:
        Windows tensor of shape (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partitioning back to original tensor.

    Args:
        windows: Windows tensor of shape (num_windows*B, window_size, window_size, C)
        window_size: Window size (M)
        H: Height of original tensor
        W: Width of original tensor

    Returns:
        Tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA) module with relative position bias.

    Args:
        dim: Number of input channels
        window_size: Window size
        num_heads: Number of attention heads
        qkv_bias: If True, add learnable bias to Q, K, V
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        # (2*Wh-1) * (2*Ww-1) possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)

        # Broadcasting to get relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)

        # Shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for window attention.

        Args:
            x: Input tensor of shape (num_windows*B, N, C) where N = window_size * window_size
            mask: Attention mask of shape (num_windows, N, N) or None

        Returns:
            Output tensor of shape (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B_, num_heads, N, head_dim)

        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask if provided (for shifted window attention)
        if mask is not None:
            nW = mask.shape[0]  # Number of windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with shifted window attention.

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        window_size: Window size
        shift_size: Shift size for shifted window attention
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: If True, add learnable bias to Q, K, V
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)

        # Window attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # Drop path
        from src.models.convnextv2 import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer normalization
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor, H: int, W: int, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Swin Transformer block.

        Args:
            x: Input tensor of shape (B, H*W, C)
            H: Height of feature map
            W: Width of feature map
            attn_mask: Attention mask for shifted window attention

        Returns:
            Output tensor of shape (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, window_size*window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (nW*B, window_size*window_size, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # Residual connection
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """
    Patch merging layer that downsamples by 2x and increases channels by 2x.

    Args:
        dim: Number of input channels
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Forward pass for patch merging.

        Args:
            x: Input tensor of shape (B, H*W, C)
            H: Height of feature map
            W: Width of feature map

        Returns:
            Tuple of:
                - Output tensor of shape (B, H/2*W/2, 2*C)
                - New height (H/2)
                - New width (W/2)
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"H ({H}) and W ({W}) must be even for patch merging"

        x = x.view(B, H, W, C)

        # Downsample by gathering 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)

        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4*C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2*C)

        return x, H // 2, W // 2


class SwinTransformerStage(nn.Module):
    """
    Swin Transformer stage: sequence of Swin Transformer blocks.

    Args:
        dim: Number of input channels
        depth: Number of blocks
        num_heads: Number of attention heads
        window_size: Window size
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: If True, add learnable bias to Q, K, V
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate (list or single value)
        downsample: Downsample layer at the end of stage (PatchMerging or None)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] = None,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        if drop_path is None:
            drop_path = [0.0] * depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # Alternate shifting
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
            )
            for i in range(depth)
        ])

        # Patch merging layer
        self.downsample = downsample

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Forward pass for Swin Transformer stage.

        Args:
            x: Input tensor of shape (B, H*W, C)
            H: Height of feature map
            W: Width of feature map

        Returns:
            Tuple of:
                - Output tensor
                - New height
                - New width
        """
        # Create attention mask for shifted window attention
        if any(block.shift_size > 0 for block in self.blocks):
            attn_mask = self._create_attention_mask(H, W)
        else:
            attn_mask = None

        # Apply blocks
        for block in self.blocks:
            x = block(x, H, W, attn_mask)

        # Downsample
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W

    def _create_attention_mask(self, H: int, W: int) -> torch.Tensor:
        """
        Create attention mask for shifted window attention.

        Args:
            H: Height of feature map
            W: Width of feature map

        Returns:
            Attention mask tensor
        """
        # Create mask for SW-MSA (shifted window multi-head self-attention)
        img_mask = torch.zeros((1, H, W, 1))  # (1, H, W, 1)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.window_size // 2),
            slice(-self.window_size // 2, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.window_size // 2),
            slice(-self.window_size // 2, None)
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (nW, window_size, window_size, 1)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # (nW, window_size*window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, 1, window_size*window_size) - (nW, window_size*window_size, 1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone for hybrid architecture.

    Processes features from ConvNeXtV2 and applies global attention.
    Uses 2 stages for computational efficiency.

    Args:
        in_channels: Number of input channels (from ConvNeXtV2)
        embed_dim: Embedding dimension for first stage
        depths: Number of blocks per stage [stage1, stage2]
        num_heads: Number of attention heads per stage [stage1, stage2]
        window_size: Window size for attention
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        qkv_bias: If True, add learnable bias to Q, K, V
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        input_resolution: Input resolution (H, W) from ConvNeXtV2
    """

    def __init__(
        self,
        in_channels: int = 512,  # From ConvNeXtV2 stage 3
        embed_dim: int = 128,
        depths: List[int] = [2, 6],
        num_heads: List[int] = [4, 8],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        input_resolution: Tuple[int, int] = (14, 14)
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.input_resolution = input_resolution

        # Projection from ConvNeXtV2 features to Swin embedding dim
        self.proj = nn.Linear(in_channels, embed_dim)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()

        # Stage 1
        self.stages.append(
            SwinTransformerStage(
                dim=embed_dim,
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depths[0]],
                downsample=PatchMerging(embed_dim) if len(depths) > 1 else None
            )
        )

        # Stage 2 (if exists)
        if len(depths) > 1:
            self.stages.append(
                SwinTransformerStage(
                    dim=embed_dim * 2,  # Doubled by PatchMerging
                    depth=depths[1],
                    num_heads=num_heads[1],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depths[0]:sum(depths)],
                    downsample=None  # No downsampling at the end
                )
            )

        # Final normalization
        final_dim = embed_dim * (2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(final_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through Swin backbone.

        Args:
            x: Input tensor from ConvNeXtV2 of shape (B, C, H, W)

        Returns:
            Tuple of:
                - Final output tensor of shape (B, C_out, H_out, W_out)
                - List of intermediate features
        """
        B, C, H, W = x.shape

        # Permute to (B, H, W, C) and flatten to (B, H*W, C)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, H * W, C)  # (B, H*W, C)

        # Project to embedding dimension
        x = self.proj(x)  # (B, H*W, embed_dim)

        features = []
        current_H, current_W = H, W

        # Apply stages
        for stage in self.stages:
            x, current_H, current_W = stage(x, current_H, current_W)
            features.append(x)

        # Final normalization
        x = self.norm(x)

        # Reshape back to (B, C, H, W)
        final_dim = x.shape[-1]
        x = x.view(B, current_H, current_W, final_dim).permute(0, 3, 1, 2).contiguous()

        return x, features

    def get_model_info(self) -> dict:
        """Get model configuration information."""
        return {
            'architecture': 'SwinBackbone',
            'depths': self.depths,
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_swin_backbone(
    variant: str = 'small',
    in_channels: int = 512,
    drop_path_rate: float = 0.2,
    input_resolution: Tuple[int, int] = (14, 14),
    pretrained: bool = False
) -> SwinBackbone:
    """
    Create Swin Transformer backbone with predefined configurations.

    Args:
        variant: Model variant ('tiny', 'small', 'base')
        in_channels: Number of input channels from ConvNeXtV2
        drop_path_rate: Stochastic depth rate
        input_resolution: Input resolution (H, W)
        pretrained: Whether to load pretrained weights (not implemented)

    Returns:
        SwinBackbone instance
    """
    configs = {
        'tiny': {
            'embed_dim': 96,
            'depths': [2, 2],
            'num_heads': [3, 6]
        },
        'small': {
            'embed_dim': 128,
            'depths': [2, 6],
            'num_heads': [4, 8]
        },
        'base': {
            'embed_dim': 192,
            'depths': [2, 6],
            'num_heads': [6, 12]
        }
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")

    config = configs[variant]

    model = SwinBackbone(
        in_channels=in_channels,
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        input_resolution=input_resolution
    )

    if pretrained:
        print("Warning: Pretrained weights for custom Swin backbone not available.")

    return model


if __name__ == "__main__":
    """Test Swin Transformer backbone."""
    print("=" * 80)
    print("Testing Swin Transformer Backbone")
    print("=" * 80)

    # Test model creation
    print("\n1. Creating Swin-small backbone...")
    model = create_swin_backbone(variant='small', in_channels=512, input_resolution=(14, 14))

    print(f"\nModel info:")
    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test forward pass
    print("\n2. Testing forward pass...")
    x = torch.randn(2, 512, 14, 14)  # Output from ConvNeXtV2 stage 3
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
    for variant in ['tiny', 'small', 'base']:
        model = create_swin_backbone(variant=variant, in_channels=512)
        output, _ = model(torch.randn(2, 512, 14, 14))
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  {variant:10s}: output={output.shape}, params={num_params:,}")

    print("\n" + "=" * 80)
    print("Swin Transformer backbone test PASSED!")
    print("=" * 80)
