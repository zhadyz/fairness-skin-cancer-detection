"""
CIRCLe Color Transformations: LAB Color Space Skin Tone Transformations

Implements skin tone transformations in CIELAB color space for CIRCLe regularization.
Uses perceptually-uniform LAB space to transform images between different Fitzpatrick
Skin Types (FST I-VI) for tone-invariant representation learning.

Based on: Pakzad et al. (2022) "CIRCLe: Color Invariant Representation Learning
for Unbiased Classification of Skin Lesions" ECCV 2022 Workshops

Clean-room implementation from research documentation only.

Approach: Simple LAB adjustments (Phase 2 implementation)
- L* channel: Lightness adjustment (primary skin tone characteristic)
- a* channel: Green-Red axis adjustment (minor)
- b* channel: Blue-Yellow axis adjustment (minor)

Framework: MENDICANT_BIAS - Phase 2, Week 7-8
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
from PIL import Image
import warnings


# FST Color Statistics (from research documentation)
# L* ranges from 0 (black) to 100 (white)
# a* ranges from -128 (green) to 127 (red)
# b* ranges from -128 (blue) to 127 (yellow)
FST_COLOR_STATS = {
    1: {"L_mean": 70.5, "a_mean": 10.2, "b_mean": 18.3},  # FST I (very light)
    2: {"L_mean": 65.0, "a_mean": 12.5, "b_mean": 20.1},  # FST II (light)
    3: {"L_mean": 58.5, "a_mean": 14.8, "b_mean": 22.0},  # FST III (light-medium)
    4: {"L_mean": 48.0, "a_mean": 16.5, "b_mean": 24.5},  # FST IV (medium)
    5: {"L_mean": 38.5, "a_mean": 18.0, "b_mean": 26.0},  # FST V (dark)
    6: {"L_mean": 28.0, "a_mean": 19.5, "b_mean": 28.0},  # FST VI (very dark)
}


class LABColorTransform(nn.Module):
    """
    LAB Color Space Transformation for Skin Tone Augmentation.

    Transforms images from source FST to target FST using CIELAB color space
    adjustments. LAB space is perceptually uniform, making it ideal for
    skin tone manipulations.

    The transformation computes delta shifts in L*, a*, b* channels based on
    FST color statistics and applies them to the entire image.

    Args:
        normalize_input: Whether input is normalized ([0,1] or ImageNet stats)
        imagenet_normalized: Whether input uses ImageNet normalization
        clamp_output: Whether to clamp output to valid range
        device: Device for computation (cuda or cpu)

    Example:
        >>> transform = LABColorTransform()
        >>> # Transform FST III images to FST I (lighten)
        >>> images_fst1 = transform(images, source_fst=3, target_fst=1)
        >>> # Transform FST III images to FST VI (darken)
        >>> images_fst6 = transform(images, source_fst=3, target_fst=6)
    """

    def __init__(
        self,
        normalize_input: bool = True,
        imagenet_normalized: bool = True,
        clamp_output: bool = True,
        device: str = "cuda"
    ):
        super(LABColorTransform, self).__init__()
        self.normalize_input = normalize_input
        self.imagenet_normalized = imagenet_normalized
        self.clamp_output = clamp_output
        self.device = device

        # ImageNet normalization statistics
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # RGB to XYZ conversion matrix (sRGB, D65 illuminant)
        # Based on ITU-R BT.709 standard
        self.rgb_to_xyz = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32)

        # XYZ to RGB conversion matrix (inverse)
        self.xyz_to_rgb = torch.tensor([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ], dtype=torch.float32)

        # D65 reference white point
        self.ref_white = torch.tensor([95.047, 100.000, 108.883], dtype=torch.float32)

    def forward(
        self,
        images: torch.Tensor,
        source_fst: Union[int, torch.Tensor],
        target_fst: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Transform images from source FST to target FST.

        Args:
            images: Input images (B, 3, H, W) in range [0,1] or ImageNet normalized
            source_fst: Source Fitzpatrick skin type (1-6) or tensor of FST labels
            target_fst: Target Fitzpatrick skin type (1-6) or tensor of FST labels

        Returns:
            Transformed images (B, 3, H, W) in same normalization as input
        """
        # Handle batch of different source/target FSTs
        if isinstance(source_fst, torch.Tensor) and isinstance(target_fst, torch.Tensor):
            return self._transform_batch_mixed(images, source_fst, target_fst)
        else:
            # Single FST pair for entire batch
            return self._transform_batch_uniform(images, int(source_fst), int(target_fst))

    def _transform_batch_uniform(
        self,
        images: torch.Tensor,
        source_fst: int,
        target_fst: int
    ) -> torch.Tensor:
        """Transform entire batch with uniform source/target FST."""
        # Denormalize if needed
        if self.imagenet_normalized:
            images = self._denormalize_imagenet(images)

        # Convert RGB → LAB
        lab = self._rgb_to_lab(images)

        # Compute LAB deltas
        delta_L = FST_COLOR_STATS[target_fst]["L_mean"] - FST_COLOR_STATS[source_fst]["L_mean"]
        delta_a = FST_COLOR_STATS[target_fst]["a_mean"] - FST_COLOR_STATS[source_fst]["a_mean"]
        delta_b = FST_COLOR_STATS[target_fst]["b_mean"] - FST_COLOR_STATS[source_fst]["b_mean"]

        # Apply shifts to LAB channels
        lab[:, 0, :, :] += delta_L  # L* channel
        lab[:, 1, :, :] += delta_a  # a* channel
        lab[:, 2, :, :] += delta_b  # b* channel

        # Convert LAB → RGB
        rgb = self._lab_to_rgb(lab)

        # Clamp to valid range
        if self.clamp_output:
            rgb = torch.clamp(rgb, 0.0, 1.0)

        # Renormalize if needed
        if self.imagenet_normalized:
            rgb = self._normalize_imagenet(rgb)

        return rgb

    def _transform_batch_mixed(
        self,
        images: torch.Tensor,
        source_fst: torch.Tensor,
        target_fst: torch.Tensor
    ) -> torch.Tensor:
        """Transform batch with per-sample source/target FST."""
        batch_size = images.size(0)
        transformed = torch.zeros_like(images)

        # Denormalize if needed
        if self.imagenet_normalized:
            images = self._denormalize_imagenet(images)

        # Convert RGB → LAB
        lab = self._rgb_to_lab(images)

        # Apply per-sample transformations
        for i in range(batch_size):
            src_fst = int(source_fst[i].item())
            tgt_fst = int(target_fst[i].item())

            delta_L = FST_COLOR_STATS[tgt_fst]["L_mean"] - FST_COLOR_STATS[src_fst]["L_mean"]
            delta_a = FST_COLOR_STATS[tgt_fst]["a_mean"] - FST_COLOR_STATS[src_fst]["a_mean"]
            delta_b = FST_COLOR_STATS[tgt_fst]["b_mean"] - FST_COLOR_STATS[src_fst]["b_mean"]

            lab[i, 0, :, :] += delta_L
            lab[i, 1, :, :] += delta_a
            lab[i, 2, :, :] += delta_b

        # Convert LAB → RGB
        rgb = self._lab_to_rgb(lab)

        # Clamp to valid range
        if self.clamp_output:
            rgb = torch.clamp(rgb, 0.0, 1.0)

        # Renormalize if needed
        if self.imagenet_normalized:
            rgb = self._normalize_imagenet(rgb)

        return rgb

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to LAB color space.

        Pipeline: RGB → linear RGB → XYZ → LAB

        Args:
            rgb: RGB images (B, 3, H, W) in range [0, 1]

        Returns:
            LAB images (B, 3, H, W)
            - L*: [0, 100]
            - a*: [-128, 127]
            - b*: [-128, 127]
        """
        # Move matrices to device if needed
        if self.rgb_to_xyz.device != rgb.device:
            self.rgb_to_xyz = self.rgb_to_xyz.to(rgb.device)
            self.ref_white = self.ref_white.to(rgb.device)

        # 1. Apply inverse gamma correction (sRGB → linear RGB)
        linear_rgb = self._gamma_expansion(rgb)

        # 2. Convert linear RGB → XYZ
        # Reshape for matrix multiplication: (B, 3, H, W) → (B, H*W, 3)
        B, C, H, W = linear_rgb.shape
        linear_rgb_flat = linear_rgb.permute(0, 2, 3, 1).reshape(B * H * W, 3)

        # Matrix multiplication
        xyz_flat = torch.matmul(linear_rgb_flat, self.rgb_to_xyz.T)

        # Reshape back: (B*H*W, 3) → (B, 3, H, W)
        xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # Scale XYZ to [0, 100] range
        xyz = xyz * 100.0

        # 3. Convert XYZ → LAB
        lab = self._xyz_to_lab(xyz)

        return lab

    def _lab_to_rgb(self, lab: torch.Tensor) -> torch.Tensor:
        """
        Convert LAB to RGB color space.

        Pipeline: LAB → XYZ → linear RGB → RGB

        Args:
            lab: LAB images (B, 3, H, W)

        Returns:
            RGB images (B, 3, H, W) in range [0, 1]
        """
        # Move matrices to device if needed
        if self.xyz_to_rgb.device != lab.device:
            self.xyz_to_rgb = self.xyz_to_rgb.to(lab.device)
            self.ref_white = self.ref_white.to(lab.device)

        # 1. Convert LAB → XYZ
        xyz = self._lab_to_xyz(lab)

        # Scale XYZ from [0, 100] to [0, 1]
        xyz = xyz / 100.0

        # 2. Convert XYZ → linear RGB
        B, C, H, W = xyz.shape
        xyz_flat = xyz.permute(0, 2, 3, 1).reshape(B * H * W, 3)

        # Matrix multiplication
        linear_rgb_flat = torch.matmul(xyz_flat, self.xyz_to_rgb.T)

        # Reshape back
        linear_rgb = linear_rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # 3. Apply gamma correction (linear RGB → sRGB)
        rgb = self._gamma_compression(linear_rgb)

        return rgb

    def _xyz_to_lab(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to LAB using standard formulae."""
        # Normalize by reference white
        xyz_normalized = xyz / self.ref_white.view(1, 3, 1, 1)

        # Apply nonlinear transformation
        # f(t) = t^(1/3) if t > (6/29)^3, else (1/3)*(29/6)^2 * t + 4/29
        epsilon = 0.008856  # (6/29)^3
        kappa = 903.3  # (29/3)^3

        f_xyz = torch.where(
            xyz_normalized > epsilon,
            torch.pow(xyz_normalized, 1.0 / 3.0),
            (kappa * xyz_normalized + 16.0) / 116.0
        )

        # Compute LAB
        L = 116.0 * f_xyz[:, 1, :, :] - 16.0  # Y channel
        a = 500.0 * (f_xyz[:, 0, :, :] - f_xyz[:, 1, :, :])  # X - Y
        b = 200.0 * (f_xyz[:, 1, :, :] - f_xyz[:, 2, :, :])  # Y - Z

        lab = torch.stack([L, a, b], dim=1)
        return lab

    def _lab_to_xyz(self, lab: torch.Tensor) -> torch.Tensor:
        """Convert LAB to XYZ using standard formulae."""
        L = lab[:, 0, :, :]
        a = lab[:, 1, :, :]
        b = lab[:, 2, :, :]

        # Compute f(Y), f(X), f(Z)
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        # Apply inverse nonlinear transformation
        epsilon = 0.008856
        kappa = 903.3

        def f_inv(t):
            return torch.where(
                torch.pow(t, 3.0) > epsilon,
                torch.pow(t, 3.0),
                (116.0 * t - 16.0) / kappa
            )

        xyz_normalized = torch.stack([f_inv(fx), f_inv(fy), f_inv(fz)], dim=1)

        # Denormalize by reference white
        xyz = xyz_normalized * self.ref_white.view(1, 3, 1, 1)

        return xyz

    def _gamma_expansion(self, rgb: torch.Tensor) -> torch.Tensor:
        """sRGB gamma expansion (sRGB → linear RGB)."""
        return torch.where(
            rgb <= 0.04045,
            rgb / 12.92,
            torch.pow((rgb + 0.055) / 1.055, 2.4)
        )

    def _gamma_compression(self, linear_rgb: torch.Tensor) -> torch.Tensor:
        """sRGB gamma compression (linear RGB → sRGB)."""
        return torch.where(
            linear_rgb <= 0.0031308,
            linear_rgb * 12.92,
            1.055 * torch.pow(linear_rgb, 1.0 / 2.4) - 0.055
        )

    def _denormalize_imagenet(self, images: torch.Tensor) -> torch.Tensor:
        """Denormalize ImageNet-normalized images to [0, 1]."""
        mean = self.imagenet_mean.to(images.device)
        std = self.imagenet_std.to(images.device)
        return images * std + mean

    def _normalize_imagenet(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images to ImageNet statistics."""
        mean = self.imagenet_mean.to(images.device)
        std = self.imagenet_std.to(images.device)
        return (images - mean) / std


def apply_fst_transformation(
    image: torch.Tensor,
    source_fst: int,
    target_fst: int,
    imagenet_normalized: bool = True
) -> torch.Tensor:
    """
    Convenience function to transform a single image or batch.

    Args:
        image: Input image (3, H, W) or (B, 3, H, W)
        source_fst: Source Fitzpatrick skin type (1-6)
        target_fst: Target Fitzpatrick skin type (1-6)
        imagenet_normalized: Whether input uses ImageNet normalization

    Returns:
        Transformed image with same shape as input

    Example:
        >>> image = torch.randn(3, 224, 224)  # Single image
        >>> transformed = apply_fst_transformation(image, source_fst=3, target_fst=1)
        >>> print(transformed.shape)  # torch.Size([3, 224, 224])
    """
    transform = LABColorTransform(imagenet_normalized=imagenet_normalized)

    # Handle single image (add batch dimension)
    if image.dim() == 3:
        image = image.unsqueeze(0)
        transformed = transform(image, source_fst, target_fst)
        return transformed.squeeze(0)
    else:
        return transform(image, source_fst, target_fst)


def batch_transform_dataset(
    images: torch.Tensor,
    fst_labels: torch.Tensor,
    target_fsts: List[int] = [1, 6],
    imagenet_normalized: bool = True
) -> Dict[int, torch.Tensor]:
    """
    Transform entire dataset to multiple target FSTs.

    Useful for pre-computing transformations before training.

    Args:
        images: Input images (B, 3, H, W)
        fst_labels: FST labels for each image (B,)
        target_fsts: List of target FST classes to transform to
        imagenet_normalized: Whether input uses ImageNet normalization

    Returns:
        Dictionary mapping target_fst → transformed images

    Example:
        >>> images = torch.randn(100, 3, 224, 224)
        >>> fst_labels = torch.randint(1, 7, (100,))
        >>> transformed_dict = batch_transform_dataset(images, fst_labels, target_fsts=[1, 6])
        >>> print(transformed_dict[1].shape)  # torch.Size([100, 3, 224, 224])
    """
    transform = LABColorTransform(imagenet_normalized=imagenet_normalized)
    results = {}

    for target_fst in target_fsts:
        # Create target_fst tensor for entire batch
        target_fst_tensor = torch.full_like(fst_labels, target_fst)

        # Transform
        transformed = transform(images, fst_labels, target_fst_tensor)
        results[target_fst] = transformed

    return results


def visualize_transformation(
    image: torch.Tensor,
    source_fst: int,
    target_fsts: List[int] = [1, 2, 3, 4, 5, 6],
    imagenet_normalized: bool = True
) -> List[torch.Tensor]:
    """
    Visualize image transformed to all FST classes.

    Args:
        image: Input image (3, H, W)
        source_fst: Source Fitzpatrick skin type
        target_fsts: List of target FSTs to transform to
        imagenet_normalized: Whether input uses ImageNet normalization

    Returns:
        List of transformed images (one per target FST)

    Example:
        >>> image = torch.randn(3, 224, 224)
        >>> transformations = visualize_transformation(image, source_fst=3)
        >>> # transformations[0] = FST I version
        >>> # transformations[5] = FST VI version
    """
    results = []
    for target_fst in target_fsts:
        transformed = apply_fst_transformation(
            image, source_fst, target_fst, imagenet_normalized
        )
        results.append(transformed)
    return results


if __name__ == "__main__":
    """Test LAB color transformations."""
    print("=" * 80)
    print("Testing LAB Color Transformations")
    print("=" * 80)

    # Test 1: Single image transformation
    print("\n1. Testing single image transformation...")
    image = torch.randn(3, 224, 224)
    transformed = apply_fst_transformation(image, source_fst=3, target_fst=1)
    print(f"   Input shape: {image.shape}")
    print(f"   Output shape: {transformed.shape}")
    print(f"   Input range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"   Output range: [{transformed.min():.2f}, {transformed.max():.2f}]")

    # Test 2: Batch transformation
    print("\n2. Testing batch transformation...")
    images = torch.randn(8, 3, 224, 224)
    transform = LABColorTransform(imagenet_normalized=False)
    transformed = transform(images, source_fst=3, target_fst=6)
    print(f"   Batch shape: {transformed.shape}")
    assert transformed.shape == images.shape

    # Test 3: Mixed FST batch transformation
    print("\n3. Testing mixed FST batch transformation...")
    source_fsts = torch.randint(1, 7, (8,))
    target_fsts = torch.randint(1, 7, (8,))
    transformed = transform(images, source_fsts, target_fsts)
    print(f"   Source FSTs: {source_fsts.tolist()}")
    print(f"   Target FSTs: {target_fsts.tolist()}")
    print(f"   Output shape: {transformed.shape}")

    # Test 4: RGB → LAB → RGB round-trip
    print("\n4. Testing RGB → LAB → RGB round-trip...")
    original = torch.rand(2, 3, 64, 64)  # Smaller for speed
    lab = transform._rgb_to_lab(original)
    reconstructed = transform._lab_to_rgb(lab)
    error = torch.abs(original - reconstructed).mean()
    print(f"   Mean reconstruction error: {error.item():.6f}")
    print(f"   Max reconstruction error: {torch.abs(original - reconstructed).max().item():.6f}")

    # Test 5: Batch transform dataset
    print("\n5. Testing batch_transform_dataset...")
    images = torch.randn(16, 3, 224, 224)
    fst_labels = torch.randint(1, 7, (16,))
    transformed_dict = batch_transform_dataset(
        images, fst_labels, target_fsts=[1, 6], imagenet_normalized=False
    )
    print(f"   Transformed to FST I: {transformed_dict[1].shape}")
    print(f"   Transformed to FST VI: {transformed_dict[6].shape}")

    # Test 6: Visualization helper
    print("\n6. Testing visualize_transformation...")
    image = torch.randn(3, 224, 224)
    transformations = visualize_transformation(
        image, source_fst=3, target_fsts=[1, 6], imagenet_normalized=False
    )
    print(f"   Number of transformations: {len(transformations)}")
    print(f"   Each transformation shape: {transformations[0].shape}")

    print("\n" + "=" * 80)
    print("LAB color transformations test PASSED!")
    print("=" * 80)
