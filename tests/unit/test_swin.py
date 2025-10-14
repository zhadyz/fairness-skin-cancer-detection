"""
Comprehensive unit tests for Swin Transformer architecture.

Tests cover:
- Window partition/reverse operations
- WindowAttention module
- SwinTransformerBlock functionality
- PatchMerging layer
- SwinTransformerStage construction
- SwinBackbone end-to-end
- Forward/backward passes
- Shape consistency
- Attention masking

Framework: MENDICANT_BIAS - Phase 3
Agent: HOLLOWED_EYES
Target: 25+ tests, 85%+ coverage
Version: 1.0
Date: 2025-10-14
"""

import unittest
import torch
import torch.nn as nn
from src.models.swin_transformer import (
    window_partition,
    window_reverse,
    WindowAttention,
    SwinTransformerBlock,
    PatchMerging,
    SwinTransformerStage,
    SwinBackbone,
    create_swin_backbone
)


class TestWindowPartition(unittest.TestCase):
    """Test window partition and reverse operations."""

    def test_window_partition_shape(self):
        """Test window partition produces correct shape."""
        x = torch.randn(2, 14, 14, 128)
        windows = window_partition(x, window_size=7)
        # 14x14 image with window size 7 = 2x2 = 4 windows per image
        # 2 images * 4 windows = 8 total windows
        self.assertEqual(windows.shape, (8, 7, 7, 128))

    def test_window_reverse_shape(self):
        """Test window reverse produces correct shape."""
        windows = torch.randn(8, 7, 7, 128)
        x = window_reverse(windows, window_size=7, H=14, W=14)
        self.assertEqual(x.shape, (2, 14, 14, 128))

    def test_window_partition_reverse_identity(self):
        """Test that partition followed by reverse is identity."""
        x = torch.randn(4, 28, 28, 64)
        windows = window_partition(x, window_size=7)
        x_recovered = window_reverse(windows, window_size=7, H=28, W=28)
        self.assertTrue(torch.allclose(x, x_recovered))

    def test_window_partition_different_sizes(self):
        """Test window partition with different window sizes."""
        x = torch.randn(2, 28, 28, 128)
        for ws in [7, 14, 28]:
            windows = window_partition(x, window_size=ws)
            num_windows = (28 // ws) ** 2
            self.assertEqual(windows.shape, (2 * num_windows, ws, ws, 128))


class TestWindowAttention(unittest.TestCase):
    """Test WindowAttention module."""

    def test_window_attention_initialization(self):
        """Test WindowAttention initialization."""
        attn = WindowAttention(dim=128, window_size=(7, 7), num_heads=4)
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.dim, 128)
        # Relative position bias table size
        self.assertEqual(attn.relative_position_bias_table.shape, ((2*7-1)*(2*7-1), 4))

    def test_window_attention_forward_shape(self):
        """Test WindowAttention forward pass shape."""
        attn = WindowAttention(dim=128, window_size=(7, 7), num_heads=4)
        x = torch.randn(8, 49, 128)  # 8 windows, 49 tokens (7x7), 128 channels
        output = attn(x)
        self.assertEqual(output.shape, x.shape)

    def test_window_attention_with_mask(self):
        """Test WindowAttention with attention mask."""
        attn = WindowAttention(dim=64, window_size=(7, 7), num_heads=2)
        x = torch.randn(4, 49, 64)
        mask = torch.zeros(4, 49, 49)
        output = attn(x, mask=mask)
        self.assertEqual(output.shape, x.shape)

    def test_window_attention_gradient_flow(self):
        """Test WindowAttention gradient flow."""
        attn = WindowAttention(dim=128, window_size=(7, 7), num_heads=4)
        x = torch.randn(4, 49, 128, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_window_attention_different_heads(self):
        """Test WindowAttention with different number of heads."""
        for num_heads in [1, 2, 4, 8]:
            attn = WindowAttention(dim=64, window_size=(7, 7), num_heads=num_heads)
            x = torch.randn(2, 49, 64)
            output = attn(x)
            self.assertEqual(output.shape, x.shape)


class TestSwinTransformerBlock(unittest.TestCase):
    """Test SwinTransformerBlock."""

    def test_block_initialization(self):
        """Test SwinTransformerBlock initialization."""
        block = SwinTransformerBlock(dim=128, num_heads=4, window_size=7)
        self.assertEqual(block.dim, 128)
        self.assertEqual(block.num_heads, 4)
        self.assertEqual(block.window_size, 7)
        self.assertEqual(block.shift_size, 0)

    def test_block_forward_shape(self):
        """Test SwinTransformerBlock forward pass shape."""
        block = SwinTransformerBlock(dim=128, num_heads=4, window_size=7)
        x = torch.randn(2, 196, 128)  # 2 images, 14x14 tokens, 128 channels
        output = block(x, H=14, W=14)
        self.assertEqual(output.shape, x.shape)

    def test_block_with_shift(self):
        """Test SwinTransformerBlock with shifted windows."""
        block = SwinTransformerBlock(dim=128, num_heads=4, window_size=7, shift_size=3)
        x = torch.randn(2, 196, 128)
        output = block(x, H=14, W=14)
        self.assertEqual(output.shape, x.shape)

    def test_block_gradient_flow(self):
        """Test SwinTransformerBlock gradient flow."""
        block = SwinTransformerBlock(dim=64, num_heads=2, window_size=7)
        x = torch.randn(2, 196, 64, requires_grad=True)
        output = block(x, H=14, W=14)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_block_different_mlp_ratios(self):
        """Test SwinTransformerBlock with different MLP ratios."""
        for mlp_ratio in [2.0, 4.0, 8.0]:
            block = SwinTransformerBlock(dim=64, num_heads=2, window_size=7, mlp_ratio=mlp_ratio)
            x = torch.randn(1, 49, 64)
            output = block(x, H=7, W=7)
            self.assertEqual(output.shape, x.shape)

    def test_block_drop_path(self):
        """Test SwinTransformerBlock with drop path."""
        block = SwinTransformerBlock(dim=128, num_heads=4, window_size=7, drop_path=0.1)
        block.train()
        x = torch.randn(2, 196, 128)
        output = block(x, H=14, W=14)
        self.assertEqual(output.shape, x.shape)


class TestPatchMerging(unittest.TestCase):
    """Test PatchMerging layer."""

    def test_patch_merging_initialization(self):
        """Test PatchMerging initialization."""
        pm = PatchMerging(dim=128)
        self.assertEqual(pm.dim, 128)

    def test_patch_merging_forward_shape(self):
        """Test PatchMerging forward pass shape."""
        pm = PatchMerging(dim=128)
        x = torch.randn(2, 196, 128)  # 2 images, 14x14 tokens, 128 channels
        output, H_out, W_out = pm(x, H=14, W=14)
        # Should downsample by 2x and double channels
        self.assertEqual(output.shape, (2, 49, 256))
        self.assertEqual(H_out, 7)
        self.assertEqual(W_out, 7)

    def test_patch_merging_gradient_flow(self):
        """Test PatchMerging gradient flow."""
        pm = PatchMerging(dim=64)
        x = torch.randn(2, 196, 64, requires_grad=True)
        output, _, _ = pm(x, H=14, W=14)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_patch_merging_channel_doubling(self):
        """Test that PatchMerging doubles channels."""
        for dim in [64, 128, 256]:
            pm = PatchMerging(dim=dim)
            x = torch.randn(1, 196, dim)
            output, _, _ = pm(x, H=14, W=14)
            self.assertEqual(output.shape[-1], dim * 2)

    def test_patch_merging_spatial_reduction(self):
        """Test that PatchMerging reduces spatial dimensions by 2x."""
        pm = PatchMerging(dim=128)
        for H, W in [(14, 14), (28, 28), (56, 56)]:
            x = torch.randn(1, H * W, 128)
            output, H_out, W_out = pm(x, H=H, W=W)
            self.assertEqual(H_out, H // 2)
            self.assertEqual(W_out, W // 2)
            self.assertEqual(output.shape[1], H_out * W_out)


class TestSwinTransformerStage(unittest.TestCase):
    """Test SwinTransformerStage."""

    def test_stage_initialization(self):
        """Test SwinTransformerStage initialization."""
        stage = SwinTransformerStage(
            dim=128,
            depth=2,
            num_heads=4,
            window_size=7
        )
        self.assertEqual(len(stage.blocks), 2)
        self.assertEqual(stage.dim, 128)

    def test_stage_forward_with_downsample(self):
        """Test SwinTransformerStage forward pass with downsampling."""
        stage = SwinTransformerStage(
            dim=128,
            depth=2,
            num_heads=4,
            window_size=7,
            downsample=PatchMerging(128)
        )
        x = torch.randn(2, 196, 128)
        output, H_out, W_out = stage(x, H=14, W=14)
        # Should downsample by 2x
        self.assertEqual(H_out, 7)
        self.assertEqual(W_out, 7)
        # Should double channels
        self.assertEqual(output.shape, (2, 49, 256))

    def test_stage_forward_without_downsample(self):
        """Test SwinTransformerStage forward pass without downsampling."""
        stage = SwinTransformerStage(
            dim=128,
            depth=2,
            num_heads=4,
            window_size=7,
            downsample=None
        )
        x = torch.randn(2, 196, 128)
        output, H_out, W_out = stage(x, H=14, W=14)
        # No downsampling
        self.assertEqual(H_out, 14)
        self.assertEqual(W_out, 14)
        self.assertEqual(output.shape, (2, 196, 128))

    def test_stage_gradient_flow(self):
        """Test SwinTransformerStage gradient flow."""
        stage = SwinTransformerStage(
            dim=64,
            depth=2,
            num_heads=2,
            window_size=7
        )
        x = torch.randn(2, 196, 64, requires_grad=True)
        output, _, _ = stage(x, H=14, W=14)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_stage_shifted_windows(self):
        """Test that stage alternates between regular and shifted windows."""
        stage = SwinTransformerStage(
            dim=128,
            depth=4,
            num_heads=4,
            window_size=7
        )
        # Check shift sizes alternate
        self.assertEqual(stage.blocks[0].shift_size, 0)
        self.assertEqual(stage.blocks[1].shift_size, 3)  # window_size // 2
        self.assertEqual(stage.blocks[2].shift_size, 0)
        self.assertEqual(stage.blocks[3].shift_size, 3)


class TestSwinBackbone(unittest.TestCase):
    """Test SwinBackbone."""

    def test_backbone_initialization(self):
        """Test SwinBackbone initialization."""
        backbone = SwinBackbone(
            in_channels=512,
            embed_dim=128,
            depths=[2, 6],
            num_heads=[4, 8]
        )
        self.assertEqual(len(backbone.stages), 2)
        self.assertEqual(backbone.embed_dim, 128)

    def test_backbone_forward_shape(self):
        """Test SwinBackbone forward pass shape."""
        backbone = SwinBackbone(
            in_channels=512,
            embed_dim=128,
            depths=[2, 6],
            num_heads=[4, 8],
            input_resolution=(14, 14)
        )
        x = torch.randn(2, 512, 14, 14)
        output, features = backbone(x)
        # After 2 stages with 1 PatchMerging: 14 -> 7
        self.assertEqual(output.shape, (2, 256, 7, 7))
        self.assertEqual(len(features), 2)

    def test_backbone_gradient_flow(self):
        """Test SwinBackbone gradient flow."""
        backbone = SwinBackbone(
            in_channels=256,
            embed_dim=96,
            depths=[2, 2],
            num_heads=[3, 6],
            input_resolution=(14, 14)
        )
        x = torch.randn(2, 256, 14, 14, requires_grad=True)
        output, _ = backbone(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_backbone_get_model_info(self):
        """Test get_model_info method."""
        backbone = SwinBackbone(
            in_channels=512,
            embed_dim=128,
            depths=[2, 6],
            num_heads=[4, 8]
        )
        info = backbone.get_model_info()
        self.assertIn('architecture', info)
        self.assertIn('num_parameters', info)
        self.assertEqual(info['depths'], [2, 6])
        self.assertEqual(info['num_heads'], [4, 8])

    def test_backbone_single_stage(self):
        """Test SwinBackbone with single stage."""
        backbone = SwinBackbone(
            in_channels=256,
            embed_dim=96,
            depths=[6],
            num_heads=[3],
            input_resolution=(14, 14)
        )
        x = torch.randn(2, 256, 14, 14)
        output, features = backbone(x)
        # No downsampling with single stage
        self.assertEqual(output.shape, (2, 96, 14, 14))
        self.assertEqual(len(features), 1)


class TestCreateSwinBackbone(unittest.TestCase):
    """Test create_swin_backbone factory function."""

    def test_create_tiny_variant(self):
        """Test creating tiny variant."""
        model = create_swin_backbone(variant='tiny', in_channels=384)
        x = torch.randn(1, 384, 14, 14)
        output, features = model(x)
        self.assertEqual(len(features), 2)

    def test_create_small_variant(self):
        """Test creating small variant."""
        model = create_swin_backbone(variant='small', in_channels=512)
        x = torch.randn(1, 512, 14, 14)
        output, features = model(x)
        self.assertEqual(len(features), 2)

    def test_create_base_variant(self):
        """Test creating base variant."""
        model = create_swin_backbone(variant='base', in_channels=768)
        x = torch.randn(1, 768, 14, 14)
        output, features = model(x)
        self.assertEqual(len(features), 2)

    def test_create_invalid_variant(self):
        """Test creating invalid variant raises error."""
        with self.assertRaises(ValueError):
            create_swin_backbone(variant='invalid')

    def test_create_with_custom_resolution(self):
        """Test creating with custom input resolution."""
        model = create_swin_backbone(
            variant='tiny',
            in_channels=256,
            input_resolution=(28, 28)
        )
        x = torch.randn(1, 256, 28, 28)
        output, _ = model(x)
        # After 2 stages with 1 PatchMerging: 28 -> 14
        self.assertEqual(output.shape[2:], (14, 14))


class TestSwinIntegration(unittest.TestCase):
    """Integration tests for Swin Transformer."""

    def test_batch_processing(self):
        """Test processing different batch sizes."""
        model = create_swin_backbone(variant='tiny', in_channels=384)
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            with torch.no_grad():
                x = torch.randn(batch_size, 384, 14, 14)
                output, features = model(x)
                self.assertEqual(output.shape[0], batch_size)

    def test_eval_vs_train_mode(self):
        """Test model behavior in train vs eval mode."""
        model = create_swin_backbone(variant='tiny', in_channels=384, drop_path_rate=0.1)
        x = torch.randn(2, 384, 14, 14)

        # Train mode
        model.train()
        output_train, _ = model(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval, _ = model(x)

        # Shapes should be the same
        self.assertEqual(output_train.shape, output_eval.shape)

    def test_with_convnext_output(self):
        """Test Swin backbone with realistic ConvNeXt output."""
        from src.models.convnextv2 import create_convnextv2_backbone

        # Create ConvNeXtV2
        convnext = create_convnextv2_backbone(variant='base', pretrained=False)
        convnext.eval()

        # Create Swin
        swin = create_swin_backbone(variant='small', in_channels=512, input_resolution=(14, 14))
        swin.eval()

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            convnext_out, _ = convnext(x)
            swin_out, _ = swin(convnext_out)

        # Check shapes
        self.assertEqual(convnext_out.shape, (2, 512, 14, 14))
        self.assertEqual(swin_out.shape, (2, 256, 7, 7))

    def test_training_step_simulation(self):
        """Simulate a training step."""
        model = create_swin_backbone(variant='tiny', in_channels=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        x = torch.randn(2, 256, 14, 14)
        target = torch.randn(2, 192, 7, 7)

        # Forward
        output, _ = model(x)
        loss = criterion(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss is finite
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
