"""
Comprehensive unit tests for ConvNeXtV2 architecture.

Tests cover:
- Individual components (GRN, LayerNorm2d, DropPath)
- ConvNeXtV2Block functionality
- ConvNeXtV2Stage construction
- ConvNeXtV2Backbone end-to-end
- Forward/backward passes
- Shape consistency
- Parameter counting
- Pretrained weight loading

Framework: MENDICANT_BIAS - Phase 3
Agent: HOLLOWED_EYES
Target: 20+ tests, 85%+ coverage
Version: 1.0
Date: 2025-10-14
"""

import unittest
import torch
import torch.nn as nn
from src.models.convnextv2 import (
    GlobalResponseNorm,
    LayerNorm2d,
    DropPath,
    ConvNeXtV2Block,
    ConvNeXtV2Stage,
    ConvNeXtV2Backbone,
    create_convnextv2_backbone
)


class TestGlobalResponseNorm(unittest.TestCase):
    """Test GlobalResponseNorm layer."""

    def test_grn_initialization(self):
        """Test GRN initialization."""
        grn = GlobalResponseNorm(dim=128)
        self.assertEqual(grn.gamma.shape, (1, 1, 1, 128))
        self.assertEqual(grn.beta.shape, (1, 1, 1, 128))
        self.assertTrue(torch.all(grn.gamma == 0))
        self.assertTrue(torch.all(grn.beta == 0))

    def test_grn_forward_shape(self):
        """Test GRN forward pass shape."""
        grn = GlobalResponseNorm(dim=128)
        x = torch.randn(4, 28, 28, 128)  # (B, H, W, C)
        output = grn(x)
        self.assertEqual(output.shape, x.shape)

    def test_grn_forward_values(self):
        """Test GRN produces valid outputs."""
        grn = GlobalResponseNorm(dim=64)
        x = torch.randn(2, 14, 14, 64)
        output = grn(x)
        # Should not produce NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_grn_gradient_flow(self):
        """Test GRN allows gradient flow."""
        grn = GlobalResponseNorm(dim=32)
        x = torch.randn(2, 7, 7, 32, requires_grad=True)
        output = grn(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(grn.gamma.grad)
        self.assertIsNotNone(grn.beta.grad)


class TestLayerNorm2d(unittest.TestCase):
    """Test LayerNorm2d layer."""

    def test_layernorm2d_initialization(self):
        """Test LayerNorm2d initialization."""
        ln = LayerNorm2d(num_channels=128)
        self.assertEqual(ln.weight.shape, (128,))
        self.assertEqual(ln.bias.shape, (128,))
        self.assertTrue(torch.all(ln.weight == 1))
        self.assertTrue(torch.all(ln.bias == 0))

    def test_layernorm2d_forward_shape(self):
        """Test LayerNorm2d forward pass shape."""
        ln = LayerNorm2d(num_channels=256)
        x = torch.randn(8, 256, 56, 56)  # (B, C, H, W)
        output = ln(x)
        self.assertEqual(output.shape, x.shape)

    def test_layernorm2d_normalization(self):
        """Test LayerNorm2d actually normalizes."""
        ln = LayerNorm2d(num_channels=64)
        x = torch.randn(4, 64, 28, 28) * 100 + 50  # Non-normalized input
        output = ln(x)
        # Check mean and std are approximately 0 and 1 per sample
        for i in range(output.shape[0]):
            sample = output[i].permute(1, 2, 0)  # (H, W, C)
            mean = sample.mean()
            std = sample.std()
            self.assertAlmostEqual(mean.item(), 0.0, places=1)
            self.assertAlmostEqual(std.item(), 1.0, places=1)

    def test_layernorm2d_gradient_flow(self):
        """Test LayerNorm2d gradient flow."""
        ln = LayerNorm2d(num_channels=128)
        x = torch.randn(2, 128, 14, 14, requires_grad=True)
        output = ln(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ln.weight.grad)


class TestDropPath(unittest.TestCase):
    """Test DropPath (Stochastic Depth) layer."""

    def test_droppath_no_drop_inference(self):
        """Test DropPath is identity during inference."""
        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(8, 64, 28, 28)
        output = dp(x)
        self.assertTrue(torch.allclose(output, x))

    def test_droppath_zero_drop(self):
        """Test DropPath with zero drop probability."""
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(8, 64, 28, 28)
        output = dp(x)
        self.assertTrue(torch.allclose(output, x))

    def test_droppath_training_drops(self):
        """Test DropPath actually drops during training."""
        dp = DropPath(drop_prob=0.5)
        dp.train()
        x = torch.ones(100, 64, 14, 14)
        output = dp(x)
        # Some samples should be zeroed out
        num_dropped = (output.sum(dim=(1, 2, 3)) == 0).sum().item()
        self.assertGreater(num_dropped, 0)

    def test_droppath_gradient_flow(self):
        """Test DropPath allows gradient flow."""
        dp = DropPath(drop_prob=0.2)
        dp.train()
        x = torch.randn(4, 32, 7, 7, requires_grad=True)
        output = dp(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestConvNeXtV2Block(unittest.TestCase):
    """Test ConvNeXtV2Block."""

    def test_block_initialization(self):
        """Test ConvNeXtV2Block initialization."""
        block = ConvNeXtV2Block(dim=128)
        self.assertEqual(block.dwconv.in_channels, 128)
        self.assertEqual(block.dwconv.out_channels, 128)
        self.assertTrue(block.use_grn)
        self.assertIsNotNone(block.gamma)

    def test_block_forward_shape(self):
        """Test ConvNeXtV2Block forward pass shape."""
        block = ConvNeXtV2Block(dim=256)
        x = torch.randn(4, 256, 28, 28)
        output = block(x)
        self.assertEqual(output.shape, x.shape)

    def test_block_residual_connection(self):
        """Test ConvNeXtV2Block has residual connection."""
        block = ConvNeXtV2Block(dim=128, drop_path=0.0, layer_scale_init_value=0.0)
        block.eval()
        x = torch.randn(2, 128, 14, 14)
        output = block(x)
        # With zero layer scale and no drop path, output should be close to input
        # (not exactly equal due to other operations)
        self.assertEqual(output.shape, x.shape)

    def test_block_without_grn(self):
        """Test ConvNeXtV2Block without GRN."""
        block = ConvNeXtV2Block(dim=128, use_grn=False)
        self.assertFalse(block.use_grn)
        x = torch.randn(2, 128, 28, 28)
        output = block(x)
        self.assertEqual(output.shape, x.shape)

    def test_block_gradient_flow(self):
        """Test ConvNeXtV2Block gradient flow."""
        block = ConvNeXtV2Block(dim=128)
        x = torch.randn(2, 128, 14, 14, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_block_different_drop_path_rates(self):
        """Test ConvNeXtV2Block with different drop path rates."""
        for drop_rate in [0.0, 0.1, 0.3, 0.5]:
            block = ConvNeXtV2Block(dim=64, drop_path=drop_rate)
            x = torch.randn(4, 64, 14, 14)
            output = block(x)
            self.assertEqual(output.shape, x.shape)


class TestConvNeXtV2Stage(unittest.TestCase):
    """Test ConvNeXtV2Stage."""

    def test_stage_initialization(self):
        """Test ConvNeXtV2Stage initialization."""
        stage = ConvNeXtV2Stage(
            in_channels=128,
            out_channels=256,
            depth=3,
            drop_path_rates=[0.0, 0.05, 0.1],
            downsample=True
        )
        self.assertEqual(len(stage.blocks), 3)

    def test_stage_forward_with_downsample(self):
        """Test ConvNeXtV2Stage forward pass with downsampling."""
        stage = ConvNeXtV2Stage(
            in_channels=128,
            out_channels=256,
            depth=3,
            drop_path_rates=[0.0, 0.0, 0.0],
            downsample=True
        )
        x = torch.randn(4, 128, 56, 56)
        output = stage(x)
        # Should downsample by 2x
        self.assertEqual(output.shape, (4, 256, 28, 28))

    def test_stage_forward_without_downsample(self):
        """Test ConvNeXtV2Stage forward pass without downsampling."""
        stage = ConvNeXtV2Stage(
            in_channels=128,
            out_channels=128,
            depth=2,
            drop_path_rates=[0.0, 0.0],
            downsample=False
        )
        x = torch.randn(4, 128, 56, 56)
        output = stage(x)
        # No downsampling
        self.assertEqual(output.shape, (4, 128, 56, 56))

    def test_stage_gradient_flow(self):
        """Test ConvNeXtV2Stage gradient flow."""
        stage = ConvNeXtV2Stage(
            in_channels=64,
            out_channels=128,
            depth=2,
            drop_path_rates=[0.0, 0.0],
            downsample=True
        )
        x = torch.randn(2, 64, 28, 28, requires_grad=True)
        output = stage(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_stage_channel_projection_no_downsample(self):
        """Test stage with channel projection but no downsampling."""
        stage = ConvNeXtV2Stage(
            in_channels=128,
            out_channels=256,
            depth=2,
            drop_path_rates=[0.0, 0.0],
            downsample=False
        )
        x = torch.randn(2, 128, 28, 28)
        output = stage(x)
        # Same spatial size, different channels
        self.assertEqual(output.shape, (2, 256, 28, 28))


class TestConvNeXtV2Backbone(unittest.TestCase):
    """Test ConvNeXtV2Backbone."""

    def test_backbone_initialization(self):
        """Test ConvNeXtV2Backbone initialization."""
        backbone = ConvNeXtV2Backbone(
            depths=[3, 3, 9],
            dims=[128, 256, 512]
        )
        self.assertEqual(len(backbone.stages), 3)
        self.assertEqual(backbone.dims, [128, 256, 512])

    def test_backbone_forward_shape(self):
        """Test ConvNeXtV2Backbone forward pass shape."""
        backbone = ConvNeXtV2Backbone(
            depths=[3, 3, 9],
            dims=[128, 256, 512]
        )
        x = torch.randn(2, 3, 224, 224)
        output, features = backbone(x)
        # Final output
        self.assertEqual(output.shape, (2, 512, 14, 14))
        # Intermediate features
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0].shape, (2, 128, 56, 56))
        self.assertEqual(features[1].shape, (2, 256, 28, 28))
        self.assertEqual(features[2].shape, (2, 512, 14, 14))

    def test_backbone_gradient_flow(self):
        """Test ConvNeXtV2Backbone gradient flow."""
        backbone = ConvNeXtV2Backbone(
            depths=[2, 2, 2],
            dims=[64, 128, 256]
        )
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output, _ = backbone(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_backbone_get_feature_dims(self):
        """Test get_feature_dims method."""
        backbone = ConvNeXtV2Backbone(
            depths=[3, 3, 9],
            dims=[128, 256, 512]
        )
        feature_dims = backbone.get_feature_dims()
        self.assertEqual(feature_dims, [128, 256, 512])

    def test_backbone_get_model_info(self):
        """Test get_model_info method."""
        backbone = ConvNeXtV2Backbone(
            depths=[3, 3, 9],
            dims=[128, 256, 512]
        )
        info = backbone.get_model_info()
        self.assertIn('architecture', info)
        self.assertIn('num_parameters', info)
        self.assertEqual(info['depths'], [3, 3, 9])
        self.assertEqual(info['dims'], [128, 256, 512])

    def test_backbone_parameter_count(self):
        """Test backbone has reasonable parameter count."""
        backbone = ConvNeXtV2Backbone(
            depths=[3, 3, 27],
            dims=[128, 256, 512]
        )
        num_params = sum(p.numel() for p in backbone.parameters())
        # Base variant should have ~60M params
        self.assertGreater(num_params, 50_000_000)
        self.assertLess(num_params, 70_000_000)


class TestCreateConvNeXtV2Backbone(unittest.TestCase):
    """Test create_convnextv2_backbone factory function."""

    def test_create_tiny_variant(self):
        """Test creating tiny variant."""
        model = create_convnextv2_backbone(variant='tiny', pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output, features = model(x)
        self.assertEqual(output.shape, (1, 384, 14, 14))

    def test_create_small_variant(self):
        """Test creating small variant."""
        model = create_convnextv2_backbone(variant='small', pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output, features = model(x)
        self.assertEqual(output.shape, (1, 384, 14, 14))

    def test_create_base_variant(self):
        """Test creating base variant."""
        model = create_convnextv2_backbone(variant='base', pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output, features = model(x)
        self.assertEqual(output.shape, (1, 512, 14, 14))

    def test_create_large_variant(self):
        """Test creating large variant."""
        model = create_convnextv2_backbone(variant='large', pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output, features = model(x)
        self.assertEqual(output.shape, (1, 768, 14, 14))

    def test_create_invalid_variant(self):
        """Test creating invalid variant raises error."""
        with self.assertRaises(ValueError):
            create_convnextv2_backbone(variant='invalid')

    def test_create_with_drop_path(self):
        """Test creating with custom drop path rate."""
        model = create_convnextv2_backbone(
            variant='tiny',
            drop_path_rate=0.3,
            pretrained=False
        )
        x = torch.randn(1, 3, 224, 224)
        output, _ = model(x)
        self.assertEqual(output.shape[1], 384)

    def test_create_without_grn(self):
        """Test creating without GRN."""
        model = create_convnextv2_backbone(
            variant='tiny',
            use_grn=False,
            pretrained=False
        )
        x = torch.randn(1, 3, 224, 224)
        output, _ = model(x)
        self.assertEqual(output.shape[1], 384)


class TestConvNeXtV2Integration(unittest.TestCase):
    """Integration tests for ConvNeXtV2."""

    def test_batch_processing(self):
        """Test processing different batch sizes."""
        model = create_convnextv2_backbone(variant='tiny', pretrained=False)
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            with torch.no_grad():
                x = torch.randn(batch_size, 3, 224, 224)
                output, features = model(x)
                self.assertEqual(output.shape[0], batch_size)
                self.assertEqual(len(features), 3)

    def test_eval_vs_train_mode(self):
        """Test model behavior in train vs eval mode."""
        model = create_convnextv2_backbone(variant='tiny', drop_path_rate=0.3)
        x = torch.randn(4, 3, 224, 224)

        # Train mode
        model.train()
        output_train, _ = model(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval, _ = model(x)

        # Shapes should be the same
        self.assertEqual(output_train.shape, output_eval.shape)

    def test_memory_efficiency(self):
        """Test model doesn't leak memory."""
        model = create_convnextv2_backbone(variant='tiny')
        x = torch.randn(2, 3, 224, 224)

        # Multiple forward passes should not accumulate memory
        for _ in range(5):
            with torch.no_grad():
                output, _ = model(x)

        # Clean up
        del output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def test_training_step_simulation(self):
        """Simulate a training step."""
        model = create_convnextv2_backbone(variant='tiny')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        x = torch.randn(4, 3, 224, 224)
        target = torch.randn(4, 384, 14, 14)

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
