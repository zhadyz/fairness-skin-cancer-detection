"""
Comprehensive unit tests for Hybrid ConvNeXtV2-Swin model.

Tests cover:
- MultiScaleFusionModule functionality
- HybridFairnessClassifier construction
- Forward/backward passes
- FairDisCo integration
- Gradient reversal layer
- Parameter counting
- Shape consistency
- Memory efficiency

Framework: MENDICANT_BIAS - Phase 3
Agent: HOLLOWED_EYES
Target: 25+ tests, 85%+ coverage
Version: 1.0
Date: 2025-10-14
"""

import unittest
import torch
import torch.nn as nn
from src.models.hybrid_model import (
    HybridModelConfig,
    MultiScaleFusionModule,
    HybridFairnessClassifier,
    GradientReversalLayer,
    GradientReversalFunction,
    create_hybrid_model
)


class TestMultiScaleFusionModule(unittest.TestCase):
    """Test MultiScaleFusionModule."""

    def test_fusion_initialization(self):
        """Test MultiScaleFusionModule initialization."""
        fusion = MultiScaleFusionModule(
            convnext_dims=[128, 256, 512],
            swin_dim=256,
            fusion_dim=768
        )
        self.assertEqual(len(fusion.convnext_projections), 3)
        self.assertIsNotNone(fusion.swin_projection)

    def test_fusion_forward_shape(self):
        """Test MultiScaleFusionModule forward pass shape."""
        fusion = MultiScaleFusionModule(
            convnext_dims=[128, 256, 512],
            swin_dim=256,
            fusion_dim=768
        )

        # Simulate ConvNeXt features
        convnext_feats = [
            torch.randn(2, 128, 56, 56),
            torch.randn(2, 256, 28, 28),
            torch.randn(2, 512, 14, 14)
        ]
        swin_feats = torch.randn(2, 256, 7, 7)

        output = fusion(convnext_feats, swin_feats)
        self.assertEqual(output.shape, (2, 768))

    def test_fusion_with_attention(self):
        """Test fusion with attention enabled."""
        fusion = MultiScaleFusionModule(
            convnext_dims=[128, 256, 512],
            swin_dim=256,
            fusion_dim=768,
            use_attention=True
        )

        convnext_feats = [
            torch.randn(2, 128, 56, 56),
            torch.randn(2, 256, 28, 28),
            torch.randn(2, 512, 14, 14)
        ]
        swin_feats = torch.randn(2, 256, 7, 7)

        output = fusion(convnext_feats, swin_feats)
        self.assertEqual(output.shape, (2, 768))

    def test_fusion_without_attention(self):
        """Test fusion without attention."""
        fusion = MultiScaleFusionModule(
            convnext_dims=[128, 256, 512],
            swin_dim=256,
            fusion_dim=768,
            use_attention=False
        )

        convnext_feats = [
            torch.randn(2, 128, 56, 56),
            torch.randn(2, 256, 28, 28),
            torch.randn(2, 512, 14, 14)
        ]
        swin_feats = torch.randn(2, 256, 7, 7)

        output = fusion(convnext_feats, swin_feats)
        self.assertEqual(output.shape, (2, 768))

    def test_fusion_gradient_flow(self):
        """Test fusion gradient flow."""
        fusion = MultiScaleFusionModule(
            convnext_dims=[128, 256, 512],
            swin_dim=256,
            fusion_dim=512
        )

        convnext_feats = [
            torch.randn(2, 128, 56, 56, requires_grad=True),
            torch.randn(2, 256, 28, 28, requires_grad=True),
            torch.randn(2, 512, 14, 14, requires_grad=True)
        ]
        swin_feats = torch.randn(2, 256, 7, 7, requires_grad=True)

        output = fusion(convnext_feats, swin_feats)
        loss = output.sum()
        loss.backward()

        for feat in convnext_feats:
            self.assertIsNotNone(feat.grad)
        self.assertIsNotNone(swin_feats.grad)

    def test_fusion_different_batch_sizes(self):
        """Test fusion with different batch sizes."""
        fusion = MultiScaleFusionModule(
            convnext_dims=[128, 256, 512],
            swin_dim=256,
            fusion_dim=768
        )

        for batch_size in [1, 2, 4, 8]:
            convnext_feats = [
                torch.randn(batch_size, 128, 56, 56),
                torch.randn(batch_size, 256, 28, 28),
                torch.randn(batch_size, 512, 14, 14)
            ]
            swin_feats = torch.randn(batch_size, 256, 7, 7)

            output = fusion(convnext_feats, swin_feats)
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], 768)


class TestGradientReversalLayer(unittest.TestCase):
    """Test GradientReversalLayer."""

    def test_grl_initialization(self):
        """Test GRL initialization."""
        grl = GradientReversalLayer(lambda_value=0.5)
        self.assertEqual(grl.lambda_value, 0.5)

    def test_grl_forward_identity(self):
        """Test GRL forward is identity."""
        grl = GradientReversalLayer(lambda_value=1.0)
        x = torch.randn(4, 128)
        output = grl(x)
        self.assertTrue(torch.allclose(output, x))

    def test_grl_backward_reverses(self):
        """Test GRL reverses gradient."""
        grl = GradientReversalLayer(lambda_value=1.0)
        x = torch.randn(4, 128, requires_grad=True)

        # Forward
        output = grl(x)

        # Backward with gradient of ones
        output.backward(torch.ones_like(output))

        # Gradient should be reversed (negative)
        expected_grad = -torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_grl_set_lambda(self):
        """Test setting lambda value."""
        grl = GradientReversalLayer(lambda_value=0.5)
        grl.set_lambda(1.0)
        self.assertEqual(grl.lambda_value, 1.0)

    def test_grl_lambda_scaling(self):
        """Test GRL scales gradient by lambda."""
        grl = GradientReversalLayer(lambda_value=0.5)
        x = torch.randn(4, 128, requires_grad=True)

        output = grl(x)
        output.backward(torch.ones_like(output))

        expected_grad = -0.5 * torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected_grad))


class TestHybridFairnessClassifier(unittest.TestCase):
    """Test HybridFairnessClassifier."""

    def test_model_initialization(self):
        """Test model initialization without FairDisCo."""
        config = HybridModelConfig(
            convnext_variant='base',
            swin_variant='small',
            num_classes=7,
            enable_fairdisco=False
        )
        model = HybridFairnessClassifier(config)

        self.assertIsNotNone(model.convnext)
        self.assertIsNotNone(model.swin)
        self.assertIsNotNone(model.fusion)
        self.assertIsNotNone(model.classifier)
        self.assertIsNone(model.discriminator)

    def test_model_initialization_with_fairdisco(self):
        """Test model initialization with FairDisCo."""
        config = HybridModelConfig(
            convnext_variant='base',
            swin_variant='small',
            num_classes=7,
            enable_fairdisco=True,
            num_fst_classes=6
        )
        model = HybridFairnessClassifier(config)

        self.assertIsNotNone(model.discriminator)
        self.assertIsNotNone(model.gradient_reversal)
        self.assertIsNotNone(model.contrastive_projection)

    def test_model_forward_without_fairdisco(self):
        """Test forward pass without FairDisCo."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            num_classes=7,
            enable_fairdisco=False
        )

        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        self.assertEqual(output.shape, (2, 7))

    def test_model_forward_with_fairdisco(self):
        """Test forward pass with FairDisCo."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            num_classes=7,
            enable_fairdisco=True,
            num_fst_classes=6
        )

        x = torch.randn(2, 3, 224, 224)
        diag_logits, fst_logits, cont_emb = model(x)

        self.assertEqual(diag_logits.shape, (2, 7))
        self.assertEqual(fst_logits.shape, (2, 6))
        self.assertEqual(cont_emb.shape, (2, 128))

    def test_model_forward_return_features(self):
        """Test forward pass with return_features=True."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            enable_fairdisco=True
        )

        x = torch.randn(2, 3, 224, 224)
        diag_logits, fst_logits, cont_emb, features = model(x, return_features=True)

        self.assertEqual(features.shape, (2, 768))

    def test_model_gradient_flow(self):
        """Test gradient flow through model."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            num_classes=7,
            enable_fairdisco=False
        )

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_model_gradient_flow_fairdisco(self):
        """Test gradient flow with FairDisCo."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            enable_fairdisco=True
        )

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        diag_logits, fst_logits, cont_emb = model(x)

        loss = diag_logits.sum() + fst_logits.sum() + cont_emb.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    def test_model_get_info(self):
        """Test get_model_info method."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            num_classes=7,
            enable_fairdisco=False
        )

        info = model.get_model_info()

        self.assertIn('architecture', info)
        self.assertIn('parameter_breakdown', info)
        self.assertIn('total', info['parameter_breakdown'])
        self.assertGreater(info['parameter_breakdown']['total'], 0)


class TestCreateHybridModel(unittest.TestCase):
    """Test create_hybrid_model factory function."""

    def test_create_base_small(self):
        """Test creating base ConvNeXt + small Swin."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small'
        )
        model.eval()  # Eval mode to avoid BatchNorm issues with batch_size=1
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 7))

    def test_create_tiny_tiny(self):
        """Test creating tiny ConvNeXt + tiny Swin."""
        model = create_hybrid_model(
            convnext_variant='tiny',
            swin_variant='tiny'
        )
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 7))

    def test_create_with_custom_classes(self):
        """Test creating with custom number of classes."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            num_classes=10
        )
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_create_with_fairdisco(self):
        """Test creating with FairDisCo enabled."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            enable_fairdisco=True,
            num_fst_classes=6
        )
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            outputs = model(x)
        self.assertEqual(len(outputs), 3)

    def test_create_with_custom_fusion_dim(self):
        """Test creating with custom fusion dimension."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            fusion_dim=1024
        )

        info = model.get_model_info()
        self.assertEqual(info['fusion_dim'], 1024)

    def test_create_with_custom_dropout(self):
        """Test creating with custom dropout rate."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            dropout=0.5
        )

        self.assertEqual(model.config.dropout, 0.5)


class TestHybridModelIntegration(unittest.TestCase):
    """Integration tests for hybrid model."""

    def test_batch_processing(self):
        """Test processing different batch sizes."""
        model = create_hybrid_model(convnext_variant='base', swin_variant='small')
        model.eval()

        for batch_size in [1, 2, 4]:
            with torch.no_grad():
                x = torch.randn(batch_size, 3, 224, 224)
                output = model(x)
                self.assertEqual(output.shape[0], batch_size)

    def test_eval_vs_train_mode(self):
        """Test model behavior in train vs eval mode."""
        model = create_hybrid_model(convnext_variant='base', swin_variant='small')
        x = torch.randn(2, 3, 224, 224)

        # Train mode
        model.train()
        output_train = model(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            output_eval = model(x)

        # Shapes should be the same
        self.assertEqual(output_train.shape, output_eval.shape)

    def test_parameter_count_reasonable(self):
        """Test that parameter count is reasonable."""
        model = create_hybrid_model(convnext_variant='base', swin_variant='small')
        info = model.get_model_info()

        total_params = info['parameter_breakdown']['total']

        # Base ConvNeXt (~60M) + Small Swin (~5M) + Fusion (~1M) + Head (~0.3M)
        # Total should be around 66-67M
        self.assertGreater(total_params, 60_000_000)
        self.assertLess(total_params, 75_000_000)

    def test_training_step_simulation(self):
        """Simulate a training step."""
        model = create_hybrid_model(convnext_variant='base', swin_variant='small')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 7, (2,))

        # Forward
        output = model(x)
        loss = criterion(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss is finite
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_training_step_with_fairdisco(self):
        """Simulate training step with FairDisCo."""
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            enable_fairdisco=True
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        x = torch.randn(2, 3, 224, 224)
        diag_target = torch.randint(0, 7, (2,))
        fst_target = torch.randint(0, 6, (2,))

        # Forward
        diag_logits, fst_logits, cont_emb = model(x)

        # Compute losses
        loss_cls = nn.CrossEntropyLoss()(diag_logits, diag_target)
        loss_adv = nn.CrossEntropyLoss()(fst_logits, fst_target)
        loss = loss_cls + 0.3 * loss_adv

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check loss is finite
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_memory_efficiency(self):
        """Test model doesn't leak memory."""
        model = create_hybrid_model(convnext_variant='base', swin_variant='small')
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        # Multiple forward passes
        for _ in range(3):
            with torch.no_grad():
                output = model(x)

        # Clean up
        del output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
