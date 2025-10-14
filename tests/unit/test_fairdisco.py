"""
Unit Tests for FairDisCo Implementation

Comprehensive test suite for FairDisCo adversarial debiasing model:
- Gradient Reversal Layer (forward/backward pass correctness)
- FST Discriminator architecture
- Supervised Contrastive Loss
- FairDisCoClassifier forward pass and outputs
- Lambda scheduling and update
- Model parameter count

Framework: MENDICANT_BIAS - Phase 2, Week 5-6
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.fairdisco_model import (
    GradientReversalFunction,
    GradientReversalLayer,
    FST_Discriminator,
    SupervisedContrastiveLoss,
    FairDisCoClassifier,
    create_fairdisco_model
)


class TestGradientReversalLayer:
    """Test suite for Gradient Reversal Layer."""

    def test_grl_forward_identity(self):
        """Test that forward pass is identity operation."""
        grl = GradientReversalLayer(lambda_=0.5)
        x = torch.randn(4, 2048, requires_grad=True)
        y = grl(x)

        # Forward should be identity
        assert torch.allclose(y, x), "GRL forward pass should be identity"

    def test_grl_backward_reversal(self):
        """Test that backward pass reverses gradient."""
        lambda_ = 0.5
        grl = GradientReversalLayer(lambda_=lambda_)

        x = torch.randn(4, 2048, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be reversed and scaled by lambda
        expected_grad = -lambda_ * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
            "GRL backward pass should reverse and scale gradient"

    def test_grl_lambda_update(self):
        """Test lambda update functionality."""
        grl = GradientReversalLayer(lambda_=0.0)
        assert grl.lambda_ == 0.0

        grl.set_lambda(0.3)
        assert grl.lambda_ == 0.3

        grl.set_lambda(0.5)
        assert grl.lambda_ == 0.5

    def test_grl_zero_lambda(self):
        """Test GRL with lambda=0 (no gradient reversal)."""
        grl = GradientReversalLayer(lambda_=0.0)

        x = torch.randn(4, 2048, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        # With lambda=0, gradient should be zero
        expected_grad = torch.zeros_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
            "GRL with lambda=0 should zero out gradient"


class TestFSTDiscriminator:
    """Test suite for FST Discriminator."""

    def test_discriminator_output_shape(self):
        """Test discriminator output shape."""
        disc = FST_Discriminator(feature_dim=2048, num_fst_classes=6)
        x = torch.randn(8, 2048)
        output = disc(x)

        assert output.shape == (8, 6), \
            f"Expected shape (8, 6), got {output.shape}"

    def test_discriminator_forward_pass(self):
        """Test discriminator forward pass is finite."""
        disc = FST_Discriminator(feature_dim=2048, num_fst_classes=6)
        x = torch.randn(16, 2048)
        output = disc(x)

        assert torch.isfinite(output).all(), \
            "Discriminator output should be finite"

    def test_discriminator_trainable(self):
        """Test that discriminator parameters are trainable."""
        disc = FST_Discriminator(feature_dim=2048, num_fst_classes=6)
        trainable_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)

        assert trainable_params > 0, \
            "Discriminator should have trainable parameters"

    def test_discriminator_gradient_flow(self):
        """Test that gradients flow through discriminator."""
        disc = FST_Discriminator(feature_dim=2048, num_fst_classes=6)
        x = torch.randn(4, 2048, requires_grad=True)
        output = disc(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradient
        assert x.grad is not None, \
            "Gradients should flow through discriminator"
        assert torch.isfinite(x.grad).all(), \
            "Gradients should be finite"


class TestSupervisedContrastiveLoss:
    """Test suite for Supervised Contrastive Loss."""

    def test_contrastive_loss_computation(self):
        """Test basic contrastive loss computation."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)

        # Create dummy data
        embeddings = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        fst_labels = torch.tensor([1, 3, 1, 3, 2, 4, 2, 4])

        loss = loss_fn(embeddings, labels, fst_labels)

        assert isinstance(loss, torch.Tensor), \
            "Loss should be a tensor"
        assert loss.ndim == 0, \
            "Loss should be scalar"
        assert torch.isfinite(loss), \
            "Loss should be finite"
        assert loss >= 0, \
            "Contrastive loss should be non-negative"

    def test_contrastive_loss_gradient_flow(self):
        """Test that gradients flow through contrastive loss."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)

        embeddings = torch.randn(8, 128, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        fst_labels = torch.tensor([1, 3, 1, 3, 2, 4, 2, 4])

        loss = loss_fn(embeddings, labels, fst_labels)
        loss.backward()

        assert embeddings.grad is not None, \
            "Gradients should flow through contrastive loss"
        assert torch.isfinite(embeddings.grad).all(), \
            "Gradients should be finite"

    def test_contrastive_loss_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        embeddings = torch.randn(16, 128)
        labels = torch.randint(0, 3, (16,))
        fst_labels = torch.randint(1, 7, (16,))

        loss_fn_low_temp = SupervisedContrastiveLoss(temperature=0.01)
        loss_fn_high_temp = SupervisedContrastiveLoss(temperature=1.0)

        loss_low = loss_fn_low_temp(embeddings, labels, fst_labels)
        loss_high = loss_fn_high_temp(embeddings, labels, fst_labels)

        # Lower temperature typically gives higher loss magnitude
        assert loss_low > loss_high, \
            "Lower temperature should generally give higher loss"

    def test_contrastive_loss_no_positives(self):
        """Test contrastive loss when no positive pairs exist."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)

        # All different labels and FST
        embeddings = torch.randn(4, 128)
        labels = torch.tensor([0, 1, 2, 3])
        fst_labels = torch.tensor([1, 2, 3, 4])

        loss = loss_fn(embeddings, labels, fst_labels)

        # Should handle gracefully (return 0 or small value)
        assert torch.isfinite(loss), \
            "Loss should be finite even with no positive pairs"


class TestFairDisCoClassifier:
    """Test suite for complete FairDisCo model."""

    def test_model_creation(self):
        """Test basic model creation."""
        model = create_fairdisco_model(
            num_classes=7,
            num_fst_classes=6,
            pretrained=False
        )

        assert isinstance(model, FairDisCoClassifier), \
            "Should create FairDisCoClassifier instance"
        assert model.num_classes == 7, \
            "Should have correct number of classes"
        assert model.num_fst_classes == 6, \
            "Should have correct number of FST classes"

    def test_model_forward_pass_shapes(self):
        """Test forward pass output shapes."""
        model = create_fairdisco_model(
            num_classes=7,
            num_fst_classes=6,
            pretrained=False
        )

        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        diagnosis_logits, fst_logits, contrastive_embeddings, _ = model(x, return_embeddings=False)

        assert diagnosis_logits.shape == (batch_size, 7), \
            f"Expected diagnosis shape (4, 7), got {diagnosis_logits.shape}"
        assert fst_logits.shape == (batch_size, 6), \
            f"Expected FST shape (4, 6), got {fst_logits.shape}"
        assert contrastive_embeddings.shape == (batch_size, 128), \
            f"Expected contrastive shape (4, 128), got {contrastive_embeddings.shape}"

    def test_model_forward_with_embeddings(self):
        """Test forward pass with embeddings return."""
        model = create_fairdisco_model(
            num_classes=7,
            num_fst_classes=6,
            pretrained=False
        )

        x = torch.randn(4, 3, 224, 224)
        diagnosis_logits, fst_logits, contrastive_embeddings, embeddings = model(x, return_embeddings=True)

        assert embeddings is not None, \
            "Should return embeddings when requested"
        assert embeddings.shape == (4, 2048), \
            f"Expected embeddings shape (4, 2048), got {embeddings.shape}"

    def test_model_outputs_finite(self):
        """Test that all model outputs are finite."""
        model = create_fairdisco_model(
            num_classes=7,
            num_fst_classes=6,
            pretrained=False
        )

        x = torch.randn(4, 3, 224, 224)
        diagnosis_logits, fst_logits, contrastive_embeddings, embeddings = model(x, return_embeddings=True)

        assert torch.isfinite(diagnosis_logits).all(), \
            "Diagnosis logits should be finite"
        assert torch.isfinite(fst_logits).all(), \
            "FST logits should be finite"
        assert torch.isfinite(contrastive_embeddings).all(), \
            "Contrastive embeddings should be finite"
        assert torch.isfinite(embeddings).all(), \
            "Raw embeddings should be finite"

    def test_model_lambda_update(self):
        """Test lambda_adv update functionality."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)

        initial_lambda = 0.3
        assert model.model_info['lambda_adv'] == initial_lambda

        new_lambda = 0.5
        model.update_lambda_adv(new_lambda)
        assert model.model_info['lambda_adv'] == new_lambda
        assert model.grl.lambda_ == new_lambda

    def test_model_trainable_parameters(self):
        """Test that model has trainable parameters."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        assert trainable_params > 0, \
            "Model should have trainable parameters"
        assert trainable_params == total_params, \
            "All parameters should be trainable by default"

    def test_model_freeze_unfreeze_backbone(self):
        """Test backbone freeze/unfreeze functionality."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)

        # Initially unfrozen
        backbone_trainable_initial = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        assert backbone_trainable_initial > 0

        # Freeze backbone
        model.freeze_backbone()
        backbone_trainable_frozen = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        assert backbone_trainable_frozen == 0, \
            "Backbone should have no trainable parameters after freezing"

        # Unfreeze backbone
        model.unfreeze_backbone()
        backbone_trainable_unfrozen = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
        assert backbone_trainable_unfrozen > 0, \
            "Backbone should have trainable parameters after unfreezing"

    def test_model_gradient_flow(self):
        """Test that gradients flow through all branches."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        diagnosis_logits, fst_logits, contrastive_embeddings, _ = model(x)

        # Test diagnosis branch
        loss_diag = diagnosis_logits.sum()
        loss_diag.backward(retain_graph=True)
        assert x.grad is not None, "Gradients should flow through diagnosis branch"
        x.grad = None

        # Test FST branch (should reverse gradients)
        loss_fst = fst_logits.sum()
        loss_fst.backward(retain_graph=True)
        assert x.grad is not None, "Gradients should flow through FST branch"
        x.grad = None

        # Test contrastive branch
        loss_con = contrastive_embeddings.sum()
        loss_con.backward()
        assert x.grad is not None, "Gradients should flow through contrastive branch"

    def test_contrastive_embeddings_normalized(self):
        """Test that contrastive embeddings are L2 normalized."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)

        x = torch.randn(4, 3, 224, 224)
        _, _, contrastive_embeddings, _ = model(x)

        # Check L2 norm is 1 for each embedding
        norms = torch.norm(contrastive_embeddings, p=2, dim=1)
        expected_norms = torch.ones_like(norms)

        assert torch.allclose(norms, expected_norms, atol=1e-6), \
            "Contrastive embeddings should be L2 normalized"

    def test_model_parameter_count(self):
        """Test expected parameter count."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)

        total_params = sum(p.numel() for p in model.parameters())

        # ResNet50 has ~25.6M params, plus heads (~1-2M)
        # Expected total: ~27-28M
        assert total_params > 25_000_000, \
            f"Expected >25M parameters, got {total_params:,}"
        assert total_params < 35_000_000, \
            f"Expected <35M parameters, got {total_params:,}"


class TestModelIntegration:
    """Integration tests for complete training scenario."""

    def test_three_loss_computation(self):
        """Test computation of all three losses together."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)

        # Create dummy batch
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        fst_labels = torch.tensor([1, 2, 3, 4])

        # Forward pass
        diagnosis_logits, fst_logits, contrastive_embeddings, _ = model(images)

        # Compute losses
        criterion_cls = nn.CrossEntropyLoss()
        criterion_adv = nn.CrossEntropyLoss()
        criterion_con = SupervisedContrastiveLoss()

        loss_cls = criterion_cls(diagnosis_logits, labels)
        loss_adv = criterion_adv(fst_logits, fst_labels)
        loss_con = criterion_con(contrastive_embeddings, labels, fst_labels)

        # Total loss
        loss = loss_cls + 0.3 * loss_adv + 0.2 * loss_con

        assert torch.isfinite(loss), "Total loss should be finite"
        assert loss >= 0, "Total loss should be non-negative"

    def test_backward_pass_three_losses(self):
        """Test backward pass with all three losses."""
        model = create_fairdisco_model(num_classes=7, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create dummy batch
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        fst_labels = torch.tensor([1, 2, 3, 4])

        # Forward pass
        diagnosis_logits, fst_logits, contrastive_embeddings, _ = model(images)

        # Compute losses
        criterion_cls = nn.CrossEntropyLoss()
        criterion_adv = nn.CrossEntropyLoss()
        criterion_con = SupervisedContrastiveLoss()

        loss_cls = criterion_cls(diagnosis_logits, labels)
        loss_adv = criterion_adv(fst_logits, fst_labels)
        loss_con = criterion_con(contrastive_embeddings, labels, fst_labels)

        loss = loss_cls + 0.3 * loss_adv + 0.2 * loss_con

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check that gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads, "Model should have gradients after backward pass"

        # Check that gradients are finite
        all_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all_finite, "All gradients should be finite"

        # Optimizer step should work
        optimizer.step()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
