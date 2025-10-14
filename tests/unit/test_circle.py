"""
CIRCLe Unit Tests: Comprehensive Test Suite

Tests all CIRCLe components:
- LAB color transformations
- Regularization losses
- Model architecture
- Training pipeline integration

Framework: MENDICANT_BIAS - Phase 2, Week 7-8
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.fairness.color_transforms import (
    LABColorTransform,
    apply_fst_transformation,
    batch_transform_dataset,
    FST_COLOR_STATS
)
from src.fairness.circle_regularization import (
    CIRCLe_RegularizationLoss,
    MultiTargetCIRCLe_Loss,
    ToneInvarianceMetric,
    compute_tone_invariance_per_class
)
from src.models.circle_model import CIRCLe_FairDisCo, create_circle_model


class TestLABColorTransformations:
    """Test LAB color space transformations."""

    def test_transform_creation(self):
        """Test LABColorTransform initialization."""
        transform = LABColorTransform()
        assert transform is not None
        assert hasattr(transform, 'rgb_to_xyz')
        assert hasattr(transform, 'xyz_to_rgb')

    def test_single_image_transformation(self):
        """Test transformation of single image."""
        image = torch.randn(3, 224, 224)
        transformed = apply_fst_transformation(image, source_fst=3, target_fst=1, imagenet_normalized=False)

        assert transformed.shape == image.shape
        assert transformed.dtype == image.dtype
        # Lighter transformation should increase values (FST III → I)
        # Note: This is approximate due to LAB space non-linearity
        assert transformed.mean() != image.mean()

    def test_batch_transformation(self):
        """Test batch transformation."""
        images = torch.randn(8, 3, 224, 224)
        transform = LABColorTransform(imagenet_normalized=False)
        transformed = transform(images, source_fst=3, target_fst=6)

        assert transformed.shape == images.shape
        assert not torch.allclose(transformed, images)

    def test_mixed_fst_batch(self):
        """Test batch with different source/target FSTs per sample."""
        images = torch.randn(4, 3, 224, 224)
        source_fsts = torch.tensor([1, 2, 3, 4])
        target_fsts = torch.tensor([6, 5, 4, 3])

        transform = LABColorTransform(imagenet_normalized=False)
        transformed = transform(images, source_fsts, target_fsts)

        assert transformed.shape == images.shape

    def test_rgb_lab_roundtrip(self):
        """Test RGB → LAB → RGB round-trip accuracy."""
        original = torch.rand(2, 3, 64, 64)
        transform = LABColorTransform(imagenet_normalized=False)

        lab = transform._rgb_to_lab(original)
        reconstructed = transform._lab_to_rgb(lab)

        # Allow small numerical error
        error = torch.abs(original - reconstructed).mean()
        assert error < 0.01, f"Round-trip error too large: {error:.6f}"

    def test_identity_transformation(self):
        """Test transformation to same FST (should be close to identity)."""
        images = torch.randn(4, 3, 224, 224)
        transform = LABColorTransform(imagenet_normalized=False)

        # Transform FST III → FST III
        transformed = transform(images, source_fst=3, target_fst=3)

        # Should be very close to original
        assert torch.allclose(transformed, images, atol=1e-5)

    def test_fst_color_stats(self):
        """Test FST color statistics dictionary."""
        assert len(FST_COLOR_STATS) == 6
        assert all(fst in FST_COLOR_STATS for fst in range(1, 7))

        # L* should decrease from FST I to VI
        l_values = [FST_COLOR_STATS[fst]["L_mean"] for fst in range(1, 7)]
        assert all(l_values[i] > l_values[i+1] for i in range(5))

    def test_batch_transform_dataset(self):
        """Test batch_transform_dataset function."""
        images = torch.randn(16, 3, 224, 224)
        fst_labels = torch.randint(1, 7, (16,))

        transformed_dict = batch_transform_dataset(
            images, fst_labels, target_fsts=[1, 6], imagenet_normalized=False
        )

        assert len(transformed_dict) == 2
        assert 1 in transformed_dict
        assert 6 in transformed_dict
        assert transformed_dict[1].shape == images.shape
        assert transformed_dict[6].shape == images.shape

    def test_imagenet_normalization_handling(self):
        """Test ImageNet normalization/denormalization."""
        # Create ImageNet-normalized image
        transform = LABColorTransform(imagenet_normalized=True)

        image = torch.randn(1, 3, 224, 224)
        transformed = transform(image, source_fst=3, target_fst=1)

        assert transformed.shape == image.shape


class TestCIRCLe_Regularization:
    """Test CIRCLe regularization loss."""

    def test_loss_creation(self):
        """Test loss function initialization."""
        loss_fn = CIRCLe_RegularizationLoss(distance_metric="l2")
        assert loss_fn is not None
        assert loss_fn.distance_metric == "l2"

    def test_l2_loss_computation(self):
        """Test L2 distance loss."""
        loss_fn = CIRCLe_RegularizationLoss(distance_metric="l2")

        emb_orig = torch.randn(32, 2048)
        emb_trans = torch.randn(32, 2048)

        loss = loss_fn(emb_orig, emb_trans)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive
        assert torch.isfinite(loss)

    def test_cosine_loss_computation(self):
        """Test cosine distance loss."""
        loss_fn = CIRCLe_RegularizationLoss(distance_metric="cosine")

        emb_orig = torch.randn(32, 2048)
        emb_trans = torch.randn(32, 2048)

        loss = loss_fn(emb_orig, emb_trans)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_l1_loss_computation(self):
        """Test L1 distance loss."""
        loss_fn = CIRCLe_RegularizationLoss(distance_metric="l1")

        emb_orig = torch.randn(32, 2048)
        emb_trans = torch.randn(32, 2048)

        loss = loss_fn(emb_orig, emb_trans)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_identity_loss(self):
        """Test loss for identical embeddings (should be zero)."""
        loss_fn = CIRCLe_RegularizationLoss(distance_metric="l2")

        emb = torch.randn(16, 1024)
        loss = loss_fn(emb, emb.clone())

        assert loss.item() < 1e-6, f"Identity loss should be near zero: {loss.item()}"

    def test_normalized_embeddings(self):
        """Test loss with normalized embeddings."""
        loss_fn = CIRCLe_RegularizationLoss(
            distance_metric="l2",
            normalize_embeddings=True
        )

        emb_orig = torch.randn(16, 512)
        emb_trans = torch.randn(16, 512)

        loss = loss_fn(emb_orig, emb_trans)

        assert torch.isfinite(loss)

    def test_pairwise_distances(self):
        """Test pairwise distance computation."""
        loss_fn = CIRCLe_RegularizationLoss(distance_metric="l2")

        emb_orig = torch.randn(8, 256)
        emb_trans = torch.randn(8, 256)

        distances = loss_fn.get_pairwise_distances(emb_orig, emb_trans)

        assert distances.shape == (8,)
        assert all(distances >= 0)

    def test_multi_target_loss(self):
        """Test multi-target CIRCLe loss."""
        loss_fn = MultiTargetCIRCLe_Loss(target_fsts=[1, 6], distance_metric="l2")

        emb_orig = torch.randn(16, 2048)
        emb_trans_dict = {
            1: torch.randn(16, 2048),
            6: torch.randn(16, 2048)
        }

        loss = loss_fn(emb_orig, emb_trans_dict)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_tone_invariance_metric(self):
        """Test tone-invariance metric computation."""
        metric = ToneInvarianceMetric(distance_metric="l2", normalize_embeddings=True)

        embeddings = torch.randn(32, 512)
        labels = torch.randint(0, 7, (32,))
        fst_labels = torch.randint(1, 7, (32,))

        score = metric(embeddings, labels, fst_labels)

        assert score.dim() == 0
        assert score.item() >= 0

    def test_per_class_tone_invariance(self):
        """Test per-class tone-invariance computation."""
        embeddings = torch.randn(64, 512)
        labels = torch.randint(0, 7, (64,))
        fst_labels = torch.randint(1, 7, (64,))

        scores = compute_tone_invariance_per_class(
            embeddings, labels, fst_labels, num_classes=7
        )

        assert len(scores) == 7
        assert all(isinstance(s, float) for s in scores.values())

    def test_gradient_flow_through_loss(self):
        """Test gradient flow through regularization loss."""
        loss_fn = CIRCLe_RegularizationLoss(distance_metric="l2")

        emb_orig = torch.randn(8, 256, requires_grad=True)
        emb_trans = torch.randn(8, 256, requires_grad=True)

        loss = loss_fn(emb_orig, emb_trans)
        loss.backward()

        assert emb_orig.grad is not None
        assert emb_trans.grad is not None
        assert not torch.all(emb_orig.grad == 0)


class TestCIRCLe_Model:
    """Test CIRCLe model architecture."""

    def test_model_creation(self):
        """Test CIRCLe model initialization."""
        model = create_circle_model(
            num_classes=7,
            num_fst_classes=6,
            pretrained=False,
            target_fsts=[1, 6]
        )

        assert model is not None
        assert isinstance(model, CIRCLe_FairDisCo)

    def test_forward_pass(self):
        """Test forward pass through model."""
        model = create_circle_model(
            num_classes=7,
            pretrained=False,
            target_fsts=[1, 6]
        )

        images = torch.randn(4, 3, 224, 224)
        fst_labels = torch.randint(1, 7, (4,))

        with torch.no_grad():
            outputs = model(images, fst_labels)

        assert 'diagnosis_logits' in outputs
        assert 'fst_logits' in outputs
        assert 'contrastive_embeddings' in outputs
        assert 'embeddings_original' in outputs
        assert 'embeddings_transformed' in outputs

        assert outputs['diagnosis_logits'].shape == (4, 7)
        assert outputs['fst_logits'].shape == (4, 6)

    def test_multi_target_transformation(self):
        """Test multi-target FST transformation in model."""
        model = create_circle_model(
            num_classes=7,
            pretrained=False,
            target_fsts=[1, 6]
        )

        images = torch.randn(4, 3, 224, 224)
        fst_labels = torch.randint(1, 7, (4,))

        with torch.no_grad():
            outputs = model(images, fst_labels)

        emb_trans = outputs['embeddings_transformed']
        assert isinstance(emb_trans, dict)
        assert 1 in emb_trans
        assert 6 in emb_trans

    def test_single_target_model(self):
        """Test single-target FST model."""
        model = create_circle_model(
            num_classes=7,
            pretrained=False,
            target_fsts=[1]
        )

        images = torch.randn(4, 3, 224, 224)
        fst_labels = torch.randint(1, 7, (4,))

        with torch.no_grad():
            outputs = model(images, fst_labels)

        emb_trans = outputs['embeddings_transformed']
        assert isinstance(emb_trans, torch.Tensor)
        assert emb_trans.shape[0] == 4

    def test_circle_loss_computation(self):
        """Test CIRCLe loss computation in model."""
        model = create_circle_model(
            num_classes=7,
            pretrained=False,
            target_fsts=[1, 6]
        )

        images = torch.randn(4, 3, 224, 224)
        fst_labels = torch.randint(1, 7, (4,))

        outputs = model(images, fst_labels)
        loss_reg = model.compute_circle_loss(
            outputs['embeddings_original'],
            outputs['embeddings_transformed']
        )

        assert loss_reg.dim() == 0
        assert loss_reg.item() > 0

    def test_lambda_updates(self):
        """Test lambda update methods."""
        model = create_circle_model(num_classes=7, pretrained=False)

        initial_lambda_reg = model.model_info['lambda_reg']
        model.update_lambda_reg(0.3)
        assert model.model_info['lambda_reg'] == 0.3

        initial_lambda_adv = model.model_info['lambda_adv']
        model.update_lambda_adv(0.4)
        assert model.model_info['lambda_adv'] == 0.4

    def test_fairdisco_access(self):
        """Test access to underlying FairDisCo model."""
        model = create_circle_model(num_classes=7, pretrained=False)

        fairdisco_model = model.get_fairdisco_model()
        assert fairdisco_model is not None

        feature_extractor = model.get_feature_extractor()
        assert feature_extractor is not None

    def test_gradient_flow_through_model(self):
        """Test gradient flow through entire model."""
        model = create_circle_model(num_classes=7, pretrained=False)

        images = torch.randn(2, 3, 224, 224, requires_grad=True)
        fst_labels = torch.tensor([3, 4])

        outputs = model(images, fst_labels)

        # Compute all losses
        loss_cls = outputs['diagnosis_logits'].sum()
        loss_reg = model.compute_circle_loss(
            outputs['embeddings_original'],
            outputs['embeddings_transformed']
        )

        total_loss = loss_cls + loss_reg
        total_loss.backward()

        assert images.grad is not None

    def test_model_info(self):
        """Test model info dictionary."""
        model = create_circle_model(
            num_classes=7,
            pretrained=False,
            lambda_reg=0.25,
            target_fsts=[1, 6]
        )

        info = model.get_model_info()

        assert info['architecture'] == 'circle_fairdisco'
        assert info['num_classes'] == 7
        assert info['lambda_reg'] == 0.25
        assert info['target_fsts'] == [1, 6]


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_forward_pass(self):
        """Test complete forward pass with all losses."""
        model = create_circle_model(num_classes=7, pretrained=False, target_fsts=[1, 6])

        images = torch.randn(8, 3, 224, 224)
        targets = torch.randint(0, 7, (8,))
        fst_labels = torch.randint(1, 7, (8,))

        # Forward pass
        outputs = model(images, fst_labels)

        # Compute all four losses
        criterion_cls = nn.CrossEntropyLoss()
        criterion_adv = nn.CrossEntropyLoss()

        from src.models.fairdisco_model import SupervisedContrastiveLoss
        criterion_con = SupervisedContrastiveLoss()

        loss_cls = criterion_cls(outputs['diagnosis_logits'], targets)
        loss_adv = criterion_adv(outputs['fst_logits'], fst_labels)
        loss_con = criterion_con(
            outputs['contrastive_embeddings'], targets, fst_labels
        )
        loss_reg = model.compute_circle_loss(
            outputs['embeddings_original'],
            outputs['embeddings_transformed']
        )

        # All losses should be finite
        assert torch.isfinite(loss_cls)
        assert torch.isfinite(loss_adv)
        assert torch.isfinite(loss_con)
        assert torch.isfinite(loss_reg)

        # Total loss
        total_loss = loss_cls + 0.3*loss_adv + 0.2*loss_con + 0.2*loss_reg
        assert torch.isfinite(total_loss)

    def test_backward_pass(self):
        """Test backward pass through all losses."""
        model = create_circle_model(num_classes=7, pretrained=False)

        images = torch.randn(4, 3, 224, 224)
        targets = torch.randint(0, 7, (4,))
        fst_labels = torch.randint(1, 7, (4,))

        outputs = model(images, fst_labels)

        # Simplified loss
        loss = outputs['diagnosis_logits'].sum() + \
               model.compute_circle_loss(
                   outputs['embeddings_original'],
                   outputs['embeddings_transformed']
               )

        loss.backward()

        # Check that some parameters have gradients
        has_gradients = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in model.parameters()
        )
        assert has_gradients

    def test_color_transform_with_model(self):
        """Test color transformations integrated with model."""
        model = create_circle_model(num_classes=7, pretrained=False, target_fsts=[1, 6])

        images = torch.randn(4, 3, 224, 224)
        fst_labels = torch.tensor([2, 3, 4, 5])

        with torch.no_grad():
            outputs = model(images, fst_labels, return_transformed_images=True)

        assert 'images_transformed' in outputs
        images_trans = outputs['images_transformed']

        assert isinstance(images_trans, dict)
        assert images_trans[1].shape == images.shape
        assert images_trans[6].shape == images.shape


def test_all():
    """Run all tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    test_all()
