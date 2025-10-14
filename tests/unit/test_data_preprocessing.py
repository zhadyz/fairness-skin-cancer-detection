"""
Unit tests for data preprocessing functionality.

Tests image transformations, augmentations, normalization, and FST stratification.
"""

import pytest
import torch
import numpy as np
from PIL import Image


# ============================================================================
# IMAGE NORMALIZATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.preprocessing
class TestImageNormalization:
    """Test image normalization functions."""

    def test_normalize_imagenet_mean_std(self, sample_image_tensor, imagenet_mean_std):
        """Test normalization using ImageNet statistics."""
        from torchvision import transforms

        normalize = transforms.Normalize(
            mean=imagenet_mean_std['mean'],
            std=imagenet_mean_std['std']
        )

        normalized = normalize(sample_image_tensor)

        # Check shape is preserved
        assert normalized.shape == sample_image_tensor.shape

        # Check dtype is preserved
        assert normalized.dtype == sample_image_tensor.dtype

        # Check values are actually transformed (should differ from input)
        assert not torch.allclose(normalized, sample_image_tensor)

    def test_normalize_zero_mean_unit_std(self):
        """Test that normalization produces approximately zero mean and unit std."""
        from torchvision import transforms

        # Create image with known statistics
        img = torch.rand(3, 224, 224) * 0.5 + 0.5  # Values in [0.5, 1.0]

        # Compute actual mean/std
        mean = img.mean(dim=[1, 2])
        std = img.std(dim=[1, 2])

        # Normalize
        normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        normalized = normalize(img)

        # Check approximately zero mean and unit std per channel
        for c in range(3):
            channel_mean = normalized[c].mean()
            channel_std = normalized[c].std()
            assert torch.abs(channel_mean) < 1e-6
            assert torch.abs(channel_std - 1.0) < 1e-5

    def test_denormalize(self, imagenet_mean_std):
        """Test denormalization to recover original image."""
        from torchvision import transforms

        # Original image
        original = torch.rand(3, 224, 224)

        # Normalize
        normalize = transforms.Normalize(
            mean=imagenet_mean_std['mean'],
            std=imagenet_mean_std['std']
        )
        normalized = normalize(original)

        # Denormalize
        mean = torch.tensor(imagenet_mean_std['mean']).view(3, 1, 1)
        std = torch.tensor(imagenet_mean_std['std']).view(3, 1, 1)
        denormalized = normalized * std + mean

        # Should recover original (within floating point precision)
        assert torch.allclose(denormalized, original, atol=1e-6)


# ============================================================================
# IMAGE RESIZE TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.preprocessing
class TestImageResize:
    """Test image resizing operations."""

    def test_resize_224x224(self):
        """Test resizing to 224x224 (standard input size)."""
        from torchvision import transforms

        resize = transforms.Resize((224, 224))

        # Test various input sizes
        input_sizes = [(300, 300), (128, 128), (512, 384), (100, 200)]

        for h, w in input_sizes:
            img = Image.new('RGB', (w, h))
            resized = resize(img)

            assert resized.size == (224, 224), f"Failed for input size {(h, w)}"

    def test_resize_preserves_aspect_ratio(self):
        """Test that resize with one dimension preserves aspect ratio."""
        from torchvision import transforms

        resize = transforms.Resize(224)  # Resize shorter edge to 224

        # Wide image
        img_wide = Image.new('RGB', (400, 200))
        resized_wide = resize(img_wide)
        assert resized_wide.size[1] == 224  # Height should be 224
        assert resized_wide.size[0] == 448  # Width should be 2x height

        # Tall image
        img_tall = Image.new('RGB', (200, 400))
        resized_tall = resize(img_tall)
        assert resized_tall.size[0] == 224  # Width should be 224
        assert resized_tall.size[1] == 448  # Height should be 2x width

    def test_center_crop(self):
        """Test center cropping after resize."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])

        img = Image.new('RGB', (300, 300))
        result = transform(img)

        assert result.size == (224, 224)


# ============================================================================
# AUGMENTATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.preprocessing
class TestAugmentation:
    """Test data augmentation pipeline."""

    def test_random_horizontal_flip(self):
        """Test random horizontal flip produces different outputs."""
        from torchvision import transforms

        flip = transforms.RandomHorizontalFlip(p=1.0)  # Always flip

        img_tensor = torch.rand(3, 224, 224)
        flipped = flip(img_tensor)

        # Check that image is actually flipped (left-right reversed)
        assert torch.allclose(flipped, img_tensor.flip(-1))

    def test_random_vertical_flip(self):
        """Test random vertical flip."""
        from torchvision import transforms

        flip = transforms.RandomVerticalFlip(p=1.0)

        img_tensor = torch.rand(3, 224, 224)
        flipped = flip(img_tensor)

        # Check that image is vertically flipped (top-bottom reversed)
        assert torch.allclose(flipped, img_tensor.flip(-2))

    def test_color_jitter(self):
        """Test color jitter augmentation."""
        from torchvision import transforms

        jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        jittered = jitter(img)

        # Check output is still valid image
        assert jittered.size == img.size
        assert jittered.mode == img.mode

    def test_random_rotation(self):
        """Test random rotation augmentation."""
        from torchvision import transforms

        rotate = transforms.RandomRotation(degrees=45)

        img = Image.new('RGB', (224, 224))
        rotated = rotate(img)

        # Check output dimensions
        assert rotated.size == img.size

    def test_augmentation_pipeline_reproducibility(self, random_seed):
        """Test that augmentation pipeline is reproducible with seed."""
        from torchvision import transforms

        torch.manual_seed(random_seed)

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ])

        img = Image.new('RGB', (224, 224), color=(100, 150, 200))

        # Apply with seed
        torch.manual_seed(random_seed)
        result1 = transform(img)

        torch.manual_seed(random_seed)
        result2 = transform(img)

        # Results should be identical
        assert np.array_equal(np.array(result1), np.array(result2))


# ============================================================================
# FST STRATIFIED SPLITTING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.preprocessing
class TestFSTStratification:
    """Test Fitzpatrick Skin Type stratified data splitting."""

    def test_stratified_split_distribution(self, fst_stratified_data):
        """Test that stratified split preserves FST distribution."""
        from sklearn.model_selection import train_test_split

        labels = fst_stratified_data['labels'].numpy()
        fst = fst_stratified_data['fst'].numpy()

        # Perform stratified split
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            stratify=fst,
            random_state=42
        )

        # Check FST distribution is similar in train and test
        fst_train = fst[train_idx]
        fst_test = fst[test_idx]

        for fst_type in range(1, 7):
            train_prop = (fst_train == fst_type).sum() / len(fst_train)
            test_prop = (fst_test == fst_type).sum() / len(fst_test)

            # Proportions should be close (within 5%)
            assert abs(train_prop - test_prop) < 0.05

    def test_stratified_split_sizes(self):
        """Test that stratified split produces correct sizes."""
        from sklearn.model_selection import train_test_split

        n_samples = 1000
        labels = np.random.randint(0, 7, n_samples)
        fst = np.random.randint(1, 7, n_samples)

        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.2,
            stratify=fst,
            random_state=42
        )

        assert len(train_idx) == 800
        assert len(test_idx) == 200
        assert len(set(train_idx) & set(test_idx)) == 0  # No overlap

    def test_stratified_kfold(self):
        """Test stratified K-fold cross-validation."""
        from sklearn.model_selection import StratifiedKFold

        n_samples = 500
        labels = np.random.randint(0, 7, n_samples)
        fst = np.random.randint(1, 7, n_samples)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_sizes = []
        for train_idx, val_idx in skf.split(np.arange(n_samples), fst):
            fold_sizes.append(len(val_idx))

            # Check FST distribution in each fold
            fst_train = fst[train_idx]
            fst_val = fst[val_idx]

            for fst_type in range(1, 7):
                train_prop = (fst_train == fst_type).sum() / len(fst_train)
                val_prop = (fst_val == fst_type).sum() / len(fst_val)

                # Should be roughly equal
                assert abs(train_prop - val_prop) < 0.1

        # All folds should be approximately equal size
        assert max(fold_sizes) - min(fold_sizes) <= 2


# ============================================================================
# DATASET LOADING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.preprocessing
class TestDatasetLoading:
    """Test dataset loading and batching."""

    def test_dataset_length(self, mock_ham10000_dataset):
        """Test dataset returns correct length."""
        assert len(mock_ham10000_dataset) == 100

    def test_dataset_getitem(self, mock_ham10000_dataset):
        """Test dataset __getitem__ returns correct structure."""
        sample = mock_ham10000_dataset[0]

        assert 'image' in sample
        assert 'label' in sample
        assert 'fst' in sample
        assert 'metadata' in sample

        assert sample['image'].shape == (3, 224, 224)
        assert isinstance(sample['label'].item(), int)
        assert 1 <= sample['fst'].item() <= 6

    def test_dataloader_batching(self, mock_dataloader):
        """Test DataLoader produces correct batch shapes."""
        batch = next(iter(mock_dataloader))

        assert batch['image'].shape[0] == 8  # Batch size
        assert batch['image'].shape[1:] == (3, 224, 224)
        assert batch['label'].shape[0] == 8
        assert batch['fst'].shape[0] == 8

    def test_dataloader_iteration(self, mock_dataloader):
        """Test iterating through entire DataLoader."""
        total_samples = 0
        for batch in mock_dataloader:
            total_samples += batch['image'].shape[0]

        assert total_samples == 32  # Total dataset size

    def test_dataset_class_names(self, mock_ham10000_dataset):
        """Test dataset has correct class names."""
        expected_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        assert mock_ham10000_dataset.class_names == expected_classes


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.preprocessing
class TestEdgeCases:
    """Test edge cases in preprocessing."""

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        from torch.utils.data import DataLoader, TensorDataset

        empty_data = torch.empty(0, 3, 224, 224)
        empty_labels = torch.empty(0, dtype=torch.long)
        dataset = TensorDataset(empty_data, empty_labels)

        loader = DataLoader(dataset, batch_size=8)

        # Should produce no batches
        batches = list(loader)
        assert len(batches) == 0

    def test_single_sample_batch(self, mock_ham10000_small):
        """Test DataLoader with batch size = 1."""
        from torch.utils.data import DataLoader

        loader = DataLoader(mock_ham10000_small, batch_size=1)

        batch = next(iter(loader))
        assert batch['image'].shape[0] == 1

    def test_normalize_extreme_values(self):
        """Test normalization with extreme pixel values."""
        from torchvision import transforms

        # All white image
        white = torch.ones(3, 224, 224)
        # All black image
        black = torch.zeros(3, 224, 224)

        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        white_norm = normalize(white)
        black_norm = normalize(black)

        # Check no NaN or Inf
        assert torch.isfinite(white_norm).all()
        assert torch.isfinite(black_norm).all()
