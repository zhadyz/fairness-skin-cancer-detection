"""
Mock datasets and fixtures for testing skin cancer detection models.

This module provides synthetic data that mimics the structure and properties
of real HAM10000 and other dermatological datasets without requiring actual
image files.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from PIL import Image


class MockHAM10000(Dataset):
    """
    Mock HAM10000 dataset for testing.

    Simulates the HAM10000 skin lesion dataset with synthetic images,
    labels, and Fitzpatrick Skin Type (FST) annotations.

    Args:
        num_samples: Number of synthetic samples to generate
        num_classes: Number of lesion classes (default: 7)
        image_size: Size of synthetic images (default: 224x224)
        include_fst: Whether to include FST annotations (default: True)
        balanced: Whether to balance class distribution (default: True)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 7,
        image_size: int = 224,
        include_fst: bool = True,
        balanced: bool = True,
        seed: Optional[int] = 42
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.include_fst = include_fst

        # Generate synthetic images with realistic skin-like appearance
        # Images have RGB channels with values resembling skin tones
        self.images = self._generate_skin_images(num_samples, image_size)

        # Generate labels (disease classes)
        if balanced:
            # Balanced distribution across classes
            self.labels = torch.tensor(
                [i % num_classes for i in range(num_samples)],
                dtype=torch.long
            )
        else:
            # Imbalanced distribution (more common lesions)
            self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

        # Generate FST annotations (Fitzpatrick I-VI)
        if include_fst:
            self.fst = torch.randint(1, 7, (num_samples,), dtype=torch.long)
        else:
            self.fst = None

        # Generate metadata
        self.metadata = self._generate_metadata(num_samples)

        # Disease class names (HAM10000 lesion types)
        self.class_names = [
            'akiec',  # Actinic keratoses and intraepithelial carcinoma
            'bcc',    # Basal cell carcinoma
            'bkl',    # Benign keratosis-like lesions
            'df',     # Dermatofibroma
            'mel',    # Melanoma
            'nv',     # Melanocytic nevi
            'vasc'    # Vascular lesions
        ]

    def _generate_skin_images(self, num_samples: int, size: int) -> torch.Tensor:
        """Generate synthetic images with skin-like color distributions."""
        # Base skin tone (RGB values in [0, 1])
        base_tones = torch.tensor([
            [0.85, 0.65, 0.55],  # Fair skin (FST I-II)
            [0.75, 0.58, 0.45],  # Medium skin (FST III-IV)
            [0.50, 0.38, 0.28],  # Dark skin (FST V-VI)
        ])

        images = []
        for _ in range(num_samples):
            # Random skin tone selection
            tone = base_tones[torch.randint(0, 3, (1,)).item()]

            # Generate image with noise around base tone
            img = tone.view(3, 1, 1).expand(3, size, size).clone()
            noise = torch.randn(3, size, size) * 0.15
            img = torch.clamp(img + noise, 0, 1)

            # Add synthetic lesion (darker region)
            lesion_size = torch.randint(20, 80, (1,)).item()
            x_center = torch.randint(lesion_size, size - lesion_size, (1,)).item()
            y_center = torch.randint(lesion_size, size - lesion_size, (1,)).item()

            for i in range(max(0, x_center - lesion_size), min(size, x_center + lesion_size)):
                for j in range(max(0, y_center - lesion_size), min(size, y_center + lesion_size)):
                    dist = np.sqrt((i - x_center)**2 + (j - y_center)**2)
                    if dist < lesion_size:
                        # Darken pixel based on distance from center
                        darkening = (1 - dist / lesion_size) * 0.4
                        img[:, i, j] *= (1 - darkening)

            images.append(img)

        return torch.stack(images)

    def _generate_metadata(self, num_samples: int) -> List[Dict]:
        """Generate mock metadata for each sample."""
        metadata = []
        for i in range(num_samples):
            metadata.append({
                'image_id': f'ISIC_{i:07d}',
                'age': int(np.random.normal(50, 20, 1).clip(18, 90)[0]),
                'sex': np.random.choice(['male', 'female']),
                'localization': np.random.choice([
                    'scalp', 'face', 'neck', 'trunk', 'upper extremity',
                    'lower extremity', 'palms/soles', 'genital'
                ])
            })
        return metadata

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - image: Tensor of shape (3, H, W)
                - label: Scalar tensor with class label
                - fst: Scalar tensor with FST (if include_fst=True)
                - metadata: Dictionary with sample metadata
        """
        sample = {
            'image': self.images[idx],
            'label': self.labels[idx],
            'metadata': self.metadata[idx]
        }

        if self.include_fst:
            sample['fst'] = self.fst[idx]

        return sample


class MockMedNodeDataset(Dataset):
    """
    Mock MedNode dataset for multi-source testing.

    Simulates the MedNode dataset with different imaging conditions.
    """

    def __init__(
        self,
        num_samples: int = 50,
        num_classes: int = 7,
        image_size: int = 224,
        seed: Optional[int] = 42
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

        # Generate images with different characteristics than HAM10000
        # (simulating different imaging equipment/conditions)
        self.images = torch.randn(num_samples, 3, image_size, image_size) * 0.3 + 0.6
        self.images = torch.clamp(self.images, 0, 1)

        self.labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
        self.fst = torch.randint(1, 7, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'fst': self.fst[idx]
        }


def create_mock_dataloader(
    dataset_type: str = 'ham10000',
    num_samples: int = 100,
    batch_size: int = 16,
    shuffle: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with mock data.

    Args:
        dataset_type: Type of dataset ('ham10000' or 'mednodee')
        num_samples: Number of samples
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        DataLoader instance with mock data
    """
    if dataset_type == 'ham10000':
        dataset = MockHAM10000(num_samples=num_samples, **kwargs)
    elif dataset_type == 'mednode':
        dataset = MockMedNodeDataset(num_samples=num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # No multiprocessing for tests
    )


def generate_mock_predictions(
    num_samples: int = 100,
    num_classes: int = 7,
    calibrated: bool = True,
    seed: Optional[int] = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate mock model predictions and ground truth labels.

    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        calibrated: If True, predictions are well-calibrated (confidence matches accuracy)
        seed: Random seed

    Returns:
        Tuple of (predictions, labels) where predictions are probability distributions
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    labels = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)

    if calibrated:
        # Well-calibrated predictions
        predictions = torch.softmax(torch.randn(num_samples, num_classes) * 2, dim=1)
        # Increase probability for correct class
        for i in range(num_samples):
            predictions[i, labels[i]] += 0.5
        predictions = predictions / predictions.sum(dim=1, keepdim=True)
    else:
        # Poorly calibrated (overconfident) predictions
        predictions = torch.softmax(torch.randn(num_samples, num_classes) * 5, dim=1)

    return predictions, labels


def generate_mock_fst_stratified_data(
    num_samples: int = 600,
    num_classes: int = 7,
    ensure_balance: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, torch.Tensor]:
    """
    Generate mock data with stratified FST distribution for fairness testing.

    Args:
        num_samples: Total number of samples
        num_classes: Number of disease classes
        ensure_balance: If True, ensure equal representation across FST groups
        seed: Random seed

    Returns:
        Dictionary with 'images', 'labels', 'fst', 'predictions'
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate FST labels (I-VI)
    if ensure_balance:
        # Equal samples per FST
        fst = torch.cat([torch.full((num_samples // 6,), i) for i in range(1, 7)])
    else:
        # Imbalanced (reflecting real-world distribution)
        fst_probs = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]  # More Type III-IV
        fst = torch.multinomial(
            torch.tensor(fst_probs),
            num_samples,
            replacement=True
        ) + 1

    # Generate images
    images = torch.randn(len(fst), 3, 224, 224)

    # Generate labels
    labels = torch.randint(0, num_classes, (len(fst),), dtype=torch.long)

    # Generate predictions with potential FST-based bias
    predictions = torch.softmax(torch.randn(len(fst), num_classes), dim=1)

    return {
        'images': images,
        'labels': labels,
        'fst': fst,
        'predictions': predictions
    }
