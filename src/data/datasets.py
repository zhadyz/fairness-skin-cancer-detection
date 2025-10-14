"""
PyTorch Dataset classes for dermoscopic image datasets.

Supports:
- HAM10000 (Human Against Machine with 10,000 Dermoscopic Images)
- ISIC 2019 Challenge Dataset
- Fitzpatrick17k (with FST labels)
- DDI (Diverse Dermatology Images)
- MIDAS (Multimodal Image Dataset for AI-based Skin Cancer)
- SCIN (Skin Condition Image Network)
- Mixed datasets with real + synthetic data

Framework: MENDICANT_BIAS - the_didact research division
Version: 1.0
Date: 2025-10-13
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class BaseD ermoscopyDataset(Dataset):
    """
    Base class for dermoscopic image datasets.

    Provides common functionality:
    - Image loading from file paths
    - Label encoding
    - FST metadata handling
    - Transform application
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dir: str,
        image_col: str = 'image_id',
        label_col: str = 'diagnosis',
        fst_col: Optional[str] = 'fitzpatrick_skin_type',
        transform: Optional[Callable] = None,
        image_ext: str = '.jpg',
    ):
        """
        Initialize base dataset.

        Args:
            metadata_df: DataFrame with image metadata
            image_dir: Directory containing images
            image_col: Column name for image filename/ID
            label_col: Column name for diagnosis labels
            fst_col: Column name for Fitzpatrick Skin Type (if available)
            transform: Albumentations or torchvision transform
            image_ext: Image file extension
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.fst_col = fst_col
        self.transform = transform
        self.image_ext = image_ext

        # Encode labels as integers
        self.label_encoder = {label: idx for idx, label in enumerate(sorted(self.metadata_df[label_col].unique()))}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        self.num_classes = len(self.label_encoder)

        print(f"Loaded {len(self.metadata_df)} samples")
        print(f"Classes: {list(self.label_encoder.keys())}")
        if self.fst_col and self.fst_col in self.metadata_df.columns:
            print(f"FST distribution: {self.metadata_df[self.fst_col].value_counts().sort_index().to_dict()}")

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Optional[int]]:
        """
        Get item by index.

        Returns:
            image: Transformed image tensor (C, H, W)
            label: Diagnosis label (integer)
            fst: Fitzpatrick Skin Type (integer, or None if not available)
        """
        row = self.metadata_df.iloc[idx]

        # Load image
        image_id = row[self.image_col]
        image_path = self.image_dir / f"{image_id}{self.image_ext}"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get label
        label = self.label_encoder[row[self.label_col]]

        # Get FST (if available)
        fst = row[self.fst_col] if self.fst_col and self.fst_col in row else None

        # Apply transform
        if self.transform:
            if hasattr(self.transform, '__call__') and 'image' in self.transform.__call__.__code__.co_varnames:
                # Albumentations transform
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Torchvision transform
                image = Image.fromarray(image)
                image = self.transform(image)

        return image, label, fst

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of diagnosis labels."""
        return self.metadata_df[self.label_col].value_counts().to_dict()

    def get_fst_distribution(self) -> Dict[int, int]:
        """Get distribution of Fitzpatrick Skin Types."""
        if self.fst_col and self.fst_col in self.metadata_df.columns:
            return self.metadata_df[self.fst_col].value_counts().sort_index().to_dict()
        return {}


class HAM10000Dataset(BaseDermoscopyDataset):
    """
    HAM10000 (Human Against Machine with 10,000 Dermoscopic Images) dataset.

    10,015 dermoscopic images of 7 diagnostic categories:
    - akiec (Actinic keratoses and intraepithelial carcinoma)
    - bcc (Basal cell carcinoma)
    - bkl (Benign keratosis-like lesions)
    - df (Dermatofibroma)
    - mel (Melanoma)
    - nv (Melanocytic nevi)
    - vasc (Vascular lesions)

    Metadata columns:
    - image_id: Image filename (without extension)
    - dx: Diagnosis abbreviation
    - dx_type: Diagnosis confirmation method
    - age: Patient age
    - sex: Patient sex
    - localization: Anatomical location
    """

    def __init__(
        self,
        data_dir: str,
        metadata_csv: str = 'HAM10000_metadata.csv',
        fst_col: Optional[str] = None,  # HAM10000 doesn't have native FST labels
        transform: Optional[Callable] = None,
    ):
        """
        Initialize HAM10000 dataset.

        Args:
            data_dir: Root directory containing images and metadata
            metadata_csv: Metadata CSV filename
            fst_col: Column name for FST (if annotated separately)
            transform: Albumentations or torchvision transform
        """
        data_path = Path(data_dir)
        metadata_path = data_path / metadata_csv

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata_df = pd.read_csv(metadata_path)

        # HAM10000-specific preprocessing
        if 'dx' in metadata_df.columns:
            # Rename dx to diagnosis for consistency
            metadata_df['diagnosis'] = metadata_df['dx']

        super().__init__(
            metadata_df=metadata_df,
            image_dir=data_path,
            image_col='image_id',
            label_col='diagnosis',
            fst_col=fst_col,
            transform=transform,
            image_ext='.jpg',
        )


class ISIC2019Dataset(BaseDermoscopyDataset):
    """
    ISIC 2019 Challenge dataset.

    25,331 training images across 8 diagnostic categories:
    - MEL (Melanoma)
    - NV (Melanocytic nevus)
    - BCC (Basal cell carcinoma)
    - AK (Actinic keratosis)
    - BKL (Benign keratosis)
    - DF (Dermatofibroma)
    - VASC (Vascular lesion)
    - SCC (Squamous cell carcinoma)

    Metadata columns:
    - image: Image filename (with .jpg extension)
    - One-hot encoded diagnosis columns
    """

    def __init__(
        self,
        data_dir: str,
        metadata_csv: str = 'ISIC_2019_Training_GroundTruth.csv',
        fst_col: Optional[str] = None,  # ISIC 2019 doesn't have native FST labels
        transform: Optional[Callable] = None,
    ):
        """
        Initialize ISIC 2019 dataset.

        Args:
            data_dir: Root directory containing images and metadata
            metadata_csv: Metadata CSV filename
            fst_col: Column name for FST (if annotated separately)
            transform: Albumentations or torchvision transform
        """
        data_path = Path(data_dir)
        metadata_path = data_path / metadata_csv

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata_df = pd.read_csv(metadata_path)

        # ISIC 2019-specific preprocessing: Convert one-hot to single column
        diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
        if all(col in metadata_df.columns for col in diagnosis_cols):
            metadata_df['diagnosis'] = metadata_df[diagnosis_cols].idxmax(axis=1)

        # Remove .jpg extension from image column for consistency
        if 'image' in metadata_df.columns:
            metadata_df['image_id'] = metadata_df['image'].str.replace('.jpg', '', regex=False)

        super().__init__(
            metadata_df=metadata_df,
            image_dir=data_path,
            image_col='image_id',
            label_col='diagnosis',
            fst_col=fst_col,
            transform=transform,
            image_ext='.jpg',
        )


class Fitzpatrick17kDataset(BaseDermoscopyDataset):
    """
    Fitzpatrick17k dataset with native FST labels.

    16,577 clinical images with Fitzpatrick Skin Type annotations (I-VI).

    Metadata columns:
    - url: Original image URL
    - fitzpatrick: FST label (0-5 for FST I-VI)
    - label: Diagnosis
    """

    def __init__(
        self,
        data_dir: str,
        metadata_csv: str = 'fitzpatrick17k.csv',
        transform: Optional[Callable] = None,
    ):
        """
        Initialize Fitzpatrick17k dataset.

        Args:
            data_dir: Root directory containing images and metadata
            metadata_csv: Metadata CSV filename
            transform: Albumentations or torchvision transform
        """
        data_path = Path(data_dir)
        metadata_path = data_path / metadata_csv

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata_df = pd.read_csv(metadata_path)

        # Fitzpatrick17k-specific preprocessing
        if 'fitzpatrick' in metadata_df.columns:
            # FST labels are 0-5, convert to 1-6 for consistency
            metadata_df['fitzpatrick_skin_type'] = metadata_df['fitzpatrick'] + 1

        if 'label' in metadata_df.columns:
            metadata_df['diagnosis'] = metadata_df['label']

        # Generate image_id from URL or use md5 hash
        if 'md5hash' in metadata_df.columns:
            metadata_df['image_id'] = metadata_df['md5hash']
        else:
            metadata_df['image_id'] = metadata_df.index.astype(str)

        super().__init__(
            metadata_df=metadata_df,
            image_dir=data_path,
            image_col='image_id',
            label_col='diagnosis',
            fst_col='fitzpatrick_skin_type',
            transform=transform,
            image_ext='.jpg',
        )


class DDIDataset(BaseDermoscopyDataset):
    """
    DDI (Diverse Dermatology Images) dataset from Stanford.

    656 images with excellent FST diversity (34% FST V-VI).
    Pathologically confirmed diagnoses.

    Metadata columns:
    - image_id: Image filename
    - diagnosis: Pathology-confirmed diagnosis
    - fitzpatrick: FST label (I-VI)
    """

    def __init__(
        self,
        data_dir: str,
        metadata_csv: str = 'ddi_metadata.csv',
        transform: Optional[Callable] = None,
    ):
        """
        Initialize DDI dataset.

        Args:
            data_dir: Root directory containing images and metadata
            metadata_csv: Metadata CSV filename
            transform: Albumentations or torchvision transform
        """
        data_path = Path(data_dir)
        metadata_path = data_path / metadata_csv

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata_df = pd.read_csv(metadata_path)

        # DDI-specific preprocessing
        if 'fitzpatrick' in metadata_df.columns:
            # Convert Roman numerals (I-VI) to integers (1-6) if needed
            fst_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
            if metadata_df['fitzpatrick'].dtype == 'object':
                metadata_df['fitzpatrick_skin_type'] = metadata_df['fitzpatrick'].map(fst_map)
            else:
                metadata_df['fitzpatrick_skin_type'] = metadata_df['fitzpatrick']

        super().__init__(
            metadata_df=metadata_df,
            image_dir=data_path,
            image_col='image_id',
            label_col='diagnosis',
            fst_col='fitzpatrick_skin_type',
            transform=transform,
            image_ext='.jpg',
        )


class MixedDermoscopyDataset(Dataset):
    """
    Mixed dataset combining real + synthetic images.

    Useful for Phase 2 fairness training with synthetic augmentation.

    Combines multiple datasets and tracks data source (real vs synthetic).
    """

    def __init__(
        self,
        real_datasets: List[Dataset],
        synthetic_dataset: Optional[Dataset] = None,
        sampling_weights: Optional[List[float]] = None,
    ):
        """
        Initialize mixed dataset.

        Args:
            real_datasets: List of real dermoscopy datasets
            synthetic_dataset: Optional synthetic dataset
            sampling_weights: Optional weights for balanced sampling
        """
        self.real_datasets = real_datasets
        self.synthetic_dataset = synthetic_dataset

        # Combine all datasets
        self.datasets = real_datasets.copy()
        if synthetic_dataset:
            self.datasets.append(synthetic_dataset)

        # Calculate cumulative sizes for indexing
        self.cumulative_sizes = np.cumsum([0] + [len(d) for d in self.datasets])
        self.total_size = self.cumulative_sizes[-1]

        print(f"Mixed dataset created: {self.total_size} total samples")
        print(f"  Real: {sum(len(d) for d in real_datasets)}")
        if synthetic_dataset:
            print(f"  Synthetic: {len(synthetic_dataset)}")

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int):
        """Get item from appropriate sub-dataset."""
        dataset_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        sample_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]


def create_balanced_sampler(dataset: BaseDermoscopyDataset, balance_by: str = 'label') -> WeightedRandomSampler:
    """
    Create weighted sampler for balanced mini-batches.

    Useful for handling class imbalance (e.g., melanoma vs nevi)
    or FST imbalance (e.g., oversample FST V-VI).

    Args:
        dataset: Dermoscopy dataset
        balance_by: 'label' (class balancing) or 'fst' (FST balancing)

    Returns:
        WeightedRandomSampler for DataLoader
    """
    if balance_by == 'label':
        # Balance by diagnosis labels
        labels = [dataset[i][1] for i in range(len(dataset))]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]
    elif balance_by == 'fst':
        # Balance by Fitzpatrick Skin Type
        fst_labels = [dataset[i][2] for i in range(len(dataset)) if dataset[i][2] is not None]
        if not fst_labels:
            raise ValueError("Dataset does not have FST labels")
        fst_counts = np.bincount(fst_labels)
        fst_weights = 1.0 / fst_counts
        sample_weights = [fst_weights[fst] if fst is not None else 1.0 for _, _, fst in dataset]
    else:
        raise ValueError(f"Invalid balance_by: {balance_by}. Choose 'label' or 'fst'.")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    balanced_sampling: bool = False,
    balance_by: str = 'label',
) -> DataLoader:
    """
    Create PyTorch DataLoader with optional balanced sampling.

    Args:
        dataset: Dermoscopy dataset
        batch_size: Batch size
        shuffle: Shuffle data (ignored if balanced_sampling=True)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        balanced_sampling: Use weighted sampler for balanced batches
        balance_by: 'label' or 'fst' (only used if balanced_sampling=True)

    Returns:
        DataLoader
    """
    if balanced_sampling:
        sampler = create_balanced_sampler(dataset, balance_by=balance_by)
        shuffle = False  # Mutually exclusive with sampler
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader


if __name__ == "__main__":
    # Demo: Test dataset loading
    print("Dermoscopy Dataset Module")
    print("=" * 60)

    # Example usage (requires actual data files)
    # Uncomment when data is available

    # from preprocessing import get_training_augmentation, get_validation_transform
    #
    # # Load HAM10000
    # transform = get_training_augmentation(image_size=224)
    # dataset = HAM10000Dataset(
    #     data_dir='data/raw/ham10000/',
    #     transform=transform
    # )
    #
    # print(f"\nDataset size: {len(dataset)}")
    # print(f"Label distribution: {dataset.get_label_distribution()}")
    #
    # # Create DataLoader
    # dataloader = create_dataloader(
    #     dataset,
    #     batch_size=32,
    #     balanced_sampling=True,
    #     balance_by='label'
    # )
    #
    # # Test batch loading
    # images, labels, fsts = next(iter(dataloader))
    # print(f"\nBatch shape: {images.shape}")
    # print(f"Label distribution in batch: {np.bincount(labels.numpy())}")

    print("\n" + "=" * 60)
    print("Dataset module loaded successfully.")
    print("Uncomment demo code and provide data paths to test.")
    print("=" * 60)
