"""
HAM10000 Dataset Implementation with FST Annotations.

Complete PyTorch Dataset class for HAM10000 (Human Against Machine 10000)
dermoscopic image dataset with Fitzpatrick Skin Type (FST) annotation support.

Dataset: 10,015 images across 7 diagnostic categories:
- akiec (Actinic keratoses and intraepithelial carcinoma)
- bcc (Basal cell carcinoma)
- bkl (Benign keratosis-like lesions)
- df (Dermatofibroma)
- mel (Melanoma)
- nv (Melanocytic nevi)
- vasc (Vascular lesions)

FST Annotation Strategy:
- ITA-based estimation (calculate_ita from fst_annotation.py)
- Support for external FST annotations (if available)
- Missing FST handling with -1 sentinel value

Framework: MENDICANT_BIAS - Phase 1.5
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, List, Union
import warnings

import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .fst_annotation import calculate_ita, ita_to_fst


# HAM10000 diagnosis label mapping (7 classes)
HAM10000_DIAGNOSIS_LABELS = {
    'akiec': 0,  # Actinic keratoses and intraepithelial carcinoma
    'bcc': 1,    # Basal cell carcinoma
    'bkl': 2,    # Benign keratosis-like lesions
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma
    'nv': 5,     # Melanocytic nevi
    'vasc': 6,   # Vascular lesions
}

# Reverse mapping
HAM10000_LABEL_NAMES = {v: k for k, v in HAM10000_DIAGNOSIS_LABELS.items()}


class HAM10000Dataset(Dataset):
    """
    HAM10000 skin lesion dataset with FST annotations.

    Supports:
    - Loading images from HAM10000 directory structure
    - Parsing HAM10000_metadata.csv
    - ITA-based FST estimation
    - External FST annotation integration
    - Train/val/test splits (stratified)
    - Comprehensive metadata access
    """

    def __init__(
        self,
        root_dir: str,
        metadata_path: Optional[str] = None,
        split: str = "train",
        split_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        use_fst_annotations: bool = True,
        fst_csv_path: Optional[str] = None,
        estimate_fst_if_missing: bool = True,
        image_parts: List[str] = ["HAM10000_images_part_1", "HAM10000_images_part_2"],
    ):
        """
        Initialize HAM10000 dataset.

        Args:
            root_dir: Path to HAM10000 root directory containing images and metadata
            metadata_path: Path to HAM10000_metadata.csv (defaults to root_dir/HAM10000_metadata.csv)
            split: Dataset split ("train", "val", or "test")
            split_indices: Specific indices for this split (if using external splits)
            transform: Image transformations (albumentations or torchvision)
            use_fst_annotations: Whether to load/estimate FST labels
            fst_csv_path: Path to external FST annotations CSV (image_id, fst)
            estimate_fst_if_missing: Estimate FST using ITA if not provided
            image_parts: List of image subdirectories (HAM10000 splits images into parts)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_fst_annotations = use_fst_annotations
        self.estimate_fst_if_missing = estimate_fst_if_missing
        self.image_parts = image_parts

        # Load metadata
        if metadata_path is None:
            metadata_path = self.root_dir / "HAM10000_metadata.csv"
        else:
            metadata_path = Path(metadata_path)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"HAM10000 metadata not found at {metadata_path}. "
                "Please download HAM10000 dataset from: "
                "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
            )

        self.metadata = pd.read_csv(metadata_path)

        # Validate required columns
        required_cols = ['image_id', 'dx']
        missing_cols = [col for col in required_cols if col not in self.metadata.columns]
        if missing_cols:
            raise ValueError(f"Metadata missing required columns: {missing_cols}")

        # Filter by split indices if provided
        if split_indices is not None:
            self.metadata = self.metadata.iloc[split_indices].reset_index(drop=True)

        # Map diagnosis labels to integers
        self.metadata['label'] = self.metadata['dx'].map(HAM10000_DIAGNOSIS_LABELS)

        # Verify all diagnoses are valid
        if self.metadata['label'].isna().any():
            invalid_dx = self.metadata.loc[self.metadata['label'].isna(), 'dx'].unique()
            raise ValueError(f"Invalid diagnosis codes found: {invalid_dx}")

        self.metadata['label'] = self.metadata['label'].astype(int)

        # Load or estimate FST annotations
        self.metadata['fst'] = -1  # Default: unknown

        if use_fst_annotations:
            if fst_csv_path and Path(fst_csv_path).exists():
                # Load external FST annotations
                fst_df = pd.read_csv(fst_csv_path)
                if 'image_id' in fst_df.columns and 'fst' in fst_df.columns:
                    fst_mapping = dict(zip(fst_df['image_id'], fst_df['fst']))
                    self.metadata['fst'] = self.metadata['image_id'].map(fst_mapping).fillna(-1).astype(int)
                    print(f"Loaded FST annotations from {fst_csv_path}")
                else:
                    warnings.warn(f"FST CSV missing required columns (image_id, fst)")

            # Estimate missing FST values using ITA
            if estimate_fst_if_missing and (self.metadata['fst'] == -1).any():
                print(f"Estimating FST for {(self.metadata['fst'] == -1).sum()} images using ITA...")
                self._estimate_fst_labels()

        # Build image path mapping
        self._build_image_path_map()

        # Store class statistics
        self.num_classes = len(HAM10000_DIAGNOSIS_LABELS)

        print(f"\nHAM10000 Dataset [{split}]")
        print(f"  Total samples: {len(self.metadata)}")
        print(f"  Diagnosis distribution:")
        for label, count in self.metadata['label'].value_counts().sort_index().items():
            print(f"    {HAM10000_LABEL_NAMES[label]:6s}: {count:5d} ({count/len(self.metadata)*100:5.2f}%)")

        if use_fst_annotations:
            print(f"  FST distribution:")
            fst_counts = self.metadata[self.metadata['fst'] != -1]['fst'].value_counts().sort_index()
            for fst, count in fst_counts.items():
                print(f"    FST {fst}: {count:5d} ({count/len(self.metadata)*100:5.2f}%)")
            unknown_fst = (self.metadata['fst'] == -1).sum()
            if unknown_fst > 0:
                print(f"    Unknown: {unknown_fst:5d} ({unknown_fst/len(self.metadata)*100:5.2f}%)")

    def _build_image_path_map(self):
        """Build mapping from image_id to full file path."""
        self.image_path_map = {}

        # Search all image parts for files
        for part_dir in self.image_parts:
            part_path = self.root_dir / part_dir
            if part_path.exists():
                for img_file in part_path.glob("*.jpg"):
                    image_id = img_file.stem
                    self.image_path_map[image_id] = img_file

        # Verify all metadata images exist
        missing_images = []
        for image_id in self.metadata['image_id']:
            if image_id not in self.image_path_map:
                missing_images.append(image_id)

        if missing_images:
            warnings.warn(
                f"{len(missing_images)} images not found in {self.root_dir}. "
                f"Example missing: {missing_images[:5]}"
            )

    def _estimate_fst_labels(self):
        """Estimate FST labels using ITA calculation for images missing FST."""
        for idx in self.metadata[self.metadata['fst'] == -1].index:
            image_id = self.metadata.loc[idx, 'image_id']

            if image_id not in self.image_path_map:
                continue

            try:
                # Load image
                image_path = self.image_path_map[image_id]
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Calculate ITA and map to FST
                ita = calculate_ita(image, exclude_lesion=True)
                fst = ita_to_fst(ita)

                self.metadata.loc[idx, 'fst'] = fst

            except Exception as e:
                warnings.warn(f"Failed to estimate FST for {image_id}: {e}")
                continue

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float, str]]:
        """
        Get dataset item by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: Tensor (C, H, W) - transformed image
                - label: int - diagnosis class (0-6)
                - fst: int - Fitzpatrick type (1-6) or -1 if unknown
                - lesion_id: str - lesion identifier
                - image_id: str - image identifier
                - age: float - patient age (or NaN)
                - sex: str - patient sex
                - localization: str - anatomical location
        """
        row = self.metadata.iloc[idx]

        # Load image
        image_id = row['image_id']
        image_path = self.image_path_map.get(image_id)

        if image_path is None:
            raise FileNotFoundError(f"Image not found for image_id: {image_id}")

        # Load as RGB
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is not None:
            # Check if albumentations or torchvision
            if hasattr(self.transform, '__call__'):
                # Try albumentations first (returns dict with 'image' key)
                try:
                    transformed = self.transform(image=image)
                    image = transformed['image']
                except (TypeError, KeyError):
                    # Fallback to torchvision (expects PIL Image)
                    image = Image.fromarray(image)
                    image = self.transform(image)

        # Ensure tensor format (C, H, W)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Prepare output
        sample = {
            'image': image,
            'label': int(row['label']),
            'fst': int(row['fst']),
            'lesion_id': row.get('lesion_id', ''),
            'image_id': image_id,
            'age': float(row.get('age', np.nan)),
            'sex': row.get('sex', ''),
            'localization': row.get('localization', ''),
        }

        return sample

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get diagnosis class distribution.

        Returns:
            Dictionary mapping class names to counts
        """
        label_counts = self.metadata['label'].value_counts()
        return {HAM10000_LABEL_NAMES[label]: count for label, count in label_counts.items()}

    def get_fst_distribution(self) -> Dict[int, int]:
        """
        Get Fitzpatrick Skin Type distribution.

        Returns:
            Dictionary mapping FST (1-6) to counts
        """
        if not self.use_fst_annotations:
            return {}

        fst_counts = self.metadata[self.metadata['fst'] != -1]['fst'].value_counts()
        return dict(sorted(fst_counts.items()))

    def get_label_name(self, label: int) -> str:
        """Convert integer label to diagnosis name."""
        return HAM10000_LABEL_NAMES.get(label, 'unknown')

    def get_metadata_df(self) -> pd.DataFrame:
        """Return full metadata DataFrame for analysis."""
        return self.metadata.copy()


def create_fst_stratified_splits(
    metadata_path: str,
    output_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_by_fst: bool = True,
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits stratified by diagnosis AND FST.

    Ensures:
    - Balanced diagnosis class distribution across splits
    - Proportional FST representation in each split
    - No data leakage (same lesion_id in single split only)
    - Reproducible splits (fixed random seed)

    Args:
        metadata_path: Path to HAM10000_metadata.csv
        output_path: Path to save split indices JSON
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.15)
        test_ratio: Test set ratio (default 0.15)
        random_seed: Random seed for reproducibility
        stratify_by_fst: Whether to stratify by FST (requires FST labels)

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to index lists
    """
    import json

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Map diagnosis to integers
    metadata['label'] = metadata['dx'].map(HAM10000_DIAGNOSIS_LABELS)

    # Create stratification column
    if stratify_by_fst and 'fst' in metadata.columns:
        # Stratify by both diagnosis and FST
        metadata['stratify_col'] = (
            metadata['label'].astype(str) + '_' +
            metadata['fst'].astype(str)
        )
    else:
        # Stratify by diagnosis only
        metadata['stratify_col'] = metadata['label'].astype(str)

    # Handle lesion_id to prevent data leakage
    # If lesion has multiple images, keep them in same split
    if 'lesion_id' in metadata.columns:
        # Group by lesion_id and assign split at lesion level
        lesion_groups = metadata.groupby('lesion_id').first().reset_index()

        # First split: train vs (val + test)
        train_lesions, temp_lesions = train_test_split(
            lesion_groups['lesion_id'],
            train_size=train_ratio,
            random_state=random_seed,
            stratify=lesion_groups['stratify_col'],
        )

        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            train_size=val_size,
            random_state=random_seed,
            stratify=lesion_groups.loc[lesion_groups['lesion_id'].isin(temp_lesions), 'stratify_col'],
        )

        # Get indices for each split
        train_indices = metadata[metadata['lesion_id'].isin(train_lesions)].index.tolist()
        val_indices = metadata[metadata['lesion_id'].isin(val_lesions)].index.tolist()
        test_indices = metadata[metadata['lesion_id'].isin(test_lesions)].index.tolist()

    else:
        # No lesion_id, split directly on images
        train_indices, temp_indices = train_test_split(
            metadata.index,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=metadata['stratify_col'],
        )

        val_size = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            random_state=random_seed,
            stratify=metadata.loc[temp_indices, 'stratify_col'],
        )

    # Prepare splits dictionary
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'metadata': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'random_seed': random_seed,
            'stratify_by_fst': stratify_by_fst,
            'total_samples': len(metadata),
        }
    }

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    # Print statistics
    print(f"\nCreated HAM10000 splits:")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices)/len(metadata)*100:.2f}%)")
    print(f"  Val:   {len(val_indices)} samples ({len(val_indices)/len(metadata)*100:.2f}%)")
    print(f"  Test:  {len(test_indices)} samples ({len(test_indices)/len(metadata)*100:.2f}%)")
    print(f"\nSaved to: {output_path}")

    return splits


def load_splits(splits_path: str) -> Dict[str, List[int]]:
    """
    Load train/val/test splits from JSON file.

    Args:
        splits_path: Path to splits JSON file

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    import json

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    return splits


if __name__ == "__main__":
    """
    Demo and testing of HAM10000Dataset.

    Usage:
        python -m src.data.ham10000_dataset
    """
    print("=" * 70)
    print("HAM10000 Dataset Module")
    print("=" * 70)

    # Check if HAM10000 data exists
    data_dir = Path("data/raw/ham10000")

    if not data_dir.exists():
        print("\nWARNING: HAM10000 data directory not found.")
        print("\nTo download HAM10000 dataset:")
        print("  1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("  2. Download HAM10000_images_part_1.zip")
        print("  3. Download HAM10000_images_part_2.zip")
        print("  4. Download HAM10000_metadata")
        print(f"  5. Extract to: {data_dir.absolute()}")
        print("\nExpected structure:")
        print("  data/raw/ham10000/")
        print("    ├── HAM10000_images_part_1/")
        print("    ├── HAM10000_images_part_2/")
        print("    └── HAM10000_metadata.csv")
    else:
        print(f"\nFound HAM10000 data directory: {data_dir.absolute()}")

        # Try to load dataset
        try:
            dataset = HAM10000Dataset(
                root_dir=str(data_dir),
                split="train",
                use_fst_annotations=True,
                estimate_fst_if_missing=True,
            )

            print(f"\nDataset loaded successfully!")
            print(f"Total samples: {len(dataset)}")

            # Show class distribution
            print("\nClass distribution:")
            for class_name, count in dataset.get_class_distribution().items():
                print(f"  {class_name}: {count}")

            # Show FST distribution
            fst_dist = dataset.get_fst_distribution()
            if fst_dist:
                print("\nFST distribution:")
                for fst, count in fst_dist.items():
                    print(f"  FST {fst}: {count}")

            # Load sample
            print("\nLoading sample...")
            sample = dataset[0]
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Label: {sample['label']} ({dataset.get_label_name(sample['label'])})")
            print(f"  FST: {sample['fst']}")
            print(f"  Image ID: {sample['image_id']}")

        except Exception as e:
            print(f"\nError loading dataset: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("HAM10000 dataset module test complete.")
    print("=" * 70)
