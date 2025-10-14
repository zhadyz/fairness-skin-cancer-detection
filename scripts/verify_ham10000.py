"""
Comprehensive HAM10000 dataset verification and integrity check.

This script verifies:
1. Dataset structure and file integrity
2. Metadata completeness
3. Image loading and format validation
4. Diagnosis and FST distributions
5. Split integrity (no data leakage)
6. Image statistics (mean, std, size distribution)
7. Lesion-level consistency

Framework: MENDICANT_BIAS - Phase 1.5
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import os
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.ham10000_dataset import HAM10000_DIAGNOSIS_LABELS, HAM10000_LABEL_NAMES


class HAM10000Verifier:
    """Comprehensive HAM10000 dataset verification."""

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        splits_path: Optional[str] = None,
    ):
        """
        Initialize verifier.

        Args:
            data_dir: HAM10000 root directory
            metadata_path: Path to metadata CSV
            splits_path: Optional path to splits JSON
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = Path(metadata_path)
        self.splits_path = Path(splits_path) if splits_path else None

        self.metadata = None
        self.splits = None
        self.image_paths = {}

        self.errors = []
        self.warnings = []

    def load_data(self):
        """Load metadata and splits."""
        print("Loading metadata and splits...")

        # Load metadata
        if not self.metadata_path.exists():
            self.errors.append(f"Metadata file not found: {self.metadata_path}")
            return False

        self.metadata = pd.read_csv(self.metadata_path)
        print(f"  Loaded metadata: {len(self.metadata)} samples")

        # Load splits (optional)
        if self.splits_path and self.splits_path.exists():
            with open(self.splits_path, 'r') as f:
                self.splits = json.load(f)
            print(f"  Loaded splits: train={len(self.splits['train'])}, "
                  f"val={len(self.splits['val'])}, test={len(self.splits['test'])}")
        else:
            print("  No splits file provided (skipping split verification)")

        return True

    def verify_directory_structure(self) -> bool:
        """Verify HAM10000 directory structure."""
        print("\n[1/8] Verifying directory structure...")

        if not self.data_dir.exists():
            self.errors.append(f"Data directory not found: {self.data_dir}")
            return False

        # Check for image directories
        image_parts = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
        found_parts = 0

        for part in image_parts:
            part_path = self.data_dir / part
            if part_path.exists():
                found_parts += 1
                # Index images
                for img_file in part_path.glob("*.jpg"):
                    self.image_paths[img_file.stem] = img_file
                print(f"  Found: {part} ({len(list(part_path.glob('*.jpg')))} images)")
            else:
                self.warnings.append(f"Image directory not found: {part}")

        print(f"  Total images found: {len(self.image_paths)}")

        if found_parts == 0:
            self.errors.append("No image directories found")
            return False

        return True

    def verify_metadata(self) -> bool:
        """Verify metadata completeness and validity."""
        print("\n[2/8] Verifying metadata...")

        # Check required columns
        required_cols = ['image_id', 'dx']
        missing_cols = [col for col in required_cols if col not in self.metadata.columns]

        if missing_cols:
            self.errors.append(f"Missing required columns: {missing_cols}")
            return False

        print(f"  Metadata columns: {list(self.metadata.columns)}")

        # Verify diagnosis labels
        invalid_dx = []
        for dx in self.metadata['dx'].unique():
            if dx not in HAM10000_DIAGNOSIS_LABELS:
                invalid_dx.append(dx)

        if invalid_dx:
            self.errors.append(f"Invalid diagnosis codes: {invalid_dx}")
            return False

        print(f"  All diagnosis labels valid: {list(HAM10000_DIAGNOSIS_LABELS.keys())}")

        # Check for missing values
        for col in self.metadata.columns:
            missing = self.metadata[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} missing values ({missing/len(self.metadata)*100:.2f}%)")

        return True

    def verify_image_integrity(self, sample_size: int = 100) -> bool:
        """Verify image files can be loaded."""
        print(f"\n[3/8] Verifying image integrity (sampling {sample_size} images)...")

        # Sample images
        sample_indices = np.random.choice(
            len(self.metadata),
            size=min(sample_size, len(self.metadata)),
            replace=False
        )

        failed_loads = []
        image_shapes = []

        for idx in tqdm(sample_indices, desc="Loading images"):
            image_id = self.metadata.iloc[idx]['image_id']

            if image_id not in self.image_paths:
                failed_loads.append((image_id, "File not found"))
                continue

            try:
                img_path = self.image_paths[image_id]
                img = cv2.imread(str(img_path))

                if img is None:
                    failed_loads.append((image_id, "Failed to load"))
                else:
                    image_shapes.append(img.shape)

            except Exception as e:
                failed_loads.append((image_id, str(e)))

        if failed_loads:
            self.errors.append(f"{len(failed_loads)} images failed to load")
            for image_id, error in failed_loads[:10]:  # Show first 10
                print(f"  Failed: {image_id} - {error}")
            return False

        # Analyze image dimensions
        if image_shapes:
            heights = [s[0] for s in image_shapes]
            widths = [s[1] for s in image_shapes]
            channels = [s[2] for s in image_shapes]

            print(f"  All {len(image_shapes)} sampled images loaded successfully")
            print(f"  Image dimensions:")
            print(f"    Height:   {np.min(heights)} - {np.max(heights)} (mean: {np.mean(heights):.0f})")
            print(f"    Width:    {np.min(widths)} - {np.max(widths)} (mean: {np.mean(widths):.0f})")
            print(f"    Channels: {np.unique(channels).tolist()}")

        return True

    def verify_diagnosis_distribution(self) -> bool:
        """Verify diagnosis class distribution."""
        print("\n[4/8] Verifying diagnosis distribution...")

        # Map diagnosis to labels
        self.metadata['label'] = self.metadata['dx'].map(HAM10000_DIAGNOSIS_LABELS)

        diagnosis_counts = self.metadata['dx'].value_counts().sort_index()

        print(f"  Diagnosis distribution:")
        for dx, count in diagnosis_counts.items():
            print(f"    {dx:6s}: {count:5d} ({count/len(self.metadata)*100:5.2f}%)")

        # Check for severe class imbalance
        max_count = diagnosis_counts.max()
        min_count = diagnosis_counts.min()
        imbalance_ratio = max_count / min_count

        print(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 100:
            self.warnings.append(f"Severe class imbalance detected: {imbalance_ratio:.2f}:1")

        return True

    def verify_fst_distribution(self) -> bool:
        """Verify FST distribution (if available)."""
        print("\n[5/8] Verifying FST distribution...")

        if 'fst' not in self.metadata.columns:
            print("  FST column not found in metadata (skipping)")
            return True

        fst_valid = self.metadata[self.metadata['fst'] != -1]
        fst_unknown = len(self.metadata) - len(fst_valid)

        if len(fst_valid) == 0:
            self.warnings.append("No FST annotations found")
            return True

        fst_counts = fst_valid['fst'].value_counts().sort_index()

        print(f"  FST distribution:")
        for fst, count in fst_counts.items():
            print(f"    FST {fst}: {count:5d} ({count/len(fst_valid)*100:5.2f}%)")

        if fst_unknown > 0:
            print(f"    Unknown: {fst_unknown:5d} ({fst_unknown/len(self.metadata)*100:5.2f}%)")

        # Check for underrepresented FSTs
        min_fst_count = fst_counts.min()
        if min_fst_count < len(self.metadata) * 0.01:  # Less than 1%
            self.warnings.append(f"Underrepresented FST detected: {min_fst_count} samples")

        return True

    def verify_splits_integrity(self) -> bool:
        """Verify train/val/test splits have no data leakage."""
        print("\n[6/8] Verifying split integrity...")

        if self.splits is None:
            print("  No splits to verify (skipping)")
            return True

        train_indices = set(self.splits['train'])
        val_indices = set(self.splits['val'])
        test_indices = set(self.splits['test'])

        # Check for overlaps
        train_val_overlap = train_indices & val_indices
        train_test_overlap = train_indices & test_indices
        val_test_overlap = val_indices & test_indices

        if train_val_overlap:
            self.errors.append(f"Train-Val overlap: {len(train_val_overlap)} samples")

        if train_test_overlap:
            self.errors.append(f"Train-Test overlap: {len(train_test_overlap)} samples")

        if val_test_overlap:
            self.errors.append(f"Val-Test overlap: {len(val_test_overlap)} samples")

        if not (train_val_overlap or train_test_overlap or val_test_overlap):
            print("  No overlap between splits (correct)")

        # Check total coverage
        all_indices = train_indices | val_indices | test_indices
        if len(all_indices) != len(self.metadata):
            self.warnings.append(
                f"Splits don't cover all samples: {len(all_indices)} vs {len(self.metadata)}"
            )

        # Verify lesion-level split integrity (if lesion_id available)
        if 'lesion_id' in self.metadata.columns:
            print("  Verifying lesion-level split integrity...")

            train_lesions = set(self.metadata.iloc[list(train_indices)]['lesion_id'])
            val_lesions = set(self.metadata.iloc[list(val_indices)]['lesion_id'])
            test_lesions = set(self.metadata.iloc[list(test_indices)]['lesion_id'])

            lesion_train_val = train_lesions & val_lesions
            lesion_train_test = train_lesions & test_lesions
            lesion_val_test = val_lesions & test_lesions

            if lesion_train_val:
                self.errors.append(f"Lesion leakage Train-Val: {len(lesion_train_val)} lesions")

            if lesion_train_test:
                self.errors.append(f"Lesion leakage Train-Test: {len(lesion_train_test)} lesions")

            if lesion_val_test:
                self.errors.append(f"Lesion leakage Val-Test: {len(lesion_val_test)} lesions")

            if not (lesion_train_val or lesion_train_test or lesion_val_test):
                print("  No lesion-level data leakage (correct)")

        return len(self.errors) == 0

    def compute_image_statistics(self, sample_size: int = 500) -> bool:
        """Compute image statistics (mean, std, size distribution)."""
        print(f"\n[7/8] Computing image statistics (sampling {sample_size} images)...")

        sample_indices = np.random.choice(
            len(self.metadata),
            size=min(sample_size, len(self.metadata)),
            replace=False
        )

        pixel_means = []
        pixel_stds = []

        for idx in tqdm(sample_indices, desc="Computing statistics"):
            image_id = self.metadata.iloc[idx]['image_id']

            if image_id not in self.image_paths:
                continue

            try:
                img = cv2.imread(str(self.image_paths[image_id]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0

                pixel_means.append(img.mean(axis=(0, 1)))
                pixel_stds.append(img.std(axis=(0, 1)))

            except Exception:
                continue

        if pixel_means:
            mean_rgb = np.mean(pixel_means, axis=0)
            std_rgb = np.mean(pixel_stds, axis=0)

            print(f"  Dataset statistics (RGB, normalized [0, 1]):")
            print(f"    Mean: [{mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f}]")
            print(f"    Std:  [{std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f}]")

        return True

    def verify_all(self) -> bool:
        """Run all verification checks."""
        print("=" * 70)
        print("HAM10000 DATASET VERIFICATION")
        print("=" * 70)

        if not self.load_data():
            return False

        # Run all checks
        checks = [
            self.verify_directory_structure(),
            self.verify_metadata(),
            self.verify_image_integrity(),
            self.verify_diagnosis_distribution(),
            self.verify_fst_distribution(),
            self.verify_splits_integrity(),
            self.compute_image_statistics(),
        ]

        # Final summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\nAll checks PASSED")
            print("Dataset is ready for training")
            success = True
        elif not self.errors:
            print("\nAll checks PASSED with warnings")
            print("Dataset is usable but review warnings")
            success = True
        else:
            print("\nVerification FAILED")
            print("Fix errors before proceeding")
            success = False

        print("=" * 70)

        return success


def main():
    parser = argparse.ArgumentParser(
        description="Verify HAM10000 dataset integrity"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ham10000",
        help="HAM10000 root directory"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/ham10000_fst_estimated.csv",
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/metadata/ham10000_splits.json",
        help="Path to splits JSON (optional)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of images to sample for integrity check"
    )

    args = parser.parse_args()

    # Create verifier
    verifier = HAM10000Verifier(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        splits_path=args.splits if Path(args.splits).exists() else None,
    )

    # Run verification
    success = verifier.verify_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    from typing import Optional
    main()
