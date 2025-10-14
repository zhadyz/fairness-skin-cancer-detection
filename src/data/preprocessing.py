"""
Image preprocessing and augmentation pipeline for fairness-aware skin cancer detection.

This module provides:
- Standard preprocessing (resize, normalization) for dermoscopic images
- Advanced augmentation (RandAugment, color jittering) preserving diagnostic features
- FST-stratified data splitting with balanced representation
- Quality control and image validation

Framework: MENDICANT_BIAS - the_didact research division
Version: 1.0
Date: 2025-10-13
"""

import os
from typing import Tuple, Optional, List, Dict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ImageNet normalization statistics (standard for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard input size for most pre-trained models
DEFAULT_IMAGE_SIZE = 224


class DermoscopyPreprocessor:
    """
    Preprocessing pipeline for dermoscopic images.

    Handles:
    - Image loading and validation
    - Resizing to standard dimensions
    - Normalization using ImageNet statistics
    - Optional hair removal and artifact reduction
    """

    def __init__(
        self,
        image_size: int = DEFAULT_IMAGE_SIZE,
        normalize: bool = True,
        remove_hair: bool = False,
    ):
        """
        Initialize preprocessor.

        Args:
            image_size: Target image size (will resize to image_size x image_size)
            normalize: Apply ImageNet normalization
            remove_hair: Apply hair removal preprocessing (DullRazor-like algorithm)
        """
        self.image_size = image_size
        self.normalize = normalize
        self.remove_hair = remove_hair

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline.

        Args:
            image: Input image (H, W, C) in RGB format

        Returns:
            Preprocessed image (image_size, image_size, C)
        """
        # Hair removal (optional)
        if self.remove_hair:
            image = self._remove_hair(image)

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        if self.normalize:
            image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)

        return image

    def _remove_hair(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts using morphological operations (DullRazor-like).

        Args:
            image: Input image (H, W, C) in RGB format

        Returns:
            Image with hair artifacts reduced
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply blackhat morphological operation to detect hair
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold to create hair mask
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # Inpaint to remove hair
        image_inpainted = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return image_inpainted


def get_training_augmentation(image_size: int = DEFAULT_IMAGE_SIZE, advanced: bool = True) -> A.Compose:
    """
    Training augmentation pipeline using Albumentations.

    Carefully tuned to preserve diagnostic features (shape, border, color patterns)
    while providing sufficient regularization.

    Args:
        image_size: Target image size
        advanced: Use advanced augmentation (RandAugment-style)

    Returns:
        Albumentations composition of augmentations
    """
    if advanced:
        # Advanced augmentation for fairness-aware training
        augmentation = A.Compose([
            # Geometric transformations
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.Transpose(p=0.3),

            # Color augmentation (CRITICAL: Preserve diagnostic color information)
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.8),
            ], p=0.7),

            # Illumination augmentation (simulate different lighting conditions)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.8),
            ], p=0.5),

            # Noise and blur (regularization, avoid overfitting to artifact-free images)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.3),

            # Normalize and convert to tensor
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        # Basic augmentation (baseline)
        augmentation = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    return augmentation


def get_validation_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> A.Compose:
    """
    Validation/test transform (no augmentation, only resize and normalize).

    Args:
        image_size: Target image size

    Returns:
        Albumentations composition for validation
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def fst_stratified_split(
    data_df,
    fst_column: str = 'fitzpatrick_skin_type',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple:
    """
    Stratified train/val/test split ensuring balanced FST distribution.

    Critical for fairness evaluation: Each split must have representative
    samples from all FST groups (I-VI).

    Args:
        data_df: DataFrame with image paths, labels, and FST annotations
        fst_column: Column name containing FST labels
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    # First split: train+val vs test (stratified by FST)
    train_val_df, test_df = train_test_split(
        data_df,
        test_size=test_size,
        stratify=data_df[fst_column],
        random_state=random_state,
    )

    # Second split: train vs val (stratified by FST)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),  # Adjust val_size relative to train_val
        stratify=train_val_df[fst_column],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def validate_fst_distribution(train_df, val_df, test_df, fst_column: str = 'fitzpatrick_skin_type'):
    """
    Validate FST distribution across splits.

    Prints statistics to ensure no FST group is missing or severely underrepresented.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        fst_column: Column name containing FST labels
    """
    print("=" * 60)
    print("FST Distribution Validation")
    print("=" * 60)

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        fst_counts = split_df[fst_column].value_counts().sort_index()
        fst_percentages = (fst_counts / len(split_df) * 100).round(2)

        print(f"\n{split_name} Set (N={len(split_df)}):")
        for fst_type in sorted(split_df[fst_column].unique()):
            count = fst_counts.get(fst_type, 0)
            pct = fst_percentages.get(fst_type, 0.0)
            print(f"  FST {fst_type}: {count:5d} ({pct:5.2f}%)")

    print("=" * 60)


def apply_mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp augmentation (for advanced fairness training).

    MixUp: Linear interpolation between two images and their labels.
    Improves calibration and reduces overfitting.

    Reference: Zhang et al. (2018). "mixup: Beyond Empirical Risk Minimization". ICLR 2018.

    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        alpha: MixUp strength (Beta distribution parameter)

    Returns:
        mixed_images, labels_a, labels_b, lambda_param
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]

    return mixed_images, labels_a, labels_b, lam


def apply_cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation (for advanced fairness training).

    CutMix: Cut and paste patches between images, mix labels proportionally.
    Improves localization and calibration.

    Reference: Yun et al. (2019). "CutMix: Regularization Strategy to Train Strong Classifiers". ICCV 2019.

    Args:
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        alpha: CutMix strength (Beta distribution parameter)

    Returns:
        mixed_images, labels_a, labels_b, lambda_param
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)

    # Generate random bounding box
    W = images.size(3)
    H = images.size(2)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform sampling of center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    mixed_images = images.clone()
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    labels_a, labels_b = labels, labels[index]

    return mixed_images, labels_a, labels_b, lam


def save_preprocessed_images(
    input_dir: str,
    output_dir: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    remove_hair: bool = False,
):
    """
    Batch preprocess and save images.

    Useful for creating preprocessed dataset once, then loading during training.

    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save preprocessed images
        image_size: Target image size
        remove_hair: Apply hair removal preprocessing
    """
    preprocessor = DermoscopyPreprocessor(image_size=image_size, normalize=False, remove_hair=remove_hair)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

    print(f"Preprocessing {len(image_files)} images from {input_dir}...")

    for i, image_file in enumerate(image_files):
        # Load image
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        preprocessed = preprocessor(image)

        # Convert back to uint8 for saving
        preprocessed = (preprocessed * 255).astype(np.uint8)

        # Save
        output_file = output_path / image_file.name
        cv2.imwrite(str(output_file), cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")

    print(f"Preprocessing complete. Saved to {output_dir}")


if __name__ == "__main__":
    # Demo: Test preprocessing pipeline
    print("Dermoscopy Preprocessing Module")
    print("=" * 60)

    # Create dummy image for testing
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Test preprocessor
    preprocessor = DermoscopyPreprocessor(image_size=224, normalize=True, remove_hair=False)
    processed = preprocessor(dummy_image)
    print(f"Preprocessed image shape: {processed.shape}")
    print(f"Preprocessed image range: [{processed.min():.3f}, {processed.max():.3f}]")

    # Test augmentation
    augmentation = get_training_augmentation(image_size=224, advanced=True)
    augmented = augmentation(image=dummy_image)
    print(f"\nAugmented image shape: {augmented['image'].shape}")
    print(f"Augmented image type: {type(augmented['image'])}")

    # Test validation transform
    val_transform = get_validation_transform(image_size=224)
    val_transformed = val_transform(image=dummy_image)
    print(f"\nValidation image shape: {val_transformed['image'].shape}")

    print("\n" + "=" * 60)
    print("All preprocessing tests passed successfully.")
    print("=" * 60)
