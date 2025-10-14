"""
Fitzpatrick Skin Type (FST) and Monk Skin Tone (MST) annotation utilities.

Provides:
- Automated ITA (Individual Typology Angle) calculation
- ITA to FST/MST mapping
- Batch annotation of image datasets
- Inter-rater reliability metrics (Cohen's Kappa)
- Visualization tools for FST distribution

Framework: MENDICANT_BIAS - the_didact research division
Version: 1.0
Date: 2025-10-13
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import cv2
from skimage import color
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# ITA to FST mapping (from literature: Chardon et al. 1991)
ITA_FST_THRESHOLDS = {
    'I': (55, np.inf),      # Very light
    'II': (41, 55),         # Light
    'III': (28, 41),        # Intermediate
    'IV': (19, 28),         # Tan/Olive
    'V': (-30, 19),         # Brown
    'VI': (-np.inf, -30),   # Dark brown/Black
}

# ITA to MST (Monk Skin Tone, 10-point scale) mapping
ITA_MST_THRESHOLDS = [
    (55, np.inf, 1),     # MST 1: Very light
    (48, 55, 2),         # MST 2: Light
    (41, 48, 3),         # MST 3: Light-medium
    (34, 41, 4),         # MST 4: Medium
    (28, 34, 5),         # MST 5: Medium
    (22, 28, 6),         # MST 6: Medium-tan
    (10, 22, 7),         # MST 7: Tan
    (-5, 10, 8),         # MST 8: Brown
    (-20, -5, 9),        # MST 9: Dark brown
    (-np.inf, -20, 10),  # MST 10: Very dark
]


def calculate_ita(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    exclude_lesion: bool = True,
) -> float:
    """
    Calculate Individual Typology Angle (ITA) for skin tone classification.

    ITA = arctan((L* - 50) / b*) × (180 / π)

    where L* (lightness) and b* (blue-yellow axis) are from CIELAB color space.

    Args:
        image: RGB image (H, W, C), numpy array
        roi: Region of Interest (x, y, width, height) for skin sampling.
             If None, use entire image.
        exclude_lesion: If True, attempt to exclude lesion area (use perilesional skin)

    Returns:
        ITA value in degrees
    """
    # Convert to RGB if needed
    if image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Extract ROI
    if roi:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]

    # Exclude lesion if requested (simple approach: remove darkest 20% pixels)
    if exclude_lesion:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = np.percentile(gray, 20)
        mask = gray > threshold
        image = image[mask]

    # Convert to CIELAB color space
    # Note: skimage expects float [0, 1], opencv uses uint8 [0, 255]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    lab_image = color.rgb2lab(image)

    # Calculate mean L* and b*
    L_mean = np.mean(lab_image[..., 0])
    b_mean = np.mean(lab_image[..., 2])

    # Calculate ITA
    ita_value = np.degrees(np.arctan((L_mean - 50) / b_mean))

    return ita_value


def ita_to_fst(ita: float) -> int:
    """
    Map ITA value to Fitzpatrick Skin Type (1-6).

    Args:
        ita: ITA value in degrees

    Returns:
        FST (1-6 for FST I-VI)
    """
    if ita > 55:
        return 1
    elif 41 <= ita <= 55:
        return 2
    elif 28 <= ita < 41:
        return 3
    elif 19 <= ita < 28:
        return 4
    elif -30 <= ita < 19:
        return 5
    else:  # ita < -30
        return 6


def ita_to_mst(ita: float) -> int:
    """
    Map ITA value to Monk Skin Tone (1-10).

    Args:
        ita: ITA value in degrees

    Returns:
        MST (1-10)
    """
    for lower, upper, mst in ITA_MST_THRESHOLDS:
        if lower <= ita < upper:
            return mst
    return 10  # Default to darkest tone if out of range


def annotate_image(
    image_path: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    exclude_lesion: bool = True,
    return_components: bool = False,
) -> Dict:
    """
    Annotate a single image with ITA, FST, and MST.

    Args:
        image_path: Path to image file
        roi: Region of Interest for skin sampling
        exclude_lesion: Exclude lesion area from ITA calculation
        return_components: If True, also return L* and b* values

    Returns:
        Dictionary with annotation results
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate ITA
    ita = calculate_ita(image, roi=roi, exclude_lesion=exclude_lesion)

    # Map to FST and MST
    fst = ita_to_fst(ita)
    mst = ita_to_mst(ita)

    result = {
        'image_path': image_path,
        'ita': ita,
        'fst': fst,
        'mst': mst,
    }

    if return_components:
        # Calculate L* and b* for debugging
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        lab_image = color.rgb2lab(image)
        result['L_mean'] = np.mean(lab_image[..., 0])
        result['b_mean'] = np.mean(lab_image[..., 2])

    return result


def batch_annotate_dataset(
    image_dir: str,
    output_csv: str,
    image_ext: str = '.jpg',
    roi: Optional[Tuple[int, int, int, int]] = None,
    exclude_lesion: bool = True,
) -> pd.DataFrame:
    """
    Batch annotate all images in a directory with ITA, FST, and MST.

    Args:
        image_dir: Directory containing images
        output_csv: Path to save annotation results
        image_ext: Image file extension
        roi: Region of Interest for all images (if uniform)
        exclude_lesion: Exclude lesion area from ITA calculation

    Returns:
        DataFrame with annotation results
    """
    image_path = Path(image_dir)
    image_files = list(image_path.glob(f"*{image_ext}"))

    print(f"Annotating {len(image_files)} images from {image_dir}...")

    results = []
    for i, image_file in enumerate(image_files):
        try:
            annotation = annotate_image(
                str(image_file),
                roi=roi,
                exclude_lesion=exclude_lesion,
                return_components=True,
            )
            results.append(annotation)

            if (i + 1) % 100 == 0:
                print(f"  Annotated {i + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"  Error annotating {image_file.name}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add image_id column (filename without extension)
    df['image_id'] = df['image_path'].apply(lambda x: Path(x).stem)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nAnnotation complete. Saved to {output_csv}")

    # Print distribution
    print("\nFST Distribution:")
    print(df['fst'].value_counts().sort_index())
    print("\nMST Distribution:")
    print(df['mst'].value_counts().sort_index())

    return df


def calculate_inter_rater_agreement(
    annotator1_labels: List[int],
    annotator2_labels: List[int],
) -> Dict:
    """
    Calculate inter-rater reliability metrics (Cohen's Kappa).

    Args:
        annotator1_labels: FST labels from annotator 1
        annotator2_labels: FST labels from annotator 2

    Returns:
        Dictionary with agreement metrics
    """
    # Cohen's Kappa
    kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)

    # Percent agreement
    agreement = np.mean(np.array(annotator1_labels) == np.array(annotator2_labels))

    # Within-1-category agreement (e.g., FST III vs IV acceptable)
    within_1 = np.mean(np.abs(np.array(annotator1_labels) - np.array(annotator2_labels)) <= 1)

    # Confusion matrix
    cm = confusion_matrix(annotator1_labels, annotator2_labels)

    return {
        'kappa': kappa,
        'percent_agreement': agreement,
        'within_1_agreement': within_1,
        'confusion_matrix': cm,
    }


def visualize_fst_distribution(
    fst_labels: List[int],
    title: str = "Fitzpatrick Skin Type Distribution",
    save_path: Optional[str] = None,
):
    """
    Visualize FST distribution as bar chart.

    Args:
        fst_labels: List of FST labels (1-6)
        title: Plot title
        save_path: Path to save figure (if None, display interactively)
    """
    fst_counts = pd.Series(fst_labels).value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    fst_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Fitzpatrick Skin Type')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(range(6), ['I', 'II', 'III', 'IV', 'V', 'VI'], rotation=0)
    plt.grid(axis='y', alpha=0.3)

    # Add percentage labels
    total = len(fst_labels)
    for i, (fst, count) in enumerate(fst_counts.items()):
        plt.text(i, count + total * 0.01, f'{count / total * 100:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_ita_distribution(
    ita_values: List[float],
    fst_labels: Optional[List[int]] = None,
    title: str = "ITA Distribution by FST",
    save_path: Optional[str] = None,
):
    """
    Visualize ITA value distribution, optionally colored by FST.

    Args:
        ita_values: List of ITA values
        fst_labels: Optional FST labels for coloring
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))

    if fst_labels:
        df = pd.DataFrame({'ITA': ita_values, 'FST': fst_labels})
        sns.violinplot(data=df, x='FST', y='ITA', palette='Set2')
        plt.xlabel('Fitzpatrick Skin Type')
        plt.ylabel('ITA (degrees)')
        plt.xticks(range(6), ['I', 'II', 'III', 'IV', 'V', 'VI'])

        # Add FST threshold lines
        thresholds = [55, 41, 28, 19, -30]
        for thresh in thresholds:
            plt.axhline(thresh, color='red', linestyle='--', alpha=0.5, linewidth=1)

    else:
        plt.hist(ita_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('ITA (degrees)')
        plt.ylabel('Frequency')

        # Add FST threshold lines
        thresholds = [55, 41, 28, 19, -30]
        for thresh in thresholds:
            plt.axvline(thresh, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def compare_ita_vs_expert(
    ita_predicted_fst: List[int],
    expert_fst: List[int],
) -> Dict:
    """
    Compare ITA-predicted FST vs expert annotations.

    Useful for validating ITA algorithm accuracy.

    Args:
        ita_predicted_fst: FST predicted by ITA algorithm
        expert_fst: FST annotated by expert dermatologists

    Returns:
        Dictionary with comparison metrics
    """
    agreement_metrics = calculate_inter_rater_agreement(ita_predicted_fst, expert_fst)

    print("=" * 60)
    print("ITA vs Expert Annotation Comparison")
    print("=" * 60)
    print(f"Cohen's Kappa: {agreement_metrics['kappa']:.3f}")
    print(f"Exact Agreement: {agreement_metrics['percent_agreement'] * 100:.2f}%")
    print(f"Within-1-Category Agreement: {agreement_metrics['within_1_agreement'] * 100:.2f}%")
    print("=" * 60)

    # Visualize confusion matrix
    cm = agreement_metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['I', 'II', 'III', 'IV', 'V', 'VI'],
                yticklabels=['I', 'II', 'III', 'IV', 'V', 'VI'])
    plt.xlabel('Expert FST')
    plt.ylabel('ITA-Predicted FST')
    plt.title('Confusion Matrix: ITA vs Expert')
    plt.tight_layout()
    plt.show()

    return agreement_metrics


if __name__ == "__main__":
    # Demo: Test ITA calculation
    print("FST Annotation Module")
    print("=" * 60)

    # Create synthetic test images with different "skin tones"
    test_tones = [
        ("Very Light (FST I)", np.array([240, 220, 210])),  # Very light pink
        ("Light (FST II)", np.array([220, 190, 170])),      # Light beige
        ("Intermediate (FST III)", np.array([200, 160, 130])),  # Medium tan
        ("Tan (FST IV)", np.array([170, 130, 100])),        # Tan
        ("Brown (FST V)", np.array([120, 90, 70])),         # Brown
        ("Dark (FST VI)", np.array([70, 50, 40])),          # Dark brown
    ]

    print("\nTesting ITA calculation on synthetic skin tones:")
    print("-" * 60)

    for tone_name, rgb_value in test_tones:
        # Create uniform image with this tone
        test_image = np.full((256, 256, 3), rgb_value, dtype=np.uint8)

        # Calculate ITA
        ita = calculate_ita(test_image, exclude_lesion=False)
        fst = ita_to_fst(ita)
        mst = ita_to_mst(ita)

        print(f"{tone_name:25s} | ITA: {ita:6.2f}° | FST: {fst} | MST: {mst:2d}")

    print("-" * 60)

    # Test inter-rater agreement
    print("\nTesting inter-rater agreement metrics:")
    print("-" * 60)

    # Simulate two annotators with slight disagreement
    annotator1 = [1, 2, 3, 3, 4, 5, 5, 6, 2, 3, 4, 4, 5]
    annotator2 = [1, 2, 3, 4, 4, 5, 6, 6, 2, 3, 4, 5, 5]  # Some 1-category disagreements

    agreement = calculate_inter_rater_agreement(annotator1, annotator2)
    print(f"Cohen's Kappa: {agreement['kappa']:.3f}")
    print(f"Exact Agreement: {agreement['percent_agreement'] * 100:.2f}%")
    print(f"Within-1 Agreement: {agreement['within_1_agreement'] * 100:.2f}%")

    print("\n" + "=" * 60)
    print("FST annotation module tests complete.")
    print("=" * 60)
