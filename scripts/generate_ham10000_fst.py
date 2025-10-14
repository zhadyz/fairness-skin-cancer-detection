"""
Generate FST annotations for HAM10000 dataset using ITA calculation.

This script:
1. Loads HAM10000 images from raw directory
2. Calculates ITA (Individual Typology Angle) for each image
3. Maps ITA to Fitzpatrick Skin Type (1-6)
4. Saves FST annotations to CSV
5. Generates distribution visualizations

Framework: MENDICANT_BIAS - Phase 1.5
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import os
import sys
from pathlib import Path
import argparse
import warnings
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.fst_annotation import (
    calculate_ita,
    ita_to_fst,
    visualize_fst_distribution,
    visualize_ita_distribution,
)


def generate_ham10000_fst_annotations(
    ham10000_dir: str,
    metadata_csv: str,
    output_csv: str,
    image_parts: list = ["HAM10000_images_part_1", "HAM10000_images_part_2"],
    exclude_lesion: bool = True,
    save_visualizations: bool = True,
):
    """
    Generate FST annotations for entire HAM10000 dataset.

    Args:
        ham10000_dir: Root directory containing HAM10000 images
        metadata_csv: Path to HAM10000_metadata.csv
        output_csv: Output path for FST annotations
        image_parts: List of image subdirectories
        exclude_lesion: Whether to exclude lesion area from ITA calculation
        save_visualizations: Whether to save distribution plots

    Returns:
        DataFrame with FST annotations
    """
    import cv2

    ham10000_path = Path(ham10000_dir)
    metadata_path = Path(metadata_csv)

    # Load metadata
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(metadata)} images")

    # Build image path mapping
    image_path_map = {}
    for part_dir in image_parts:
        part_path = ham10000_path / part_dir
        if part_path.exists():
            for img_file in part_path.glob("*.jpg"):
                image_id = img_file.stem
                image_path_map[image_id] = img_file
            print(f"Found {len(list(part_path.glob('*.jpg')))} images in {part_dir}")

    print(f"Total images found: {len(image_path_map)}")

    # Calculate FST for each image
    results = []
    failed_images = []

    print("\nCalculating ITA and FST for each image...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_id = row['image_id']

        if image_id not in image_path_map:
            warnings.warn(f"Image not found: {image_id}")
            failed_images.append(image_id)
            results.append({
                'image_id': image_id,
                'ita': np.nan,
                'fst': -1,
                'L_mean': np.nan,
                'b_mean': np.nan,
                'error': 'Image file not found',
            })
            continue

        try:
            # Load image
            image_path = image_path_map[image_id]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Calculate ITA
            ita = calculate_ita(image, exclude_lesion=exclude_lesion)

            # Map to FST
            fst = ita_to_fst(ita)

            # Store results with components for debugging
            from skimage import color

            # Calculate L* and b* for debugging
            if image.dtype == np.uint8:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image

            lab_image = color.rgb2lab(image_float)
            L_mean = np.mean(lab_image[..., 0])
            b_mean = np.mean(lab_image[..., 2])

            results.append({
                'image_id': image_id,
                'ita': ita,
                'fst': fst,
                'L_mean': L_mean,
                'b_mean': b_mean,
                'error': None,
            })

        except Exception as e:
            warnings.warn(f"Error processing {image_id}: {e}")
            failed_images.append(image_id)
            results.append({
                'image_id': image_id,
                'ita': np.nan,
                'fst': -1,
                'L_mean': np.nan,
                'b_mean': np.nan,
                'error': str(e),
            })

    # Create DataFrame
    fst_df = pd.DataFrame(results)

    # Merge with original metadata
    full_df = metadata.merge(fst_df, on='image_id', how='left')

    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_csv, index=False)
    print(f"\nSaved FST annotations to: {output_csv}")

    # Print statistics
    print("\n" + "=" * 70)
    print("FST ANNOTATION SUMMARY")
    print("=" * 70)

    successful = len(fst_df[fst_df['fst'] != -1])
    print(f"Successfully annotated: {successful}/{len(fst_df)} images ({successful/len(fst_df)*100:.2f}%)")
    print(f"Failed: {len(failed_images)} images")

    if successful > 0:
        print("\nFST Distribution:")
        fst_counts = fst_df[fst_df['fst'] != -1]['fst'].value_counts().sort_index()
        for fst, count in fst_counts.items():
            print(f"  FST {fst}: {count:5d} ({count/successful*100:5.2f}%)")

        print("\nITA Statistics:")
        ita_stats = fst_df[fst_df['fst'] != -1]['ita'].describe()
        print(f"  Mean: {ita_stats['mean']:.2f}°")
        print(f"  Std:  {ita_stats['std']:.2f}°")
        print(f"  Min:  {ita_stats['min']:.2f}°")
        print(f"  Max:  {ita_stats['max']:.2f}°")

        # FST by diagnosis
        print("\nFST Distribution by Diagnosis:")
        fst_by_dx = full_df[full_df['fst'] != -1].groupby(['dx', 'fst']).size().unstack(fill_value=0)
        print(fst_by_dx)

    # Generate visualizations
    if save_visualizations and successful > 0:
        output_dir = output_path.parent
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # FST distribution
        fst_labels = fst_df[fst_df['fst'] != -1]['fst'].tolist()
        visualize_fst_distribution(
            fst_labels,
            title="HAM10000 FST Distribution (ITA-based)",
            save_path=viz_dir / "ham10000_fst_distribution.png"
        )

        # ITA distribution by FST
        ita_values = fst_df[fst_df['fst'] != -1]['ita'].tolist()
        visualize_ita_distribution(
            ita_values,
            fst_labels=fst_labels,
            title="HAM10000 ITA Distribution by FST",
            save_path=viz_dir / "ham10000_ita_distribution.png"
        )

        # FST by diagnosis heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(fst_by_dx, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Count'})
        plt.xlabel('Fitzpatrick Skin Type')
        plt.ylabel('Diagnosis')
        plt.title('HAM10000: FST Distribution by Diagnosis')
        plt.tight_layout()
        plt.savefig(viz_dir / "ham10000_fst_by_diagnosis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nVisualizations saved to: {viz_dir}")

    print("=" * 70)

    return full_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate FST annotations for HAM10000 dataset using ITA calculation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ham10000",
        help="HAM10000 root directory (default: data/raw/ham10000)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to HAM10000_metadata.csv (default: {data-dir}/HAM10000_metadata.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/metadata/ham10000_fst_estimated.csv",
        help="Output path for FST annotations (default: data/metadata/ham10000_fst_estimated.csv)"
    )
    parser.add_argument(
        "--no-exclude-lesion",
        action="store_true",
        help="Do NOT exclude lesion area from ITA calculation"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Do NOT generate distribution visualizations"
    )

    args = parser.parse_args()

    # Set metadata path
    if args.metadata is None:
        args.metadata = Path(args.data_dir) / "HAM10000_metadata.csv"

    # Check if data exists
    if not Path(args.data_dir).exists():
        print("ERROR: HAM10000 data directory not found.")
        print(f"Expected location: {Path(args.data_dir).absolute()}")
        print("\nTo download HAM10000 dataset:")
        print("  1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("  2. Download HAM10000_images_part_1.zip")
        print("  3. Download HAM10000_images_part_2.zip")
        print("  4. Download HAM10000_metadata")
        print(f"  5. Extract to: {Path(args.data_dir).absolute()}")
        sys.exit(1)

    if not Path(args.metadata).exists():
        print(f"ERROR: Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Generate FST annotations
    print("=" * 70)
    print("HAM10000 FST Annotation Generation")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Metadata: {args.metadata}")
    print(f"Output: {args.output}")
    print(f"Exclude lesion: {not args.no_exclude_lesion}")
    print(f"Generate visualizations: {not args.no_visualizations}")
    print("=" * 70)

    generate_ham10000_fst_annotations(
        ham10000_dir=args.data_dir,
        metadata_csv=args.metadata,
        output_csv=args.output,
        exclude_lesion=not args.no_exclude_lesion,
        save_visualizations=not args.no_visualizations,
    )

    print("\nFST annotation generation complete!")


if __name__ == "__main__":
    main()
