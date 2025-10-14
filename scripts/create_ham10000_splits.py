"""
Create stratified train/val/test splits for HAM10000 dataset.

This script:
1. Loads HAM10000 metadata (with FST annotations if available)
2. Creates stratified splits by diagnosis AND FST
3. Ensures no data leakage (same lesion_id in single split only)
4. Saves split indices to JSON
5. Generates split statistics and visualizations

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.ham10000_dataset import create_fst_stratified_splits, HAM10000_DIAGNOSIS_LABELS


def visualize_split_distribution(
    metadata: pd.DataFrame,
    splits: dict,
    output_dir: str,
):
    """
    Visualize class and FST distribution across splits.

    Args:
        metadata: Full metadata DataFrame
        splits: Dictionary with train/val/test indices
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare split DataFrames
    train_df = metadata.iloc[splits['train']]
    val_df = metadata.iloc[splits['val']]
    test_df = metadata.iloc[splits['test']]

    # 1. Diagnosis distribution across splits
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (split_name, split_df) in enumerate([
        ('Train', train_df),
        ('Val', val_df),
        ('Test', test_df)
    ]):
        ax = axes[idx]
        diagnosis_counts = split_df['dx'].value_counts().sort_index()

        diagnosis_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'{split_name} Split - Diagnosis Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)

        # Add percentages
        for i, (dx, count) in enumerate(diagnosis_counts.items()):
            ax.text(i, count + len(split_df) * 0.01, f'{count/len(split_df)*100:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / "diagnosis_distribution_by_split.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'diagnosis_distribution_by_split.png'}")

    # 2. FST distribution across splits (if available)
    if 'fst' in metadata.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (split_name, split_df) in enumerate([
            ('Train', train_df),
            ('Val', val_df),
            ('Test', test_df)
        ]):
            ax = axes[idx]
            fst_df = split_df[split_df['fst'] != -1]

            if len(fst_df) > 0:
                fst_counts = fst_df['fst'].value_counts().sort_index()
                fst_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
                ax.set_title(f'{split_name} Split - FST Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Fitzpatrick Skin Type')
                ax.set_ylabel('Count')
                ax.set_xticklabels([f'FST {i}' for i in fst_counts.index], rotation=0)
                ax.grid(axis='y', alpha=0.3)

                # Add percentages
                for i, (fst, count) in enumerate(fst_counts.items()):
                    ax.text(i, count + len(fst_df) * 0.01, f'{count/len(fst_df)*100:.1f}%',
                           ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No FST data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{split_name} Split - FST Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / "fst_distribution_by_split.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'fst_distribution_by_split.png'}")

    # 3. Stacked bar chart: diagnosis distribution comparison
    diagnosis_data = []
    for split_name in ['train', 'val', 'test']:
        split_df = metadata.iloc[splits[split_name]]
        for dx in metadata['dx'].unique():
            count = (split_df['dx'] == dx).sum()
            diagnosis_data.append({
                'split': split_name,
                'diagnosis': dx,
                'count': count,
                'percentage': count / len(split_df) * 100
            })

    diagnosis_comparison_df = pd.DataFrame(diagnosis_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df = diagnosis_comparison_df.pivot(index='diagnosis', columns='split', values='percentage')
    pivot_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
    ax.set_title('Diagnosis Distribution Across Splits (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Percentage')
    ax.legend(title='Split', labels=['Train', 'Val', 'Test'])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / "diagnosis_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'diagnosis_comparison.png'}")

    # 4. Lesion-level statistics (if lesion_id available)
    if 'lesion_id' in metadata.columns:
        lesion_stats = []
        for split_name in ['train', 'val', 'test']:
            split_df = metadata.iloc[splits[split_name]]
            unique_lesions = split_df['lesion_id'].nunique()
            total_images = len(split_df)
            lesion_stats.append({
                'Split': split_name.capitalize(),
                'Unique Lesions': unique_lesions,
                'Total Images': total_images,
                'Images per Lesion': total_images / unique_lesions if unique_lesions > 0 else 0
            })

        lesion_df = pd.DataFrame(lesion_stats)
        print("\nLesion-level Statistics:")
        print(lesion_df.to_string(index=False))

        # Save to CSV
        lesion_df.to_csv(output_path / "lesion_statistics.csv", index=False)


def print_split_summary(metadata: pd.DataFrame, splits: dict):
    """
    Print detailed summary of splits.

    Args:
        metadata: Full metadata DataFrame
        splits: Dictionary with train/val/test indices
    """
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    total_samples = len(metadata)

    for split_name in ['train', 'val', 'test']:
        split_indices = splits[split_name]
        split_df = metadata.iloc[split_indices]

        print(f"\n{split_name.upper()} Split:")
        print(f"  Total samples: {len(split_df)} ({len(split_df)/total_samples*100:.2f}%)")

        # Diagnosis distribution
        print(f"  Diagnosis distribution:")
        for dx, count in split_df['dx'].value_counts().sort_index().items():
            print(f"    {dx:6s}: {count:5d} ({count/len(split_df)*100:5.2f}%)")

        # FST distribution (if available)
        if 'fst' in split_df.columns:
            fst_df = split_df[split_df['fst'] != -1]
            if len(fst_df) > 0:
                print(f"  FST distribution:")
                for fst, count in fst_df['fst'].value_counts().sort_index().items():
                    print(f"    FST {fst}: {count:5d} ({count/len(fst_df)*100:5.2f}%)")
                unknown = len(split_df) - len(fst_df)
                if unknown > 0:
                    print(f"    Unknown: {unknown:5d} ({unknown/len(split_df)*100:5.2f}%)")

        # Lesion statistics (if available)
        if 'lesion_id' in split_df.columns:
            unique_lesions = split_df['lesion_id'].nunique()
            print(f"  Unique lesions: {unique_lesions}")
            print(f"  Images per lesion: {len(split_df)/unique_lesions:.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits for HAM10000 dataset"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/ham10000_fst_estimated.csv",
        help="Path to HAM10000 metadata CSV (with FST if available)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/metadata/ham10000_splits.json",
        help="Output path for split indices JSON"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-stratify-fst",
        action="store_true",
        help="Do NOT stratify by FST (only by diagnosis)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate split distribution visualizations"
    )

    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("ERROR: Split ratios must sum to 1.0")
        sys.exit(1)

    # Check metadata file
    if not Path(args.metadata).exists():
        print(f"ERROR: Metadata file not found: {args.metadata}")
        print("\nIf you haven't generated FST annotations yet, run:")
        print("  python scripts/generate_ham10000_fst.py")
        print("\nOr use the original metadata:")
        print("  python scripts/create_ham10000_splits.py --metadata data/raw/ham10000/HAM10000_metadata.csv")
        sys.exit(1)

    # Create splits
    print("=" * 70)
    print("HAM10000 Split Creation")
    print("=" * 70)
    print(f"Metadata: {args.metadata}")
    print(f"Output: {args.output}")
    print(f"Train:Val:Test = {args.train_ratio}:{args.val_ratio}:{args.test_ratio}")
    print(f"Random seed: {args.random_seed}")
    print(f"Stratify by FST: {not args.no_stratify_fst}")
    print("=" * 70)

    splits = create_fst_stratified_splits(
        metadata_path=args.metadata,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
        stratify_by_fst=not args.no_stratify_fst,
    )

    # Load full metadata for analysis
    metadata = pd.read_csv(args.metadata)

    # Print summary
    print_split_summary(metadata, splits)

    # Generate visualizations
    if args.visualize:
        output_dir = Path(args.output).parent / "split_visualizations"
        print(f"\nGenerating visualizations...")
        visualize_split_distribution(metadata, splits, str(output_dir))
        print(f"Visualizations saved to: {output_dir}")

    print("\nSplit creation complete!")


if __name__ == "__main__":
    main()
