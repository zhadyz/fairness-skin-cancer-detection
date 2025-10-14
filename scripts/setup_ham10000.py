"""
Quick setup script for HAM10000 dataset.

This script guides users through the complete HAM10000 setup process:
1. Check if dataset is downloaded
2. Generate FST annotations
3. Create train/val/test splits
4. Verify dataset integrity

Framework: MENDICANT_BIAS - Phase 1.5
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13
"""

import os
import sys
from pathlib import Path
import argparse
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_step(step_num, total_steps, title):
    """Print step header."""
    print(f"\n[Step {step_num}/{total_steps}] {title}")
    print("-" * 80)


def check_dataset_exists(data_dir):
    """Check if HAM10000 dataset is downloaded."""
    data_path = Path(data_dir)

    if not data_path.exists():
        return False, "Data directory does not exist"

    # Check for image directories
    part1 = data_path / "HAM10000_images_part_1"
    part2 = data_path / "HAM10000_images_part_2"
    metadata = data_path / "HAM10000_metadata.csv"

    missing = []
    if not part1.exists():
        missing.append("HAM10000_images_part_1/")
    if not part2.exists():
        missing.append("HAM10000_images_part_2/")
    if not metadata.exists():
        missing.append("HAM10000_metadata.csv")

    if missing:
        return False, f"Missing files: {', '.join(missing)}"

    # Count images
    num_images_part1 = len(list(part1.glob("*.jpg"))) if part1.exists() else 0
    num_images_part2 = len(list(part2.glob("*.jpg"))) if part2.exists() else 0
    total_images = num_images_part1 + num_images_part2

    if total_images < 10000:
        return False, f"Only {total_images} images found (expected ~10,000)"

    return True, f"Found {total_images} images"


def run_command(cmd, description):
    """Run a command and display output."""
    print(f"\nRunning: {description}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n{description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {description} failed!")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\nERROR: Failed to run {description}")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quick setup script for HAM10000 dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/ham10000",
        help="HAM10000 root directory (default: data/raw/ham10000)"
    )
    parser.add_argument(
        "--skip-fst",
        action="store_true",
        help="Skip FST annotation generation"
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Skip split creation"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip dataset verification"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing files"
    )

    args = parser.parse_args()

    print_header("HAM10000 Dataset Setup")
    print("This script will guide you through setting up HAM10000 for training.")

    # Step 1: Check if dataset exists
    print_step(1, 4, "Checking HAM10000 Dataset")

    exists, message = check_dataset_exists(args.data_dir)
    print(f"Status: {message}")

    if not exists:
        print("\nERROR: HAM10000 dataset not found!")
        print("\nTo download HAM10000:")
        print("  1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("  2. Download HAM10000_images_part_1.zip (~2.5 GB)")
        print("  3. Download HAM10000_images_part_2.zip (~2.5 GB)")
        print("  4. Download HAM10000_metadata")
        print(f"  5. Extract to: {Path(args.data_dir).absolute()}")
        print("\nExpected structure:")
        print("  data/raw/ham10000/")
        print("    ├── HAM10000_images_part_1/")
        print("    ├── HAM10000_images_part_2/")
        print("    └── HAM10000_metadata.csv")
        print("\nRun this script again after downloading the dataset.")
        sys.exit(1)

    print("Dataset found!")

    # Step 2: Generate FST annotations
    fst_output = Path("data/metadata/ham10000_fst_estimated.csv")

    if not args.skip_fst:
        print_step(2, 4, "Generating FST Annotations")

        if fst_output.exists() and not args.force:
            print(f"FST annotations already exist at: {fst_output}")
            print("Skipping (use --force to regenerate)")
        else:
            cmd = [
                sys.executable,
                "scripts/generate_ham10000_fst.py",
                "--data-dir", args.data_dir,
                "--output", str(fst_output),
            ]
            success = run_command(cmd, "FST annotation generation")
            if not success:
                print("\nWARNING: FST annotation failed. Continuing anyway...")
    else:
        print_step(2, 4, "Skipping FST Annotation Generation")

    # Step 3: Create splits
    splits_output = Path("data/metadata/ham10000_splits.json")

    if not args.skip_splits:
        print_step(3, 4, "Creating Train/Val/Test Splits")

        if splits_output.exists() and not args.force:
            print(f"Splits already exist at: {splits_output}")
            print("Skipping (use --force to regenerate)")
        else:
            # Use FST metadata if available, otherwise original metadata
            metadata_path = fst_output if fst_output.exists() else Path(args.data_dir) / "HAM10000_metadata.csv"

            cmd = [
                sys.executable,
                "scripts/create_ham10000_splits.py",
                "--metadata", str(metadata_path),
                "--output", str(splits_output),
                "--visualize",
            ]
            success = run_command(cmd, "Split creation")
            if not success:
                print("\nERROR: Split creation failed!")
                sys.exit(1)
    else:
        print_step(3, 4, "Skipping Split Creation")

    # Step 4: Verify dataset
    if not args.skip_verify:
        print_step(4, 4, "Verifying Dataset Integrity")

        metadata_path = fst_output if fst_output.exists() else Path(args.data_dir) / "HAM10000_metadata.csv"
        splits_arg = str(splits_output) if splits_output.exists() else ""

        cmd = [
            sys.executable,
            "scripts/verify_ham10000.py",
            "--data-dir", args.data_dir,
            "--metadata", str(metadata_path),
        ]

        if splits_arg:
            cmd.extend(["--splits", splits_arg])

        success = run_command(cmd, "Dataset verification")
        if not success:
            print("\nWARNING: Verification found issues. Review output above.")
    else:
        print_step(4, 4, "Skipping Dataset Verification")

    # Final summary
    print_header("Setup Complete!")

    print("HAM10000 dataset is ready for training.\n")
    print("Generated files:")
    if fst_output.exists():
        print(f"  - FST annotations: {fst_output}")
    if splits_output.exists():
        print(f"  - Train/val/test splits: {splits_output}")

    print("\nNext steps:")
    print("  1. Review documentation: docs/ham10000_integration.md")
    print("  2. Run baseline training:")
    print("     python experiments/baseline/train_resnet50.py --config configs/baseline_config.yaml")
    print("  3. Monitor training logs in experiments/baseline/logs/")
    print("  4. Evaluate fairness metrics on test set")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
