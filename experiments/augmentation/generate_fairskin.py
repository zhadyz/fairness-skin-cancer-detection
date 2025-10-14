"""
Synthetic Image Generation Script for FairSkin.

Generates FST-balanced synthetic dermoscopy images using trained
LoRA-adapted Stable Diffusion model with quality filtering.

Usage:
    python experiments/augmentation/generate_fairskin.py \\
        --config configs/fairskin_config.yaml \\
        --lora_weights checkpoints/fairskin_lora/lora_weights_final.pt \\
        --num_images 60000

Expected Generation Time (RTX 3090):
    - 60,000 images at 50 steps: ~50-100 GPU hours (1.0-1.5 img/min)
    - With quality filtering (1.5x overgeneration): ~75-150 GPU hours

Framework: MENDICANT_BIAS - Phase 2
Agent: HOLLOWED_EYES
Version: 0.3.0
Date: 2025-10-13
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
from typing import Optional, List
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

from src.augmentation.fairskin_diffusion import FairSkinDiffusionModel
from src.augmentation.quality_metrics import QualityFilter, compute_fid, compute_lpips
from src.data.ham10000_dataset import HAM10000Dataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic dermoscopy images with FairSkin")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/fairskin_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to trained LoRA weights"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Override total number of images to generate"
    )

    parser.add_argument(
        "--target_fsts",
        type=int,
        nargs="+",
        default=None,
        help="Target FST classes (e.g., 5 6)"
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override number of diffusion steps"
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Override guidance scale"
    )

    parser.add_argument(
        "--skip_quality_filter",
        action="store_true",
        help="Skip quality filtering (faster but lower quality)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda or cpu)"
    )

    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test mode (100 images only)"
    )

    return parser.parse_args()


def load_reference_images(config: dict, max_images: int = 500) -> List[Image.Image]:
    """
    Load reference real images for quality filtering.

    Args:
        config: Configuration dictionary
        max_images: Maximum number of reference images

    Returns:
        List of PIL Images
    """
    data_cfg = config.get('data', {})
    data_dir = data_cfg.get('real_data_dir', 'data/raw/ham10000')

    print(f"\nLoading reference images from: {data_dir}")

    try:
        dataset = HAM10000Dataset(
            root_dir=data_dir,
            split="train",
            use_fst_annotations=False,  # Don't need FST for reference
        )

        # Sample random images
        num_samples = min(max_images, len(dataset))
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        reference_images = []
        for idx in tqdm(indices, desc="Loading reference images"):
            sample = dataset[idx]
            # Convert tensor to PIL Image
            image_tensor = sample['image']
            if image_tensor.dim() == 3:
                image_array = image_tensor.permute(1, 2, 0).numpy()
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                reference_images.append(Image.fromarray(image_array))

        print(f"  Loaded {len(reference_images)} reference images")
        return reference_images

    except Exception as e:
        print(f"  Warning: Failed to load reference images: {e}")
        return []


def setup_quality_filter(config: dict, reference_images: List[Image.Image]) -> Optional[QualityFilter]:
    """
    Setup quality filter for synthetic images.

    Args:
        config: Configuration dictionary
        reference_images: Reference real images

    Returns:
        QualityFilter instance or None if disabled
    """
    gen_cfg = config.get('generation', {})
    quality_cfg = gen_cfg.get('quality_thresholds', {})

    if not gen_cfg.get('apply_quality_filter', True):
        return None

    print("\nSetting up quality filter...")

    quality_filter = QualityFilter(
        real_reference_images=reference_images,
        fid_threshold=quality_cfg.get('fid_max', 30.0),
        lpips_threshold=quality_cfg.get('lpips_max', 0.2),
        confidence_threshold=quality_cfg.get('confidence_min', 0.6),
        classifier=None,  # TODO: Load trained classifier if available
        device=config.get('stable_diffusion', {}).get('device', 'cuda'),
    )

    return quality_filter


def generate_balanced_dataset(
    model: FairSkinDiffusionModel,
    config: dict,
    args: argparse.Namespace,
    quality_filter: Optional[QualityFilter] = None,
):
    """
    Generate FST-balanced synthetic dataset.

    Args:
        model: FairSkin diffusion model
        config: Configuration dictionary
        args: Command-line arguments
        quality_filter: Optional quality filter
    """
    gen_cfg = config.get('generation', {})

    # Get generation parameters
    target_fsts = args.target_fsts or gen_cfg.get('target_fsts', [5, 6])
    diagnoses = gen_cfg.get('diagnoses', [0, 1, 2, 3, 4, 5, 6])
    num_images_per_fst = args.num_images or gen_cfg.get('num_images_per_fst', 10000)
    output_dir = args.output_dir or gen_cfg.get('output_dir', 'data/synthetic/fairskin')

    # Inference parameters
    num_inference_steps = args.num_inference_steps or gen_cfg.get('num_inference_steps', 50)
    guidance_scale = args.guidance_scale or gen_cfg.get('guidance_scale', 7.5)
    prompt_style = gen_cfg.get('prompt_style', 'clinical')

    # Quality filtering
    overgeneration_factor = gen_cfg.get('overgeneration_factor', 1.5) if quality_filter else 1.0

    print("\n" + "=" * 70)
    print("Synthetic Image Generation Configuration")
    print("=" * 70)
    print(f"  Target FSTs: {target_fsts}")
    print(f"  Diagnoses: {len(diagnoses)} classes")
    print(f"  Images per FST: {num_images_per_fst}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Prompt style: {prompt_style}")
    print(f"  Quality filter: {'Enabled' if quality_filter else 'Disabled'}")
    if quality_filter:
        print(f"  Overgeneration factor: {overgeneration_factor}x")
    print(f"  Output directory: {output_dir}")
    print("=" * 70 + "\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate generation plan
    images_per_class = num_images_per_fst // len(diagnoses)
    total_target = num_images_per_fst * len(target_fsts)
    total_to_generate = int(total_target * overgeneration_factor)

    print(f"Generation Plan:")
    print(f"  Total target images: {total_target:,}")
    print(f"  Total to generate (before filtering): {total_to_generate:,}")
    print(f"  Images per (FST, diagnosis): {images_per_class:,}")
    print()

    # Generate images
    generated_count = 0
    filtered_count = 0

    for fst in target_fsts:
        print(f"\nGenerating images for FST {fst}...")

        for dx in diagnoses:
            dx_name = model.create_prompt(dx, fst).split(' on ')[0].split(' of ')[-1]
            dx_name = dx_name.replace(' ', '_')

            num_to_generate = int(images_per_class * overgeneration_factor)
            print(f"  Diagnosis {dx} ({dx_name}): generating {num_to_generate} images...")

            batch_size = gen_cfg.get('batch_size', 4)
            num_batches = (num_to_generate + batch_size - 1) // batch_size

            kept_for_this_class = 0

            for batch_idx in tqdm(range(num_batches), desc=f"    FST{fst}-{dx_name}"):
                current_batch_size = min(batch_size, num_to_generate - batch_idx * batch_size)

                # Generate batch
                batch_diagnoses = [dx] * current_batch_size
                batch_fsts = [fst] * current_batch_size

                try:
                    images = model.generate_batch(
                        diagnoses=batch_diagnoses,
                        fsts=batch_fsts,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        batch_size=current_batch_size,
                        prompt_style=prompt_style,
                    )

                    # Apply quality filter
                    if quality_filter:
                        for img in images:
                            if quality_filter.passes_filter(img, dx, fst):
                                # Save image
                                filename = f"synthetic_fst{fst}_{dx_name}_{kept_for_this_class:05d}.png"
                                img.save(output_path / filename)
                                kept_for_this_class += 1
                                filtered_count += 1

                                # Stop if we have enough for this class
                                if kept_for_this_class >= images_per_class:
                                    break
                    else:
                        # Save all images
                        for img_idx, img in enumerate(images):
                            filename = f"synthetic_fst{fst}_{dx_name}_{kept_for_this_class:05d}.png"
                            img.save(output_path / filename)
                            kept_for_this_class += 1
                            filtered_count += 1

                    generated_count += len(images)

                    # Early stop if we have enough
                    if kept_for_this_class >= images_per_class:
                        break

                except Exception as e:
                    print(f"\n      Warning: Batch generation failed: {e}")
                    continue

            print(f"      Generated: {generated_count} | Kept: {kept_for_this_class}/{images_per_class}")

    # Final statistics
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"  Total images generated: {generated_count:,}")
    print(f"  Total images saved: {filtered_count:,}")
    if quality_filter:
        print(f"  Acceptance rate: {filtered_count/max(generated_count, 1)*100:.1f}%")
    print(f"  Saved to: {output_path}")
    print("=" * 70 + "\n")


def main():
    """Main generation script."""
    # Parse arguments
    args = parse_args()

    print("=" * 70)
    print("FairSkin Synthetic Image Generation")
    print("=" * 70)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Quick test mode
    if args.quick_test:
        print("\n" + "=" * 70)
        print("QUICK TEST MODE - Generating 100 images only")
        print("=" * 70 + "\n")
        if not args.num_images:
            config['generation']['num_images_per_fst'] = 50  # 50 per FST = 100 total for 2 FSTs

    # Setup device
    device = args.device or config.get('stable_diffusion', {}).get('device', 'cuda')
    dtype_str = config.get('stable_diffusion', {}).get('dtype', 'float16')
    dtype = torch.float16 if dtype_str == 'float16' else torch.float32

    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")

    # Initialize model
    print(f"\nInitializing FairSkin diffusion model...")
    print(f"  LoRA weights: {args.lora_weights}")

    model = FairSkinDiffusionModel(
        model_id=config.get('stable_diffusion', {}).get('model_id', 'runwayml/stable-diffusion-v1-5'),
        device=device,
        dtype=dtype,
        lora_weights_path=args.lora_weights,
        safety_checker=False,
        scheduler_type=config.get('stable_diffusion', {}).get('scheduler_type', 'pndm'),
    )

    # Load reference images for quality filtering
    reference_images = []
    if not args.skip_quality_filter and config.get('generation', {}).get('apply_quality_filter', True):
        reference_images = load_reference_images(config, max_images=500)

    # Setup quality filter
    quality_filter = None
    if not args.skip_quality_filter and reference_images:
        quality_filter = setup_quality_filter(config, reference_images)

    # Generate dataset
    generate_balanced_dataset(model, config, args, quality_filter)

    print("\nGeneration complete!")
    print("\nNext steps:")
    print("  1. Verify image quality: visually inspect samples")
    print("  2. Compute FID/LPIPS metrics: python experiments/augmentation/evaluate_quality.py")
    print("  3. Train classifier with synthetic data: python experiments/augmentation/train_with_fairskin.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
