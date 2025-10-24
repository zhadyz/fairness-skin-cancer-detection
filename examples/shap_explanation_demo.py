"""
SHAP Explainability Demonstration for Skin Cancer Detection

Demonstrates:
1. Loading a trained model (ResNet, Hybrid, etc.)
2. Generating SHAP explanations for individual predictions
3. Batch processing with FST comparison
4. Creating visualization outputs
5. Fairness analysis across Fitzpatrick Skin Types
6. Exporting results for clinical review

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-24

Usage:
    python examples/shap_explanation_demo.py --model_path checkpoints/hybrid_best.pth \
                                              --data_dir data/processed/ham10000 \
                                              --output_dir outputs/explanations \
                                              --num_samples 50
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explainability.shap_explainer import SHAPExplainer, ExplanationResult
from src.explainability.visualization import (
    visualize_explanation,
    visualize_fst_comparison,
    ExplanationVisualizer,
    create_saliency_overlay,
    create_shap_heatmap
)
from src.models.hybrid_model import create_hybrid_model, HybridModelConfig
from src.models.resnet_baseline import create_resnet_model
from src.data.datasets import BaseDermoscopyDataset
from src.data.preprocessing import get_inference_transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    model_type: str = "hybrid",
    num_classes: int = 7,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pth file)
        model_type: Model architecture type ("hybrid", "resnet50", "resnet18")
        num_classes: Number of classes
        device: Computation device

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading {model_type} model from {model_path}")

    # Create model architecture
    if model_type == "hybrid":
        model = create_hybrid_model(
            convnext_variant='base',
            swin_variant='small',
            num_classes=num_classes,
            enable_fairdisco=False
        )
    elif model_type.startswith("resnet"):
        backbone = model_type  # resnet18, resnet50, etc.
        model = create_resnet_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    return model


def load_dataset(
    data_dir: str,
    metadata_file: str = "metadata.csv",
    split: str = "test",
    max_samples: Optional[int] = None
) -> BaseDermoscopyDataset:
    """
    Load dataset for explanation generation.

    Args:
        data_dir: Directory containing images and metadata
        metadata_file: Metadata CSV filename
        split: Dataset split to use
        max_samples: Maximum number of samples to load

    Returns:
        Dataset instance
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    logger.info(f"Loading dataset from {data_dir}")

    # Load metadata
    df = pd.read_csv(metadata_path)

    # Filter by split if column exists
    if 'split' in df.columns:
        df = df[df['split'] == split]

    # Limit samples
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # Get transforms
    transforms = get_inference_transforms(image_size=224)

    # Create dataset
    dataset = BaseDermoscopyDataset(
        metadata_df=df,
        image_dir=data_dir / "images",
        image_col='image_id',
        label_col='diagnosis',
        fst_col='fitzpatrick_skin_type' if 'fitzpatrick_skin_type' in df.columns else None,
        transform=transforms
    )

    return dataset


def create_background_dataset(
    dataset: BaseDermoscopyDataset,
    n_samples: int = 50
) -> DataLoader:
    """
    Create background dataset for GradientSHAP.

    Selects diverse samples across classes for background.

    Args:
        dataset: Full dataset
        n_samples: Number of background samples

    Returns:
        DataLoader for background samples
    """
    logger.info(f"Creating background dataset with {n_samples} samples")

    # Get label distribution
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_labels = list(set(labels))

    # Sample uniformly from each class
    samples_per_class = max(1, n_samples // len(unique_labels))
    background_indices = []

    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        selected = np.random.choice(
            label_indices,
            size=min(samples_per_class, len(label_indices)),
            replace=False
        )
        background_indices.extend(selected)

    # Create subset
    background_subset = Subset(dataset, background_indices[:n_samples])
    background_loader = DataLoader(background_subset, batch_size=n_samples, shuffle=False)

    logger.info(f"Background dataset created with {len(background_indices[:n_samples])} samples")
    return background_loader


def generate_explanations(
    explainer: SHAPExplainer,
    dataset: BaseDermoscopyDataset,
    num_samples: int = 50,
    batch_size: int = 8
) -> List[ExplanationResult]:
    """
    Generate explanations for multiple samples.

    Args:
        explainer: Configured SHAPExplainer
        dataset: Dataset to explain
        num_samples: Number of samples to process
        batch_size: Batch size for processing

    Returns:
        List of ExplanationResult objects
    """
    logger.info(f"Generating explanations for {num_samples} samples")

    results = []
    num_samples = min(num_samples, len(dataset))

    # Process samples
    for i in tqdm(range(num_samples), desc="Generating explanations"):
        image, label, fst = dataset[i]

        # Generate explanation
        result = explainer.explain_prediction(
            image=image,
            target_class=None,  # Use predicted class
            fst_label=fst
        )

        results.append(result)

        # Log performance
        if (i + 1) % 10 == 0:
            avg_time = np.mean([r.computation_time for r in results])
            logger.info(f"  Processed {i+1}/{num_samples} samples, avg time: {avg_time:.3f}s")

    return results


def analyze_fairness(
    results: List[ExplanationResult],
    explainer: SHAPExplainer,
    output_dir: Path
) -> Dict[str, any]:
    """
    Perform fairness analysis across FST groups.

    Args:
        results: List of explanation results with FST labels
        explainer: SHAPExplainer instance
        output_dir: Output directory for visualizations

    Returns:
        Dictionary with fairness metrics
    """
    logger.info("Performing fairness analysis across FST groups")

    # Group results by FST
    results_by_fst = {}
    for result in results:
        if result.fst_label is not None:
            fst = int(result.fst_label)
            if fst not in results_by_fst:
                results_by_fst[fst] = []
            results_by_fst[fst].append(result)

    logger.info(f"FST groups found: {sorted(results_by_fst.keys())}")
    for fst, fst_results in sorted(results_by_fst.items()):
        logger.info(f"  FST {fst}: {len(fst_results)} samples")

    # Aggregate statistics
    fst_stats = explainer.aggregate_fst_statistics(results_by_fst)

    # Pairwise comparisons
    comparison_results = []
    fst_groups = sorted(results_by_fst.keys())

    # Compare FST I-III vs IV-VI (light vs dark)
    if len(fst_groups) >= 2:
        light_fsts = [f for f in fst_groups if f <= 3]
        dark_fsts = [f for f in fst_groups if f >= 4]

        if light_fsts and dark_fsts:
            # Aggregate light and dark groups
            light_results = []
            dark_results = []

            for fst in light_fsts:
                light_results.extend(results_by_fst[fst])
            for fst in dark_fsts:
                dark_results.extend(results_by_fst[fst])

            logger.info(f"Comparing light skin (FST I-III, n={len(light_results)}) vs dark skin (FST IV-VI, n={len(dark_results)})")

            comparison = explainer.compare_fst_explanations(
                results_fst1=light_results,
                results_fst2=dark_results,
                fst1=1,  # Representative for light
                fst2=6   # Representative for dark
            )

            comparison_results.append(comparison)

            # Visualize comparison
            fig = visualize_fst_comparison(
                comparison,
                save_path=output_dir / "fst_light_vs_dark_comparison.png"
            )
            plt.close(fig)

    # Compare all pairwise combinations
    for i, fst1 in enumerate(fst_groups):
        for fst2 in fst_groups[i+1:]:
            if len(results_by_fst[fst1]) >= 5 and len(results_by_fst[fst2]) >= 5:
                logger.info(f"Comparing FST {fst1} vs FST {fst2}")

                comparison = explainer.compare_fst_explanations(
                    results_fst1=results_by_fst[fst1],
                    results_fst2=results_by_fst[fst2],
                    fst1=fst1,
                    fst2=fst2
                )

                comparison_results.append(comparison)

                # Visualize
                fig = visualize_fst_comparison(
                    comparison,
                    save_path=output_dir / f"fst_{fst1}_vs_{fst2}_comparison.png"
                )
                plt.close(fig)

    return {
        'fst_statistics': fst_stats,
        'comparisons': comparison_results,
        'results_by_fst': results_by_fst
    }


def create_visualizations(
    results: List[ExplanationResult],
    results_by_fst: Dict[int, List[ExplanationResult]],
    comparison_results: List,
    output_dir: Path
):
    """
    Create comprehensive visualizations for explanations.

    Args:
        results: All explanation results
        results_by_fst: Results grouped by FST
        comparison_results: FST comparison results
        output_dir: Output directory
    """
    logger.info("Creating visualizations")

    visualizer = ExplanationVisualizer(output_dir)

    # 1. Individual sample visualizations (first 20 samples)
    logger.info("  Creating individual sample visualizations...")
    samples_dir = output_dir / "individual_samples"
    samples_dir.mkdir(exist_ok=True)

    for i, result in enumerate(results[:20]):
        fig = visualize_explanation(
            result,
            show_top_regions=True,
            save_path=samples_dir / f"sample_{i:03d}_class{result.predicted_class}_fst{result.fst_label}.png"
        )
        plt.close(fig)

    # 2. Sample grid
    logger.info("  Creating sample grid...")
    fig = visualizer.create_sample_grid(
        results=results,
        n_samples=min(9, len(results)),
        save_name="explanation_grid.png"
    )
    plt.close(fig)

    # 3. FST dashboard
    if results_by_fst:
        logger.info("  Creating FST dashboard...")
        fig = visualizer.create_fst_dashboard(
            results_by_fst=results_by_fst,
            save_name="fst_dashboard.png"
        )
        plt.close(fig)

    # 4. HTML report
    logger.info("  Generating HTML report...")
    visualizer.generate_html_report(
        results_by_fst=results_by_fst,
        comparison_results=comparison_results,
        report_name="explanation_report.html"
    )

    logger.info(f"All visualizations saved to {output_dir}")


def save_results(
    results: List[ExplanationResult],
    fairness_analysis: Dict,
    output_dir: Path
):
    """
    Save explanation results and analysis to disk.

    Args:
        results: List of explanation results
        fairness_analysis: Fairness analysis dictionary
        output_dir: Output directory
    """
    logger.info("Saving results to disk")

    # Save attribution maps
    attributions_dir = output_dir / "attributions"
    attributions_dir.mkdir(exist_ok=True)

    for i, result in enumerate(results):
        np.save(attributions_dir / f"attribution_{i:04d}.npy", result.shap_values)

    # Save summary CSV
    summary_data = []
    for i, result in enumerate(results):
        summary_data.append({
            'sample_id': i,
            'predicted_class': result.predicted_class,
            'confidence': result.confidence,
            'fst_label': result.fst_label,
            'attribution_magnitude': result.attribution_magnitude,
            'concentration_score': result.concentration_score,
            'computation_time': result.computation_time
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "explanation_summary.csv", index=False)

    # Save fairness analysis
    if 'fst_statistics' in fairness_analysis:
        import json
        with open(output_dir / "fairness_analysis.json", 'w') as f:
            # Convert numpy types to native Python types
            fst_stats = fairness_analysis['fst_statistics']
            fst_stats_serializable = {}
            for k, v in fst_stats.items():
                if isinstance(v, dict):
                    fst_stats_serializable[k] = {
                        kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()
                    }
                else:
                    fst_stats_serializable[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v

            json.dump(fst_stats_serializable, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SHAP Explainability Demo for Skin Cancer Detection")

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['hybrid', 'resnet18', 'resnet50'],
                       help='Model architecture type')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='Number of classes')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing dataset')
    parser.add_argument('--metadata_file', type=str, default='metadata.csv',
                       help='Metadata CSV filename')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to use')

    # Explainer arguments
    parser.add_argument('--method', type=str, default='gradient_shap',
                       choices=['gradient_shap', 'integrated_gradients', 'saliency'],
                       help='Attribution method')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to explain')
    parser.add_argument('--n_background', type=int, default=50,
                       help='Number of background samples for GradientSHAP')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/explanations',
                       help='Output directory for results')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Computation device')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("SHAP Explainability Demo for Skin Cancer Detection")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

    # 1. Load model
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading Model")
    logger.info("=" * 80)
    model = load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
        device=device
    )

    # 2. Load dataset
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Loading Dataset")
    logger.info("=" * 80)
    dataset = load_dataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        split=args.split,
        max_samples=args.num_samples * 2  # Load extra for background
    )

    # 3. Create explainer
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Creating Explainer")
    logger.info("=" * 80)
    explainer = SHAPExplainer(
        model=model,
        device=device,
        method=args.method,
        n_background_samples=args.n_background
    )

    # 4. Set background data (if needed)
    if args.method == "gradient_shap":
        logger.info("Creating background dataset for GradientSHAP...")
        background_loader = create_background_dataset(
            dataset=dataset,
            n_samples=args.n_background
        )
        explainer.set_background_data(background_loader)

    # 5. Generate explanations
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Generating Explanations")
    logger.info("=" * 80)
    start_time = time.time()

    results = generate_explanations(
        explainer=explainer,
        dataset=dataset,
        num_samples=args.num_samples
    )

    total_time = time.time() - start_time
    avg_time = total_time / len(results)

    logger.info(f"Generated {len(results)} explanations in {total_time:.2f}s")
    logger.info(f"Average time per explanation: {avg_time:.3f}s")

    # 6. Fairness analysis
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Fairness Analysis")
    logger.info("=" * 80)
    fairness_analysis = analyze_fairness(
        results=results,
        explainer=explainer,
        output_dir=output_dir
    )

    # 7. Create visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Creating Visualizations")
    logger.info("=" * 80)
    create_visualizations(
        results=results,
        results_by_fst=fairness_analysis.get('results_by_fst', {}),
        comparison_results=fairness_analysis.get('comparisons', []),
        output_dir=output_dir
    )

    # 8. Save results
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Saving Results")
    logger.info("=" * 80)
    save_results(
        results=results,
        fairness_analysis=fairness_analysis,
        output_dir=output_dir
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Total samples processed: {len(results)}")
    logger.info(f"Average attribution magnitude: {np.mean([r.attribution_magnitude for r in results]):.6f}")
    logger.info(f"Average concentration score: {np.mean([r.concentration_score for r in results]):.3f}")
    logger.info(f"Average computation time: {avg_time:.3f}s")

    if fairness_analysis.get('comparisons'):
        logger.info(f"\nFairness Comparisons: {len(fairness_analysis['comparisons'])}")
        for comp in fairness_analysis['comparisons']:
            logger.info(f"  FST {comp.fst_group_1} vs {comp.fst_group_2}:")
            logger.info(f"    Magnitude ratio: {comp.magnitude_ratio:.3f}")
            logger.info(f"    Spatial correlation: {comp.spatial_correlation:.3f}")
            logger.info(f"    Significant: {comp.significant} (p={comp.p_value:.4f})")

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
