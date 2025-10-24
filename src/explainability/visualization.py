"""
Visualization Utilities for SHAP Explanations

Creates visual overlays, heatmaps, and comparative dashboards for
SHAP-based explanations in dermoscopy classification.

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import logging

from .shap_explainer import ExplanationResult, FairnessComparisonResult

logger = logging.getLogger(__name__)


# Custom colormap for SHAP values (blue-white-red)
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap",
    ["#0000FF", "#FFFFFF", "#FF0000"],
    N=256
)


def create_saliency_overlay(
    image: np.ndarray,
    attribution: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    clip_percentile: float = 99.0
) -> np.ndarray:
    """
    Create saliency heatmap overlay on original image.

    Args:
        image: Original image (H, W, 3), values [0, 1]
        attribution: Attribution map (H, W, 3), SHAP values
        alpha: Overlay transparency (0 = invisible, 1 = opaque)
        colormap: Matplotlib colormap name
        clip_percentile: Percentile for clipping extreme values

    Returns:
        Overlay image (H, W, 3) with heatmap
    """
    # Average attribution across channels
    attr_magnitude = np.abs(attribution).mean(axis=2)  # (H, W)

    # Normalize to [0, 1]
    clip_val = np.percentile(attr_magnitude, clip_percentile)
    attr_magnitude = np.clip(attr_magnitude, 0, clip_val)
    attr_magnitude = attr_magnitude / (attr_magnitude.max() + 1e-8)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(attr_magnitude)[:, :, :3]  # (H, W, 3), drop alpha

    # Ensure image is in [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    # Blend image with heatmap
    overlay = (1 - alpha) * image + alpha * heatmap

    return overlay


def create_shap_heatmap(
    attribution: np.ndarray,
    clip_percentile: float = 99.0,
    use_absolute: bool = True
) -> np.ndarray:
    """
    Create pure SHAP heatmap (no image overlay).

    Args:
        attribution: Attribution map (H, W, 3)
        clip_percentile: Percentile for clipping
        use_absolute: Use absolute values (magnitude) or signed values

    Returns:
        Heatmap (H, W, 3)
    """
    if use_absolute:
        attr_map = np.abs(attribution).mean(axis=2)
        cmap = plt.get_cmap("hot")
    else:
        # Signed: positive (red) vs negative (blue)
        attr_map = attribution.mean(axis=2)
        cmap = SHAP_CMAP

    # Normalize
    clip_val = np.percentile(np.abs(attr_map), clip_percentile)
    if use_absolute:
        attr_map = np.clip(attr_map, 0, clip_val)
        attr_map = attr_map / (attr_map.max() + 1e-8)
    else:
        attr_map = np.clip(attr_map, -clip_val, clip_val)
        attr_map = (attr_map + clip_val) / (2 * clip_val + 1e-8)  # Scale to [0, 1]

    heatmap = cmap(attr_map)[:, :, :3]

    return heatmap


def visualize_explanation(
    result: ExplanationResult,
    show_top_regions: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    save_svg: bool = False
) -> plt.Figure:
    """
    Create comprehensive visualization of explanation result.

    Args:
        result: ExplanationResult object
        show_top_regions: Whether to mark top regions
        save_path: Path to save figure (if None, don't save)
        dpi: Figure DPI
        save_svg: If True, also save SVG version

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax = axes[0]
    ax.imshow(result.image)
    ax.set_title("Original Image", fontsize=12, fontweight='bold')
    ax.axis("off")

    # SHAP heatmap
    ax = axes[1]
    heatmap = create_shap_heatmap(result.shap_values)
    ax.imshow(heatmap)
    ax.set_title("SHAP Heatmap", fontsize=12, fontweight='bold')
    ax.axis("off")

    # Overlay
    ax = axes[2]
    overlay = create_saliency_overlay(result.image, result.shap_values, alpha=0.5)
    ax.imshow(overlay)

    # Mark top regions
    if show_top_regions and result.top_regions:
        for i, (x, y, importance) in enumerate(result.top_regions[:5]):
            circle = patches.Circle((x, y), radius=5, color='yellow', fill=False, linewidth=2)
            ax.add_patch(circle)
            if i == 0:  # Label first region
                ax.text(x, y - 10, "Top", color='yellow', fontsize=10, ha='center',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ax.set_title("Saliency Overlay", fontsize=12, fontweight='bold')
    ax.axis("off")

    # Add text info
    info_text = f"Predicted: Class {result.predicted_class} ({result.confidence:.2%})\n"
    info_text += f"Method: {result.explanation_method}\n"
    info_text += f"Attribution: {result.attribution_magnitude:.6f}\n"
    info_text += f"Concentration: {result.concentration_score:.3f}"
    if result.fst_label is not None:
        info_text += f"\nFST: {result.fst_label}"

    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")

        # Save SVG version if requested
        if save_svg:
            svg_path = save_path.with_suffix('.svg')
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            logger.info(f"Saved SVG version to {svg_path}")

    return fig


def visualize_fst_comparison(
    result: FairnessComparisonResult,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create visualization comparing explanations between FST groups.

    Args:
        result: FairnessComparisonResult object
        save_path: Path to save figure
        dpi: Figure DPI

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Attribution magnitude comparison
    ax = axes[0, 0]
    fst_labels = [f"FST {result.fst_group_1}", f"FST {result.fst_group_2}"]
    magnitudes = [result.mean_magnitude_1, result.mean_magnitude_2]
    bars = ax.bar(fst_labels, magnitudes, color=['skyblue', 'coral'])
    ax.set_ylabel("Mean Attribution Magnitude")
    ax.set_title("Attribution Magnitude by FST")
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mag in zip(bars, magnitudes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mag:.6f}', ha='center', va='bottom')

    # 2. Fairness metrics
    ax = axes[0, 1]
    metrics = {
        'Magnitude\nRatio': result.magnitude_ratio,
        'Spatial\nCorrelation': result.spatial_correlation,
        'Overlap\nScore': result.overlap_score,
        'Focus\nParity': 1 - result.focus_parity  # Invert so higher is better
    }

    x = np.arange(len(metrics))
    values = list(metrics.values())
    bars = ax.bar(x, values, color=['green', 'blue', 'purple', 'orange'])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys(), fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Fairness Metrics")
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal Parity')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Lesion focus comparison
    ax = axes[1, 0]
    focus_data = [result.lesion_focus_1, result.lesion_focus_2]
    bars = ax.bar(fst_labels, focus_data, color=['skyblue', 'coral'])
    ax.set_ylabel("Lesion Focus Score")
    ax.set_title("Clinical Relevance: Lesion Focus")
    ax.grid(axis='y', alpha=0.3)

    for bar, focus in zip(bars, focus_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{focus:.3f}', ha='center', va='bottom')

    # 4. Statistical summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"Fairness Comparison: FST {result.fst_group_1} vs FST {result.fst_group_2}\n\n"
    summary_text += f"Sample sizes:\n"
    summary_text += f"  FST {result.fst_group_1}: {result.num_samples_1}\n"
    summary_text += f"  FST {result.fst_group_2}: {result.num_samples_2}\n\n"
    summary_text += f"Statistical Significance:\n"
    summary_text += f"  p-value: {result.p_value:.4f}\n"
    summary_text += f"  Significant: {'YES' if result.significant else 'NO'} (α=0.05)\n\n"
    summary_text += f"Interpretation:\n"
    if result.magnitude_ratio > 1.2:
        summary_text += f"  FST {result.fst_group_2} has {result.magnitude_ratio:.1f}x stronger\n"
        summary_text += f"  attributions (potential bias signal)\n"
    elif result.magnitude_ratio < 0.8:
        summary_text += f"  FST {result.fst_group_1} has {1/result.magnitude_ratio:.1f}x stronger\n"
        summary_text += f"  attributions (potential bias signal)\n"
    else:
        summary_text += f"  Attribution magnitudes are similar\n"
        summary_text += f"  (ratio={result.magnitude_ratio:.2f}, good parity)\n"

    if result.spatial_correlation > 0.7:
        summary_text += f"  High spatial agreement ({result.spatial_correlation:.2f})\n"
        summary_text += f"  → Model focuses on similar regions\n"
    elif result.spatial_correlation < 0.3:
        summary_text += f"  Low spatial agreement ({result.spatial_correlation:.2f})\n"
        summary_text += f"  → Model focuses on different regions (concern)\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           family='monospace')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved FST comparison to {save_path}")

    return fig


class ExplanationVisualizer:
    """
    High-level visualizer for batch explanation analysis.

    Provides methods for creating dashboards, reports, and comparative analyses.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ExplanationVisualizer initialized: output_dir={self.output_dir}")

    def create_sample_grid(
        self,
        results: List[ExplanationResult],
        n_samples: int = 9,
        save_name: str = "sample_grid.png",
        dpi: int = 150
    ) -> plt.Figure:
        """
        Create grid of sample explanations.

        Args:
            results: List of ExplanationResult objects
            n_samples: Number of samples to show
            save_name: Filename for saving
            dpi: Figure DPI

        Returns:
            Matplotlib figure
        """
        n_samples = min(n_samples, len(results))
        n_cols = 3
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for i, result in enumerate(results[:n_samples]):
            ax = axes[i]

            # Create overlay
            overlay = create_saliency_overlay(result.image, result.shap_values)
            ax.imshow(overlay)

            title = f"Class {result.predicted_class} ({result.confidence:.2f})"
            if result.fst_label is not None:
                title += f" | FST {result.fst_label}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved sample grid to {save_path}")

        return fig

    def create_fst_dashboard(
        self,
        results_by_fst: Dict[int, List[ExplanationResult]],
        save_name: str = "fst_dashboard.png",
        dpi: int = 150
    ) -> plt.Figure:
        """
        Create comprehensive dashboard comparing all FST groups.

        Args:
            results_by_fst: Dictionary mapping FST labels to results
            save_name: Filename for saving
            dpi: Figure DPI

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fst_labels = sorted(results_by_fst.keys())

        # 1. Attribution magnitude distribution
        ax = fig.add_subplot(gs[0, :])
        for fst in fst_labels:
            magnitudes = [r.attribution_magnitude for r in results_by_fst[fst]]
            ax.hist(magnitudes, alpha=0.5, label=f"FST {fst}", bins=20)
        ax.set_xlabel("Attribution Magnitude")
        ax.set_ylabel("Frequency")
        ax.set_title("Attribution Magnitude Distribution by FST")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. Mean metrics by FST
        ax = fig.add_subplot(gs[1, 0])
        mean_mags = [np.mean([r.attribution_magnitude for r in results_by_fst[fst]])
                     for fst in fst_labels]
        ax.bar([f"FST {fst}" for fst in fst_labels], mean_mags, color='skyblue')
        ax.set_ylabel("Mean Attribution")
        ax.set_title("Mean Attribution by FST")
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 3. Concentration scores
        ax = fig.add_subplot(gs[1, 1])
        mean_conc = [np.mean([r.concentration_score for r in results_by_fst[fst]])
                     for fst in fst_labels]
        ax.bar([f"FST {fst}" for fst in fst_labels], mean_conc, color='coral')
        ax.set_ylabel("Mean Concentration")
        ax.set_title("Attribution Concentration by FST")
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 4. Confidence by FST
        ax = fig.add_subplot(gs[1, 2])
        mean_conf = [np.mean([r.confidence for r in results_by_fst[fst]])
                     for fst in fst_labels]
        ax.bar([f"FST {fst}" for fst in fst_labels], mean_conf, color='lightgreen')
        ax.set_ylabel("Mean Confidence")
        ax.set_title("Prediction Confidence by FST")
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 5. Sample explanations (one per FST)
        for i, fst in enumerate(fst_labels[:3]):
            ax = fig.add_subplot(gs[2, i])
            if results_by_fst[fst]:
                result = results_by_fst[fst][0]  # First sample
                overlay = create_saliency_overlay(result.image, result.shap_values)
                ax.imshow(overlay)
                ax.set_title(f"FST {fst} Sample", fontsize=10)
            ax.axis('off')

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved FST dashboard to {save_path}")

        return fig

    def generate_html_report(
        self,
        results_by_fst: Dict[int, List[ExplanationResult]],
        comparison_results: List[FairnessComparisonResult],
        report_name: str = "explanation_report.html"
    ):
        """
        Generate HTML report with explanations and fairness analysis.

        Args:
            results_by_fst: Dictionary mapping FST to results
            comparison_results: List of FairnessComparisonResult objects
            report_name: HTML filename
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SHAP Explanation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .summary { background-color: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .metric { display: inline-block; margin: 10px 20px; }
                .metric-label { font-weight: bold; color: #666; }
                .metric-value { font-size: 1.2em; color: #4CAF50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .significant { color: #d9534f; font-weight: bold; }
                .not-significant { color: #5cb85c; }
            </style>
        </head>
        <body>
            <h1>SHAP Explainability Report</h1>
        """

        # Summary statistics
        html += '<div class="summary"><h2>Summary Statistics</h2>'
        total_samples = sum(len(results) for results in results_by_fst.values())
        html += f'<div class="metric"><span class="metric-label">Total Samples:</span> <span class="metric-value">{total_samples}</span></div>'
        html += f'<div class="metric"><span class="metric-label">FST Groups:</span> <span class="metric-value">{len(results_by_fst)}</span></div>'
        html += '</div>'

        # Per-FST statistics
        html += '<h2>Per-FST Statistics</h2><table><tr><th>FST</th><th>Samples</th><th>Mean Attribution</th><th>Mean Concentration</th><th>Mean Confidence</th></tr>'

        for fst in sorted(results_by_fst.keys()):
            results = results_by_fst[fst]
            mean_attr = np.mean([r.attribution_magnitude for r in results])
            mean_conc = np.mean([r.concentration_score for r in results])
            mean_conf = np.mean([r.confidence for r in results])

            html += f'<tr><td>FST {fst}</td><td>{len(results)}</td><td>{mean_attr:.6f}</td><td>{mean_conc:.3f}</td><td>{mean_conf:.3f}</td></tr>'

        html += '</table>'

        # Fairness comparisons
        html += '<h2>Fairness Comparisons</h2><table><tr><th>Comparison</th><th>Magnitude Ratio</th><th>Spatial Corr</th><th>Overlap</th><th>p-value</th><th>Significant</th></tr>'

        for comp in comparison_results:
            sig_class = "significant" if comp.significant else "not-significant"
            sig_text = "YES" if comp.significant else "NO"

            html += f'<tr><td>FST {comp.fst_group_1} vs {comp.fst_group_2}</td>'
            html += f'<td>{comp.magnitude_ratio:.3f}</td>'
            html += f'<td>{comp.spatial_correlation:.3f}</td>'
            html += f'<td>{comp.overlap_score:.3f}</td>'
            html += f'<td>{comp.p_value:.4f}</td>'
            html += f'<td class="{sig_class}">{sig_text}</td></tr>'

        html += '</table></body></html>'

        # Save HTML
        report_path = self.output_dir / report_name
        with open(report_path, 'w') as f:
            f.write(html)

        logger.info(f"Generated HTML report: {report_path}")

    def export_clinical_report(
        self,
        result: ExplanationResult,
        patient_id: str = "UNKNOWN",
        report_name: str = "clinical_report.png",
        include_metadata: bool = True
    ) -> Path:
        """
        Export a clinical-focused report for a single case.

        Designed for presentation to medical professionals.

        Args:
            result: ExplanationResult for the case
            patient_id: Patient identifier (anonymized)
            report_name: Output filename
            include_metadata: Include technical metadata

        Returns:
            Path to saved report
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f"AI Explanation Report - Patient {patient_id}",
                    fontsize=16, fontweight='bold', y=0.98)

        # Row 1: Original, Heatmap, Overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(result.image)
        ax1.set_title("Original Image", fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        heatmap = create_shap_heatmap(result.shap_values)
        ax2.imshow(heatmap)
        ax2.set_title("AI Attention Heatmap", fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        overlay = create_saliency_overlay(result.image, result.shap_values, alpha=0.6)
        ax3.imshow(overlay)

        # Mark top regions
        if result.top_regions:
            for i, (x, y, importance) in enumerate(result.top_regions[:3]):
                circle = patches.Circle((x, y), radius=8, color='cyan',
                                      fill=False, linewidth=3)
                ax3.add_patch(circle)
                ax3.text(x, y - 15, f"{i+1}", color='cyan', fontsize=12,
                        ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax3.set_title("Critical Regions Identified", fontweight='bold')
        ax3.axis('off')

        # Row 2: Prediction info and interpretation guide
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.axis('off')

        # Prediction information
        pred_text = "═══ PREDICTION SUMMARY ═══\n\n"
        pred_text += f"Predicted Class: {result.predicted_class}\n"
        pred_text += f"Confidence: {result.confidence:.1%}\n"
        pred_text += f"FST Type: {'I-III (Light)' if result.fst_label and result.fst_label <= 3 else 'IV-VI (Dark)' if result.fst_label else 'Unknown'}\n\n"

        pred_text += "═══ ATTRIBUTION METRICS ═══\n\n"
        pred_text += f"Attention Magnitude: {result.attribution_magnitude:.6f}\n"
        pred_text += f"  (Higher = stronger feature reliance)\n\n"
        pred_text += f"Concentration Score: {result.concentration_score:.3f}\n"
        pred_text += f"  (0.0-0.5: Diffuse, 0.5-0.8: Moderate, 0.8-1.0: Focused)\n\n"

        if result.concentration_score > 0.7:
            interpretation = "✓ Focused: Model relies on specific regions"
        elif result.concentration_score > 0.4:
            interpretation = "⚠ Moderate: Model uses multiple regions"
        else:
            interpretation = "⚠ Diffuse: Model attention is scattered"

        pred_text += f"Interpretation: {interpretation}\n"

        ax4.text(0.05, 0.95, pred_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Interpretation guide
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        guide_text = "═══ READING GUIDE ═══\n\n"
        guide_text += "Heatmap Colors:\n"
        guide_text += "• Red: High importance\n"
        guide_text += "• Yellow: Medium\n"
        guide_text += "• Blue: Low\n\n"

        guide_text += "Numbered Circles:\n"
        guide_text += "• Mark top 3 regions\n"
        guide_text += "• Where AI focuses\n"
        guide_text += "  most attention\n\n"

        guide_text += "Clinical Note:\n"
        guide_text += "• Use as decision\n"
        guide_text += "  support tool\n"
        guide_text += "• Not diagnostic\n"
        guide_text += "• Verify findings\n"

        ax5.text(0.05, 0.95, guide_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Row 3: Top regions table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        if result.top_regions:
            table_text = "═══ TOP ATTENTION REGIONS ═══\n\n"
            table_text += "Rank    X      Y      Importance     Location Hint\n"
            table_text += "─" * 60 + "\n"

            H, W = result.image.shape[:2]

            for i, (x, y, imp) in enumerate(result.top_regions[:5], 1):
                # Location hint based on position
                if x < W/3 and y < H/3:
                    location = "Upper-Left"
                elif x > 2*W/3 and y < H/3:
                    location = "Upper-Right"
                elif x < W/3 and y > 2*H/3:
                    location = "Lower-Left"
                elif x > 2*W/3 and y > 2*H/3:
                    location = "Lower-Right"
                elif y < H/3:
                    location = "Upper-Center"
                elif y > 2*H/3:
                    location = "Lower-Center"
                elif x < W/3:
                    location = "Left-Center"
                elif x > 2*W/3:
                    location = "Right-Center"
                else:
                    location = "Center"

                table_text += f"{i:3d}    {x:4d}   {y:4d}   {imp:9.6f}      {location}\n"

            ax6.text(0.05, 0.95, table_text, transform=ax6.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Metadata footer
        if include_metadata:
            footer = f"Method: {result.explanation_method} | "
            footer += f"Computation Time: {result.computation_time:.3f}s | "
            footer += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"

            fig.text(0.5, 0.01, footer, ha='center', fontsize=8,
                    style='italic', color='gray')

        # Save
        report_path = self.output_dir / report_name
        fig.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Clinical report saved to {report_path}")
        return report_path


if __name__ == "__main__":
    """Test visualization utilities."""
    print("=" * 80)
    print("Testing Visualization Utilities")
    print("=" * 80)

    # Create dummy data
    dummy_image = np.random.rand(224, 224, 3)
    dummy_attr = np.random.randn(224, 224, 3) * 0.1

    # Test saliency overlay
    print("\nTesting saliency overlay...")
    overlay = create_saliency_overlay(dummy_image, dummy_attr, alpha=0.5)
    print(f"Overlay shape: {overlay.shape}")
    print(f"Overlay range: [{overlay.min():.3f}, {overlay.max():.3f}]")

    # Test heatmap
    print("\nTesting SHAP heatmap...")
    heatmap = create_shap_heatmap(dummy_attr)
    print(f"Heatmap shape: {heatmap.shape}")

    print("\n" + "=" * 80)
    print("Visualization utilities test PASSED!")
    print("=" * 80)
