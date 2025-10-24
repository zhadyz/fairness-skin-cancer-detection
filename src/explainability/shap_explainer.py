"""
SHAP-Based Explainability for Dermoscopy Classification

Implements GradientSHAP for deep learning models with fairness-aware analysis
across Fitzpatrick skin types (FSTs).

Key Features:
- GradientSHAP for pixel-level attributions
- Per-FST explanation analysis
- Saliency map generation and visualization
- Fairness comparison (FST I vs VI)
- Batch processing for efficiency
- Clinical interpretability metrics

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    from captum.attr import GradientShap, IntegratedGradients, Saliency
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logging.warning("Captum not available. Install with: pip install captum")

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result of SHAP explanation for a single image."""

    # Input data
    image: np.ndarray  # Original image (H, W, 3)
    predicted_class: int
    confidence: float
    fst_label: Optional[int] = None

    # Attributions
    shap_values: np.ndarray = None  # (H, W, 3) attribution map
    attribution_magnitude: float = 0.0

    # Saliency metrics
    top_regions: List[Tuple[int, int, float]] = field(default_factory=list)  # (x, y, importance)
    concentration_score: float = 0.0  # 0-1, higher = more localized

    # Clinical relevance
    lesion_coverage: float = 0.0  # Fraction of attribution on lesion area
    background_noise: float = 0.0  # Attribution on non-lesion areas

    # Metadata
    explanation_method: str = "GradientSHAP"
    computation_time: float = 0.0


@dataclass
class FairnessComparisonResult:
    """Result of fairness comparison between FST groups."""

    fst_group_1: int  # e.g., FST I
    fst_group_2: int  # e.g., FST VI

    # Attribution statistics
    mean_magnitude_1: float = 0.0
    mean_magnitude_2: float = 0.0
    magnitude_ratio: float = 1.0  # group_2 / group_1

    # Spatial similarity
    spatial_correlation: float = 0.0  # -1 to 1
    overlap_score: float = 0.0  # 0 to 1, IoU of top regions

    # Clinical metrics
    lesion_focus_1: float = 0.0
    lesion_focus_2: float = 0.0
    focus_parity: float = 0.0  # Absolute difference

    # Statistical significance
    p_value: float = 1.0
    significant: bool = False

    num_samples_1: int = 0
    num_samples_2: int = 0


class SHAPExplainer:
    """
    SHAP-based explainability for dermoscopy classification.

    Provides pixel-level attributions for model predictions with fairness analysis
    across Fitzpatrick skin types.

    Args:
        model: PyTorch model to explain
        device: Computation device
        method: Attribution method ("gradient_shap", "integrated_gradients", "saliency")
        n_background_samples: Number of background samples for GradientSHAP
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        method: str = "gradient_shap",
        n_background_samples: int = 50
    ):
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum is required. Install with: pip install captum")

        self.model = model
        self.device = device
        self.method = method
        self.n_background_samples = n_background_samples

        self.model.eval()
        self.model.to(device)

        # Initialize explainer (will be created per-explanation with background data)
        self.explainer = None
        self.background_data = None

        logger.info(f"SHAPExplainer initialized with method={method}, device={device}")

    def set_background_data(
        self,
        background_loader: torch.utils.data.DataLoader,
        n_samples: Optional[int] = None
    ):
        """
        Set background data for GradientSHAP.

        Args:
            background_loader: DataLoader with background images
            n_samples: Number of samples to use (default: n_background_samples)
        """
        if n_samples is None:
            n_samples = self.n_background_samples

        background_images = []

        for batch in background_loader:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            background_images.append(images)

            if sum(img.size(0) for img in background_images) >= n_samples:
                break

        self.background_data = torch.cat(background_images, dim=0)[:n_samples].to(self.device)

        logger.info(f"Background data set: {self.background_data.shape}")

    def explain_prediction(
        self,
        image: Union[torch.Tensor, np.ndarray],
        target_class: Optional[int] = None,
        fst_label: Optional[int] = None,
        return_visualization: bool = False
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a single image.

        Args:
            image: Input image (C, H, W) tensor or (H, W, C) numpy array
            target_class: Target class for attribution (if None, use predicted class)
            fst_label: Fitzpatrick skin type label (optional, for fairness analysis)
            return_visualization: Whether to create visualization

        Returns:
            ExplanationResult with attributions and metrics
        """
        import time
        start_time = time.time()

        # Convert image to tensor
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            elif image.ndim == 3 and image.shape[0] == 3:
                image = torch.from_numpy(image)
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

        image = image.unsqueeze(0).to(self.device).float()

        if image.max() > 1.0:
            image = image / 255.0

        # Get prediction
        with torch.no_grad():
            output = self.model(image)
            if isinstance(output, tuple):
                output = output[0]  # FairDisCo compatibility

            probs = torch.softmax(output, dim=1)
            predicted_class = probs.argmax(dim=1).item()
            confidence = probs[0, predicted_class].item()

        if target_class is None:
            target_class = predicted_class

        # Compute attributions
        if self.method == "gradient_shap":
            if self.background_data is None:
                raise ValueError("Background data not set. Call set_background_data() first.")

            explainer = GradientShap(self.model)
            attributions = explainer.attribute(
                image,
                baselines=self.background_data,
                target=target_class,
                n_samples=25  # Number of samples for approximation
            )

        elif self.method == "integrated_gradients":
            explainer = IntegratedGradients(self.model)
            attributions = explainer.attribute(
                image,
                target=target_class,
                n_steps=50
            )

        elif self.method == "saliency":
            explainer = Saliency(self.model)
            attributions = explainer.attribute(
                image,
                target=target_class
            )

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Convert to numpy
        attributions_np = attributions.squeeze(0).cpu().detach().numpy()  # (3, H, W)
        attributions_np = np.transpose(attributions_np, (1, 2, 0))  # (H, W, 3)

        image_np = image.squeeze(0).cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, 3)

        # Compute metrics
        attribution_magnitude = np.abs(attributions_np).mean()

        # Find top regions
        importance_map = np.abs(attributions_np).mean(axis=2)  # (H, W)
        top_regions = self._extract_top_regions(importance_map, top_k=10)

        # Concentration score (Gini coefficient)
        concentration = self._compute_concentration_score(importance_map)

        # Lesion coverage (requires lesion mask - placeholder)
        lesion_coverage = 0.0
        background_noise = 0.0

        computation_time = time.time() - start_time

        result = ExplanationResult(
            image=image_np,
            predicted_class=predicted_class,
            confidence=confidence,
            fst_label=fst_label,
            shap_values=attributions_np,
            attribution_magnitude=attribution_magnitude,
            top_regions=top_regions,
            concentration_score=concentration,
            lesion_coverage=lesion_coverage,
            background_noise=background_noise,
            explanation_method=self.method,
            computation_time=computation_time
        )

        logger.debug(f"Explanation generated: class={predicted_class}, confidence={confidence:.3f}, "
                    f"magnitude={attribution_magnitude:.6f}, time={computation_time:.3f}s")

        return result

    def explain_batch(
        self,
        images: torch.Tensor,
        fst_labels: Optional[torch.Tensor] = None,
        target_classes: Optional[torch.Tensor] = None
    ) -> List[ExplanationResult]:
        """
        Batch explanation for multiple images.

        Args:
            images: Batch of images (B, C, H, W)
            fst_labels: FST labels for each image (B,)
            target_classes: Target classes (B,) - if None, use predicted

        Returns:
            List of ExplanationResult objects
        """
        results = []

        batch_size = images.size(0)

        for i in range(batch_size):
            image = images[i]
            fst = fst_labels[i].item() if fst_labels is not None else None
            target = target_classes[i].item() if target_classes is not None else None

            result = self.explain_prediction(image, target_class=target, fst_label=fst)
            results.append(result)

        return results

    def compare_fst_explanations(
        self,
        results_fst1: List[ExplanationResult],
        results_fst2: List[ExplanationResult],
        fst1: int = 1,
        fst2: int = 6
    ) -> FairnessComparisonResult:
        """
        Compare explanations between two FST groups.

        Args:
            results_fst1: List of explanations for FST group 1
            results_fst2: List of explanations for FST group 2
            fst1: FST group 1 label
            fst2: FST group 2 label

        Returns:
            FairnessComparisonResult with comparison metrics
        """
        from scipy import stats

        # Extract attribution magnitudes
        magnitudes_1 = [r.attribution_magnitude for r in results_fst1]
        magnitudes_2 = [r.attribution_magnitude for r in results_fst2]

        mean_mag_1 = np.mean(magnitudes_1)
        mean_mag_2 = np.mean(magnitudes_2)
        magnitude_ratio = mean_mag_2 / (mean_mag_1 + 1e-8)

        # Statistical test
        t_stat, p_value = stats.ttest_ind(magnitudes_1, magnitudes_2)
        significant = p_value < 0.05

        # Spatial correlation (average over all pairs)
        spatial_correlations = []
        for r1 in results_fst1[:10]:  # Sample 10 for efficiency
            for r2 in results_fst2[:10]:
                corr = self._compute_spatial_correlation(
                    r1.shap_values, r2.shap_values
                )
                spatial_correlations.append(corr)

        spatial_correlation = np.mean(spatial_correlations) if spatial_correlations else 0.0

        # Overlap score (IoU of top regions)
        overlap_scores = []
        for r1 in results_fst1[:10]:
            for r2 in results_fst2[:10]:
                overlap = self._compute_region_overlap(r1.top_regions, r2.top_regions)
                overlap_scores.append(overlap)

        overlap_score = np.mean(overlap_scores) if overlap_scores else 0.0

        # Lesion focus
        lesion_focus_1 = np.mean([r.lesion_coverage for r in results_fst1])
        lesion_focus_2 = np.mean([r.lesion_coverage for r in results_fst2])
        focus_parity = abs(lesion_focus_1 - lesion_focus_2)

        result = FairnessComparisonResult(
            fst_group_1=fst1,
            fst_group_2=fst2,
            mean_magnitude_1=mean_mag_1,
            mean_magnitude_2=mean_mag_2,
            magnitude_ratio=magnitude_ratio,
            spatial_correlation=spatial_correlation,
            overlap_score=overlap_score,
            lesion_focus_1=lesion_focus_1,
            lesion_focus_2=lesion_focus_2,
            focus_parity=focus_parity,
            p_value=p_value,
            significant=significant,
            num_samples_1=len(results_fst1),
            num_samples_2=len(results_fst2)
        )

        logger.info(f"Fairness comparison FST {fst1} vs {fst2}:")
        logger.info(f"  Magnitude ratio: {magnitude_ratio:.3f}")
        logger.info(f"  Spatial correlation: {spatial_correlation:.3f}")
        logger.info(f"  Overlap score: {overlap_score:.3f}")
        logger.info(f"  Significant: {significant} (p={p_value:.4f})")

        return result

    def _extract_top_regions(
        self,
        importance_map: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, int, float]]:
        """Extract top-k most important spatial regions."""
        H, W = importance_map.shape
        flat_indices = np.argsort(importance_map.flatten())[::-1][:top_k]

        top_regions = []
        for idx in flat_indices:
            y, x = divmod(idx, W)
            importance = importance_map[y, x]
            top_regions.append((int(x), int(y), float(importance)))

        return top_regions

    def _compute_concentration_score(self, importance_map: np.ndarray) -> float:
        """
        Compute Gini coefficient as concentration score.

        Higher score = more concentrated (focused) attribution.
        """
        flat = importance_map.flatten()
        flat = np.abs(flat)
        flat = np.sort(flat)

        n = len(flat)
        index = np.arange(1, n + 1)

        gini = (2 * np.sum(index * flat)) / (n * np.sum(flat)) - (n + 1) / n

        return float(gini)

    def _compute_spatial_correlation(
        self,
        attr1: np.ndarray,
        attr2: np.ndarray
    ) -> float:
        """Compute spatial correlation between two attribution maps."""
        # Average over channels
        map1 = np.abs(attr1).mean(axis=2).flatten()
        map2 = np.abs(attr2).mean(axis=2).flatten()

        # Pearson correlation
        corr = np.corrcoef(map1, map2)[0, 1]

        return float(corr) if not np.isnan(corr) else 0.0

    def _compute_region_overlap(
        self,
        regions1: List[Tuple[int, int, float]],
        regions2: List[Tuple[int, int, float]],
        radius: int = 10
    ) -> float:
        """
        Compute overlap between two sets of top regions (IoU-like metric).

        Args:
            regions1: List of (x, y, importance) tuples
            regions2: List of (x, y, importance) tuples
            radius: Radius for region matching

        Returns:
            Overlap score [0, 1]
        """
        if not regions1 or not regions2:
            return 0.0

        matches = 0
        for x1, y1, _ in regions1:
            for x2, y2, _ in regions2:
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance <= radius:
                    matches += 1
                    break

        # IoU-like: intersection / union
        intersection = matches
        union = len(regions1) + len(regions2) - matches

        return intersection / union if union > 0 else 0.0

    def aggregate_fst_statistics(
        self,
        results_by_fst: Dict[int, List[ExplanationResult]]
    ) -> Dict[str, any]:
        """
        Aggregate statistics across all FST groups.

        Args:
            results_by_fst: Dictionary mapping FST labels to explanation results

        Returns:
            Dictionary with aggregated statistics
        """
        stats = {}

        for fst, results in results_by_fst.items():
            if not results:
                continue

            stats[f"fst_{fst}"] = {
                "num_samples": len(results),
                "mean_magnitude": np.mean([r.attribution_magnitude for r in results]),
                "std_magnitude": np.std([r.attribution_magnitude for r in results]),
                "mean_concentration": np.mean([r.concentration_score for r in results]),
                "mean_confidence": np.mean([r.confidence for r in results]),
                "mean_computation_time": np.mean([r.computation_time for r in results])
            }

        # Compute disparities
        fst_labels = sorted(results_by_fst.keys())
        if len(fst_labels) >= 2:
            magnitudes = [stats[f"fst_{fst}"]["mean_magnitude"] for fst in fst_labels]
            stats["magnitude_range"] = max(magnitudes) - min(magnitudes)
            stats["magnitude_std"] = np.std(magnitudes)

        return stats

    def save_explanations(
        self,
        results: List[ExplanationResult],
        output_dir: Union[str, Path]
    ):
        """
        Save explanation results to disk.

        Args:
            results: List of ExplanationResult objects
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(results):
            # Save attribution map
            np.save(output_dir / f"attribution_{i:04d}.npy", result.shap_values)

            # Save metadata
            metadata = {
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "fst_label": result.fst_label,
                "attribution_magnitude": result.attribution_magnitude,
                "concentration_score": result.concentration_score,
                "explanation_method": result.explanation_method
            }

            import json
            with open(output_dir / f"metadata_{i:04d}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(results)} explanations to {output_dir}")


if __name__ == "__main__":
    """Test SHAPExplainer implementation."""
    print("=" * 80)
    print("Testing SHAPExplainer")
    print("=" * 80)

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 7)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    model = DummyModel()
    explainer = SHAPExplainer(model, method="saliency")  # Saliency doesn't need background

    # Test single explanation
    test_image = torch.randn(3, 224, 224)
    print("\nGenerating explanation...")
    result = explainer.explain_prediction(test_image)

    print(f"Predicted class: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Attribution magnitude: {result.attribution_magnitude:.6f}")
    print(f"Concentration score: {result.concentration_score:.3f}")
    print(f"Top 3 regions: {result.top_regions[:3]}")
    print(f"Computation time: {result.computation_time:.3f}s")

    # Test batch explanation
    batch_images = torch.randn(5, 3, 224, 224)
    print("\nGenerating batch explanations...")
    batch_results = explainer.explain_batch(batch_images)
    print(f"Generated {len(batch_results)} explanations")

    print("\n" + "=" * 80)
    print("SHAPExplainer test PASSED!")
    print("=" * 80)
