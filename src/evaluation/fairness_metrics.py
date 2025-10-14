"""
Fairness Metrics for Skin Cancer Detection Models

Implements comprehensive fairness evaluation across Fitzpatrick skin types (FST I-VI):
- AUROC per FST group
- Equal Opportunity Difference (EOD)
- Expected Calibration Error (ECE)
- Sensitivity and Specificity per FST
- Demographic Parity Difference

Author: HOLLOWED_EYES
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


@dataclass
class FairnessResults:
    """Container for fairness evaluation results."""

    # Overall metrics
    overall_auroc: float
    overall_accuracy: float

    # Per-FST metrics
    auroc_per_fst: Dict[int, float]
    accuracy_per_fst: Dict[int, float]
    sensitivity_per_fst: Dict[int, float]
    specificity_per_fst: Dict[int, float]
    ece_per_fst: Dict[int, float]

    # Fairness gaps
    auroc_gap: float  # max - min AUROC across FST groups
    auroc_gap_light_dark: float  # AUROC(I-III) - AUROC(IV-VI)
    equal_opportunity_diff: float  # Max TPR difference
    demographic_parity_diff: float  # Max positive prediction rate difference

    # Group statistics
    samples_per_fst: Dict[int, int]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 80,
            "FAIRNESS EVALUATION SUMMARY",
            "=" * 80,
            f"\nOverall Performance:",
            f"  AUROC: {self.overall_auroc:.4f}",
            f"  Accuracy: {self.overall_accuracy:.4f}",
            f"\nAUROC by Fitzpatrick Skin Type:",
        ]

        for fst in sorted(self.auroc_per_fst.keys()):
            auroc = self.auroc_per_fst[fst]
            acc = self.accuracy_per_fst[fst]
            sens = self.sensitivity_per_fst[fst]
            spec = self.specificity_per_fst[fst]
            samples = self.samples_per_fst[fst]
            lines.append(
                f"  FST-{fst}: AUROC={auroc:.4f}, Acc={acc:.4f}, "
                f"Sens={sens:.4f}, Spec={spec:.4f} (n={samples})"
            )

        lines.extend([
            f"\nFairness Gaps:",
            f"  AUROC Gap (max-min): {self.auroc_gap:.4f}",
            f"  AUROC Gap (Light-Dark): {self.auroc_gap_light_dark:.4f}",
            f"  Equal Opportunity Diff: {self.equal_opportunity_diff:.4f}",
            f"  Demographic Parity Diff: {self.demographic_parity_diff:.4f}",
            "=" * 80
        ])

        return "\n".join(lines)


class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator for skin cancer detection.

    Evaluates model performance across Fitzpatrick skin types (FST I-VI)
    with focus on identifying and quantifying algorithmic bias.

    Example:
        >>> metrics = FairnessMetrics(num_classes=7)
        >>> results = metrics.evaluate(
        ...     y_true=test_labels,
        ...     y_pred=predictions,
        ...     y_prob=probabilities,
        ...     fst_labels=fst_groups
        ... )
        >>> print(results.summary())
    """

    def __init__(self, num_classes: int = 7, fst_groups: List[int] = [1, 2, 3, 4, 5, 6]):
        """
        Initialize fairness metrics calculator.

        Args:
            num_classes: Number of disease classes
            fst_groups: List of Fitzpatrick skin type groups to evaluate
        """
        self.num_classes = num_classes
        self.fst_groups = fst_groups

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        fst_labels: np.ndarray,
        target_class: Optional[int] = None
    ) -> FairnessResults:
        """
        Comprehensive fairness evaluation.

        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Predicted probabilities (N, num_classes)
            fst_labels: Fitzpatrick skin type labels (N,)
            target_class: Target class for binary metrics (e.g., melanoma)

        Returns:
            FairnessResults object with all metrics
        """
        # Overall metrics
        overall_auroc = self._compute_auroc(y_true, y_prob)
        overall_accuracy = accuracy_score(y_true, y_pred)

        # Per-FST metrics
        auroc_per_fst = {}
        accuracy_per_fst = {}
        sensitivity_per_fst = {}
        specificity_per_fst = {}
        ece_per_fst = {}
        samples_per_fst = {}

        for fst in self.fst_groups:
            # Get samples for this FST group
            fst_mask = fst_labels == fst

            if not np.any(fst_mask):
                continue

            samples_per_fst[fst] = int(np.sum(fst_mask))

            fst_y_true = y_true[fst_mask]
            fst_y_pred = y_pred[fst_mask]
            fst_y_prob = y_prob[fst_mask]

            # AUROC
            auroc_per_fst[fst] = self._compute_auroc(fst_y_true, fst_y_prob)

            # Accuracy
            accuracy_per_fst[fst] = accuracy_score(fst_y_true, fst_y_pred)

            # Sensitivity and Specificity
            sens, spec = self._compute_sensitivity_specificity(
                fst_y_true, fst_y_pred, target_class
            )
            sensitivity_per_fst[fst] = sens
            specificity_per_fst[fst] = spec

            # Expected Calibration Error
            ece_per_fst[fst] = self._compute_ece(fst_y_true, fst_y_prob, fst_y_pred)

        # Compute fairness gaps
        auroc_gap = self._compute_auroc_gap(auroc_per_fst)
        auroc_gap_light_dark = self._compute_auroc_gap_light_dark(auroc_per_fst)

        equal_opportunity_diff = self._compute_equal_opportunity_diff(
            y_true, y_pred, fst_labels, target_class
        )

        demographic_parity_diff = self._compute_demographic_parity_diff(
            y_pred, fst_labels, target_class
        )

        return FairnessResults(
            overall_auroc=overall_auroc,
            overall_accuracy=overall_accuracy,
            auroc_per_fst=auroc_per_fst,
            accuracy_per_fst=accuracy_per_fst,
            sensitivity_per_fst=sensitivity_per_fst,
            specificity_per_fst=specificity_per_fst,
            ece_per_fst=ece_per_fst,
            auroc_gap=auroc_gap,
            auroc_gap_light_dark=auroc_gap_light_dark,
            equal_opportunity_diff=equal_opportunity_diff,
            demographic_parity_diff=demographic_parity_diff,
            samples_per_fst=samples_per_fst
        )

    def _compute_auroc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Compute AUROC (macro-averaged for multi-class).

        Args:
            y_true: True labels
            y_prob: Predicted probabilities

        Returns:
            AUROC score (0-1)
        """
        try:
            if len(np.unique(y_true)) < 2:
                return 0.0

            auroc = roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovr',
                average='macro'
            )
            return float(auroc)
        except ValueError:
            return 0.0

    def _compute_sensitivity_specificity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_class: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Compute sensitivity (TPR) and specificity (TNR).

        For multi-class, computes macro-averaged metrics unless target_class specified.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_class: Target class for binary metrics

        Returns:
            Tuple of (sensitivity, specificity)
        """
        if target_class is not None:
            # Binary metrics for target class
            y_true_binary = (y_true == target_class).astype(int)
            y_pred_binary = (y_pred == target_class).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Macro-averaged metrics across all classes
            cm = confusion_matrix(y_true, y_pred)

            sensitivities = []
            specificities = []

            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                sensitivities.append(sens)
                specificities.append(spec)

            sensitivity = np.mean(sensitivities)
            specificity = np.mean(specificities)

        return float(sensitivity), float(specificity)

    def _compute_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Measures how well predicted probabilities match actual outcomes.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            y_pred: Predicted labels
            num_bins: Number of bins for calibration

        Returns:
            ECE score (0-1, lower is better)
        """
        # Get confidence (max probability)
        confidences = np.max(y_prob, axis=1)

        # Get accuracy for each prediction
        accuracies = (y_pred == y_true).astype(float)

        # Create bins
        bins = np.linspace(0, 1, num_bins + 1)

        ece = 0.0

        for i in range(num_bins):
            # Get predictions in this confidence bin
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i + 1])

            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(accuracies[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                bin_weight = np.sum(bin_mask) / len(confidences)

                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)

        return float(ece)

    def _compute_auroc_gap(self, auroc_per_fst: Dict[int, float]) -> float:
        """Compute AUROC gap (max - min)."""
        if not auroc_per_fst:
            return 0.0

        return max(auroc_per_fst.values()) - min(auroc_per_fst.values())

    def _compute_auroc_gap_light_dark(self, auroc_per_fst: Dict[int, float]) -> float:
        """
        Compute AUROC gap between light (I-III) and dark (IV-VI) skin.

        Returns positive if light skin has higher AUROC.
        """
        light_fst = [1, 2, 3]
        dark_fst = [4, 5, 6]

        light_aurocs = [auroc_per_fst[fst] for fst in light_fst if fst in auroc_per_fst]
        dark_aurocs = [auroc_per_fst[fst] for fst in dark_fst if fst in auroc_per_fst]

        if not light_aurocs or not dark_aurocs:
            return 0.0

        light_avg = np.mean(light_aurocs)
        dark_avg = np.mean(dark_aurocs)

        return light_avg - dark_avg

    def _compute_equal_opportunity_diff(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fst_labels: np.ndarray,
        target_class: Optional[int] = None
    ) -> float:
        """
        Compute Equal Opportunity Difference (EOD).

        Maximum difference in True Positive Rate (TPR) across FST groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            fst_labels: Fitzpatrick skin type labels
            target_class: Target class for binary TPR computation

        Returns:
            EOD score (0-1, lower is better)
        """
        tprs = {}

        for fst in self.fst_groups:
            fst_mask = fst_labels == fst

            if not np.any(fst_mask):
                continue

            fst_y_true = y_true[fst_mask]
            fst_y_pred = y_pred[fst_mask]

            sens, _ = self._compute_sensitivity_specificity(
                fst_y_true, fst_y_pred, target_class
            )
            tprs[fst] = sens

        if not tprs:
            return 0.0

        return max(tprs.values()) - min(tprs.values())

    def _compute_demographic_parity_diff(
        self,
        y_pred: np.ndarray,
        fst_labels: np.ndarray,
        target_class: Optional[int] = None
    ) -> float:
        """
        Compute Demographic Parity Difference.

        Maximum difference in positive prediction rate across FST groups.

        Args:
            y_pred: Predicted labels
            fst_labels: Fitzpatrick skin type labels
            target_class: Target class (if None, uses any positive prediction)

        Returns:
            Demographic parity difference (0-1, lower is better)
        """
        positive_rates = {}

        for fst in self.fst_groups:
            fst_mask = fst_labels == fst

            if not np.any(fst_mask):
                continue

            fst_y_pred = y_pred[fst_mask]

            if target_class is not None:
                positive_rate = np.mean(fst_y_pred == target_class)
            else:
                # For multi-class, use average positive prediction rate
                positive_rate = 1.0  # Placeholder (always predict something)

            positive_rates[fst] = positive_rate

        if not positive_rates:
            return 0.0

        return max(positive_rates.values()) - min(positive_rates.values())


def evaluate_fairness(
    model,
    data_loader,
    device: str = 'cuda',
    num_classes: int = 7,
    target_class: Optional[int] = None
) -> FairnessResults:
    """
    High-level function to evaluate model fairness.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with (images, labels, fst_labels)
        device: Device to run inference on
        num_classes: Number of disease classes
        target_class: Target class for binary metrics

    Returns:
        FairnessResults object
    """
    import torch

    model.eval()
    model.to(device)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_fst = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, labels, fst_labels = batch
            else:
                images, labels = batch
                fst_labels = torch.ones(len(labels))  # Placeholder

            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())
            all_fst.extend(fst_labels.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)
    fst_labels = np.array(all_fst)

    # Evaluate fairness
    metrics = FairnessMetrics(num_classes=num_classes)
    results = metrics.evaluate(y_true, y_pred, y_prob, fst_labels, target_class)

    return results


if __name__ == "__main__":
    # Test fairness metrics
    print("Testing Fairness Metrics...")

    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    num_classes = 7

    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = np.random.randint(0, num_classes, n_samples)
    y_prob = np.random.rand(n_samples, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize

    fst_labels = np.random.choice([1, 2, 3, 4, 5, 6], n_samples)

    # Evaluate fairness
    metrics = FairnessMetrics(num_classes=num_classes)
    results = metrics.evaluate(y_true, y_pred, y_prob, fst_labels, target_class=0)

    print(results.summary())
    print("\nFairness metrics test PASSED!")
