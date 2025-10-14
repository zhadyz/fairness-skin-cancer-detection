"""
Unit tests for fairness metrics computation.

Tests AUROC per FST, Equalized Odds Difference (EOD), Expected Calibration Error (ECE),
and sensitivity/specificity per demographic group.
"""

import pytest
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


# ============================================================================
# AUROC PER FST TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.fairness
class TestAUROCPerFST:
    """Test AUROC computation for different FST groups."""

    def test_auroc_perfect_classifier(self):
        """Test AUROC = 1.0 for perfect classifier."""
        # Perfect predictions: high confidence for correct class
        predictions = torch.tensor([
            [0.99, 0.01],
            [0.99, 0.01],
            [0.01, 0.99],
            [0.01, 0.99],
        ])
        labels = torch.tensor([0, 0, 1, 1])

        # Compute AUROC
        probs = predictions[:, 1].numpy()  # Probability of positive class
        auroc = roc_auc_score(labels.numpy(), probs)

        assert auroc == 1.0

    def test_auroc_random_classifier(self):
        """Test AUROC ≈ 0.5 for random classifier."""
        np.random.seed(42)

        n_samples = 1000
        predictions = np.random.rand(n_samples)
        labels = np.random.randint(0, 2, n_samples)

        auroc = roc_auc_score(labels, predictions)

        # Should be close to 0.5 (random guessing)
        assert 0.45 <= auroc <= 0.55

    def test_auroc_per_fst_group(self, mock_fairness_metrics_data):
        """Test AUROC computation for each FST group."""
        predictions, labels, fst = mock_fairness_metrics_data

        # Convert to binary classification (e.g., melanoma vs rest)
        binary_labels = (labels == 4).long()  # Class 4 = melanoma
        probs = predictions[:, 4]  # Probability of melanoma

        auroc_per_fst = {}

        for fst_type in range(1, 7):
            mask = fst == fst_type
            if mask.sum() > 0:
                group_probs = probs[mask].numpy()
                group_labels = binary_labels[mask].numpy()

                # Need at least one positive and one negative sample
                if len(np.unique(group_labels)) > 1:
                    auroc = roc_auc_score(group_labels, group_probs)
                    auroc_per_fst[fst_type] = auroc

                    # AUROC should be in valid range
                    assert 0.0 <= auroc <= 1.0

    def test_auroc_multiclass(self):
        """Test AUROC for multiclass classification (one-vs-rest)."""
        from sklearn.metrics import roc_auc_score

        n_samples = 200
        n_classes = 7

        predictions = torch.softmax(torch.randn(n_samples, n_classes), dim=1)
        labels = torch.randint(0, n_classes, (n_samples,))

        # One-vs-rest AUROC
        auroc_ovr = roc_auc_score(
            labels.numpy(),
            predictions.numpy(),
            multi_class='ovr',
            average='macro'
        )

        assert 0.0 <= auroc_ovr <= 1.0


# ============================================================================
# EQUALIZED ODDS DIFFERENCE (EOD) TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.fairness
class TestEqualizedOddsDifference:
    """Test Equalized Odds Difference metric."""

    def test_eod_perfect_fairness(self):
        """Test EOD = 0 for perfectly fair classifier."""
        # Same TPR and FPR across groups

        # Group 1: 100% TPR, 0% FPR
        pred_g1 = torch.tensor([1, 1, 0, 0])
        labels_g1 = torch.tensor([1, 1, 0, 0])

        # Group 2: 100% TPR, 0% FPR (same as group 1)
        pred_g2 = torch.tensor([1, 1, 0, 0])
        labels_g2 = torch.tensor([1, 1, 0, 0])

        # Compute TPR and FPR for each group
        def compute_tpr_fpr(pred, labels):
            tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return tpr, fpr

        tpr1, fpr1 = compute_tpr_fpr(pred_g1, labels_g1)
        tpr2, fpr2 = compute_tpr_fpr(pred_g2, labels_g2)

        # EOD = max(|TPR1 - TPR2|, |FPR1 - FPR2|)
        eod = max(abs(tpr1 - tpr2), abs(fpr1 - fpr2))

        assert eod == 0.0

    def test_eod_maximum_disparity(self):
        """Test EOD = 1 for maximum disparity."""
        # Group 1: 100% TPR, 0% FPR
        pred_g1 = torch.tensor([1, 1, 0, 0])
        labels_g1 = torch.tensor([1, 1, 0, 0])

        # Group 2: 0% TPR, 100% FPR (completely wrong)
        pred_g2 = torch.tensor([0, 0, 1, 1])
        labels_g2 = torch.tensor([1, 1, 0, 0])

        def compute_tpr_fpr(pred, labels):
            tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return tpr, fpr

        tpr1, fpr1 = compute_tpr_fpr(pred_g1, labels_g1)
        tpr2, fpr2 = compute_tpr_fpr(pred_g2, labels_g2)

        eod = max(abs(tpr1 - tpr2), abs(fpr1 - fpr2))

        assert eod == 1.0

    def test_eod_across_fst_groups(self, mock_fairness_metrics_data):
        """Test EOD computation across FST groups."""
        predictions, labels, fst = mock_fairness_metrics_data

        # Binary classification
        binary_preds = (predictions.argmax(dim=1) == 4).long()
        binary_labels = (labels == 4).long()

        # Compute TPR/FPR for light vs dark skin
        light_mask = (fst >= 1) & (fst <= 2)
        dark_mask = (fst >= 5) & (fst <= 6)

        def compute_rates(mask):
            preds = binary_preds[mask]
            labs = binary_labels[mask]
            if len(labs) == 0 or len(np.unique(labs)) < 2:
                return None, None
            tn, fp, fn, tp = confusion_matrix(labs, preds).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return tpr, fpr

        tpr_light, fpr_light = compute_rates(light_mask)
        tpr_dark, fpr_dark = compute_rates(dark_mask)

        if tpr_light is not None and tpr_dark is not None:
            eod = max(abs(tpr_light - tpr_dark), abs(fpr_light - fpr_dark))
            assert 0.0 <= eod <= 1.0


# ============================================================================
# EXPECTED CALIBRATION ERROR (ECE) TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.fairness
class TestExpectedCalibrationError:
    """Test Expected Calibration Error computation."""

    def test_ece_perfectly_calibrated(self):
        """Test ECE ≈ 0 for perfectly calibrated predictions."""
        # Predictions that match actual frequencies
        n_bins = 10
        n_samples_per_bin = 100

        predictions = []
        labels = []

        for i in range(n_bins):
            confidence = (i + 0.5) / n_bins  # Center of bin
            n_positive = int(confidence * n_samples_per_bin)

            # Create samples where accuracy matches confidence
            bin_preds = torch.full((n_samples_per_bin,), confidence)
            bin_labels = torch.cat([
                torch.ones(n_positive),
                torch.zeros(n_samples_per_bin - n_positive)
            ])

            predictions.append(bin_preds)
            labels.append(bin_labels)

        predictions = torch.cat(predictions)
        labels = torch.cat(labels)

        # Compute ECE
        ece = compute_ece(predictions, labels, n_bins=n_bins)

        # Should be very close to 0 (within binning error)
        assert ece < 0.05

    def test_ece_overconfident(self):
        """Test ECE > 0 for overconfident predictions."""
        # High confidence but poor accuracy
        predictions = torch.full((100,), 0.9)  # 90% confidence
        labels = torch.zeros(100)  # All negative (0% accuracy for positive class)
        labels[:40] = 1  # Only 40% are actually positive

        ece = compute_ece(predictions, labels, n_bins=10)

        # ECE should be high (overconfident)
        assert ece > 0.3

    def test_ece_underconfident(self):
        """Test ECE for underconfident predictions."""
        # Low confidence but high accuracy
        predictions = torch.full((100,), 0.4)  # 40% confidence
        labels = torch.ones(100)  # All positive
        labels[:10] = 0  # 90% are actually positive

        ece = compute_ece(predictions, labels, n_bins=10)

        # ECE should be significant (underconfident)
        assert ece > 0.3

    def test_ece_per_fst_group(self, mock_fairness_metrics_data):
        """Test ECE computation for each FST group."""
        predictions, labels, fst = mock_fairness_metrics_data

        # Use max probability as confidence
        confidences, pred_labels = predictions.max(dim=1)

        ece_per_fst = {}

        for fst_type in range(1, 7):
            mask = fst == fst_type
            if mask.sum() > 10:  # Need sufficient samples
                group_conf = confidences[mask]
                group_pred = pred_labels[mask]
                group_label = labels[mask]

                # Binary: correct vs incorrect
                correct = (group_pred == group_label).float()

                ece = compute_ece(group_conf, correct, n_bins=10)
                ece_per_fst[fst_type] = ece

                assert 0.0 <= ece <= 1.0


def compute_ece(confidences, labels, n_bins=10):
    """
    Compute Expected Calibration Error.

    Args:
        confidences: Model confidence scores [0, 1]
        labels: Binary ground truth labels (or correctness)
        n_bins: Number of bins for calibration

    Returns:
        ECE value
    """
    confidences = confidences.numpy() if torch.is_tensor(confidences) else confidences
    labels = labels.numpy() if torch.is_tensor(labels) else labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# ============================================================================
# SENSITIVITY / SPECIFICITY TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.fairness
class TestSensitivitySpecificity:
    """Test sensitivity and specificity per FST group."""

    def test_sensitivity_perfect(self):
        """Test sensitivity = 1.0 for perfect recall."""
        predictions = torch.tensor([1, 1, 1, 1])
        labels = torch.tensor([1, 1, 1, 1])

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        sensitivity = tp / (tp + fn)

        assert sensitivity == 1.0

    def test_specificity_perfect(self):
        """Test specificity = 1.0 for perfect negative recall."""
        predictions = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 0, 0, 0])

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp)

        assert specificity == 1.0

    def test_sensitivity_specificity_tradeoff(self):
        """Test sensitivity-specificity tradeoff."""
        # High sensitivity, low specificity (predicts mostly positive)
        predictions_high_sens = torch.tensor([1, 1, 1, 1, 1, 1])
        labels = torch.tensor([1, 1, 0, 0, 0, 0])

        tn, fp, fn, tp = confusion_matrix(labels, predictions_high_sens).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        assert sensitivity == 1.0  # Caught all positives
        assert specificity == 0.0  # Missed all negatives

    def test_sensitivity_specificity_per_fst(self, mock_fairness_metrics_data):
        """Test sensitivity and specificity across FST groups."""
        predictions, labels, fst = mock_fairness_metrics_data

        # Binary classification
        binary_preds = (predictions.argmax(dim=1) == 4).long()
        binary_labels = (labels == 4).long()

        results = {}

        for fst_type in range(1, 7):
            mask = fst == fst_type
            if mask.sum() > 10:
                group_preds = binary_preds[mask].numpy()
                group_labels = binary_labels[mask].numpy()

                if len(np.unique(group_labels)) > 1:
                    tn, fp, fn, tp = confusion_matrix(group_labels, group_preds).ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                    results[fst_type] = {
                        'sensitivity': sensitivity,
                        'specificity': specificity
                    }

                    assert 0.0 <= sensitivity <= 1.0
                    assert 0.0 <= specificity <= 1.0


# ============================================================================
# DEMOGRAPHIC PARITY TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.fairness
class TestDemographicParity:
    """Test demographic parity metric."""

    def test_demographic_parity_perfect(self):
        """Test demographic parity = 0 when positive rates are equal."""
        # Group 1: 50% positive predictions
        pred_g1 = torch.tensor([1, 1, 0, 0])

        # Group 2: 50% positive predictions
        pred_g2 = torch.tensor([1, 1, 0, 0])

        ppr1 = pred_g1.float().mean()
        ppr2 = pred_g2.float().mean()

        dp = abs(ppr1 - ppr2)
        assert dp == 0.0

    def test_demographic_parity_disparity(self):
        """Test demographic parity with disparity."""
        # Group 1: 75% positive predictions
        pred_g1 = torch.tensor([1, 1, 1, 0])

        # Group 2: 25% positive predictions
        pred_g2 = torch.tensor([1, 0, 0, 0])

        ppr1 = pred_g1.float().mean()
        ppr2 = pred_g2.float().mean()

        dp = abs(ppr1 - ppr2)
        assert dp == 0.5


# ============================================================================
# CONFUSION MATRIX TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.fairness
class TestConfusionMatrixMetrics:
    """Test confusion matrix computation for fairness analysis."""

    def test_confusion_matrix_binary(self):
        """Test confusion matrix for binary classification."""
        predictions = torch.tensor([1, 0, 1, 0, 1, 1])
        labels = torch.tensor([1, 0, 0, 0, 1, 1])

        cm = confusion_matrix(labels, predictions)

        # TN=2, FP=1, FN=0, TP=3
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[1, 1] == 3  # TP

    def test_confusion_matrix_multiclass(self):
        """Test confusion matrix for multiclass classification."""
        predictions = torch.tensor([0, 1, 2, 0, 1, 2])
        labels = torch.tensor([0, 1, 2, 1, 2, 0])

        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])

        # Check diagonal (correct predictions)
        assert cm[0, 0] == 1  # Class 0 correct
        assert cm[1, 1] == 1  # Class 1 correct
        assert cm[2, 2] == 1  # Class 2 correct
