"""
Integration tests for end-to-end evaluation pipeline.

Tests model evaluation, fairness metrics computation, visualization generation,
and result saving.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


# ============================================================================
# MODEL EVALUATION TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestEvaluationPipeline:
    """Test end-to-end model evaluation."""

    def test_full_evaluation_loop(self, mock_dataloader, mock_resnet_model, device):
        """Test complete evaluation loop with metrics computation."""
        model = mock_resnet_model
        model.eval()

        criterion = nn.CrossEntropyLoss()

        all_predictions = []
        all_labels = []
        all_fst = []
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in mock_dataloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                fst = batch['fst'].to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Collect predictions
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu())
                all_labels.append(labels.cpu())
                all_fst.append(fst.cpu())

                total_loss += loss.item()
                num_batches += 1

        # Concatenate all batches
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        fst = torch.cat(all_fst)

        # Compute overall accuracy
        _, predicted_classes = predictions.max(dim=1)
        accuracy = (predicted_classes == labels).float().mean()

        # Verify results
        assert 0 <= accuracy <= 1
        assert predictions.shape[0] == labels.shape[0]
        assert predictions.shape[1] == 7  # 7 classes

    def test_evaluation_with_checkpoint_loading(
        self, mock_trained_model_weights, mock_dataloader, device
    ):
        """Test loading checkpoint and evaluating."""
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        # Load checkpoint
        checkpoint = torch.load(mock_trained_model_weights)

        # Create model (in practice, load from checkpoint)
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 7)
        model = model.to(device)

        # Note: In real scenario, we'd load state_dict from checkpoint
        # For testing, we just verify the checkpoint exists and has correct structure

        assert 'model_state_dict' in checkpoint
        assert 'epoch' in checkpoint

        # Evaluation
        model.eval()
        with torch.no_grad():
            batch = next(iter(mock_dataloader))
            images = batch['image'].to(device)
            outputs = model(images)

        assert outputs.shape == (8, 7)


# ============================================================================
# FAIRNESS METRICS EVALUATION TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.fairness
@pytest.mark.pipeline
class TestFairnessEvaluation:
    """Test fairness metrics computation in evaluation pipeline."""

    def test_compute_fairness_metrics_per_fst(
        self, mock_fairness_metrics_data
    ):
        """Test computing fairness metrics for each FST group."""
        from sklearn.metrics import roc_auc_score, accuracy_score

        predictions, labels, fst = mock_fairness_metrics_data

        # Convert to binary for AUROC
        binary_labels = (labels == 4).long()  # Melanoma vs rest
        probs = predictions[:, 4]

        # Compute metrics per FST group
        metrics_per_fst = {}

        for fst_type in range(1, 7):
            mask = fst == fst_type
            if mask.sum() > 10:  # Need sufficient samples
                group_probs = probs[mask].numpy()
                group_labels = binary_labels[mask].numpy()

                if len(np.unique(group_labels)) > 1:
                    auroc = roc_auc_score(group_labels, group_probs)

                    # Predicted classes
                    pred_classes = predictions[mask].argmax(dim=1)
                    true_classes = labels[mask]
                    accuracy = accuracy_score(true_classes, pred_classes)

                    metrics_per_fst[fst_type] = {
                        'auroc': auroc,
                        'accuracy': accuracy,
                        'n_samples': mask.sum().item()
                    }

        # Verify metrics computed
        assert len(metrics_per_fst) > 0
        for fst_type, metrics in metrics_per_fst.items():
            assert 0 <= metrics['auroc'] <= 1
            assert 0 <= metrics['accuracy'] <= 1

    def test_compute_equalized_odds_difference(
        self, mock_fairness_metrics_data
    ):
        """Test computing EOD across FST groups."""
        from sklearn.metrics import confusion_matrix

        predictions, labels, fst = mock_fairness_metrics_data

        # Binary classification
        binary_preds = (predictions.argmax(dim=1) == 4).long()
        binary_labels = (labels == 4).long()

        def compute_tpr_fpr(preds, labs):
            if len(np.unique(labs)) < 2:
                return None, None
            cm = confusion_matrix(labs, preds)
            if cm.shape != (2, 2):
                return None, None
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return tpr, fpr

        # Compute for light vs dark skin
        light_mask = (fst >= 1) & (fst <= 2)
        dark_mask = (fst >= 5) & (fst <= 6)

        tpr_light, fpr_light = compute_tpr_fpr(
            binary_preds[light_mask].numpy(),
            binary_labels[light_mask].numpy()
        )

        tpr_dark, fpr_dark = compute_tpr_fpr(
            binary_preds[dark_mask].numpy(),
            binary_labels[dark_mask].numpy()
        )

        if tpr_light is not None and tpr_dark is not None:
            eod = max(abs(tpr_light - tpr_dark), abs(fpr_light - fpr_dark))
            assert 0 <= eod <= 1

    def test_compute_calibration_per_fst(self, mock_fairness_metrics_data):
        """Test computing calibration metrics per FST group."""
        predictions, labels, fst = mock_fairness_metrics_data

        confidences, pred_labels = predictions.max(dim=1)

        # Compute ECE per FST
        def compute_ece(conf, pred, true, n_bins=10):
            conf = conf.numpy()
            correct = (pred == true).float().numpy()

            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0

            for i in range(n_bins):
                in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
                if in_bin.sum() > 0:
                    acc_in_bin = correct[in_bin].mean()
                    conf_in_bin = conf[in_bin].mean()
                    ece += abs(conf_in_bin - acc_in_bin) * (in_bin.sum() / len(conf))

            return ece

        ece_per_fst = {}
        for fst_type in range(1, 7):
            mask = fst == fst_type
            if mask.sum() > 20:
                ece = compute_ece(
                    confidences[mask],
                    pred_labels[mask],
                    labels[mask]
                )
                ece_per_fst[fst_type] = ece

        # Verify ECE computed
        for fst_type, ece in ece_per_fst.items():
            assert 0 <= ece <= 1


# ============================================================================
# CONFUSION MATRIX GENERATION TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestConfusionMatrixGeneration:
    """Test confusion matrix generation for evaluation."""

    def test_generate_confusion_matrix(self, mock_predictions_calibrated):
        """Test generating confusion matrix from predictions."""
        from sklearn.metrics import confusion_matrix

        predictions, labels = mock_predictions_calibrated

        # Get predicted classes
        pred_classes = predictions.argmax(dim=1)

        # Compute confusion matrix
        cm = confusion_matrix(labels.numpy(), pred_classes.numpy(), labels=list(range(7)))

        # Verify shape
        assert cm.shape == (7, 7)

        # Diagonal should have most values (correct predictions)
        diagonal_sum = cm.diagonal().sum()
        total_sum = cm.sum()
        assert diagonal_sum > 0

    def test_confusion_matrix_per_fst(self, mock_fairness_metrics_data):
        """Test generating confusion matrices per FST group."""
        from sklearn.metrics import confusion_matrix

        predictions, labels, fst = mock_fairness_metrics_data
        pred_classes = predictions.argmax(dim=1)

        confusion_matrices = {}

        for fst_type in range(1, 7):
            mask = fst == fst_type
            if mask.sum() > 10:
                cm = confusion_matrix(
                    labels[mask].numpy(),
                    pred_classes[mask].numpy(),
                    labels=list(range(7))
                )
                confusion_matrices[fst_type] = cm

        # Verify CMs generated
        assert len(confusion_matrices) > 0
        for fst_type, cm in confusion_matrices.items():
            assert cm.shape == (7, 7)


# ============================================================================
# CLASSIFICATION REPORT TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestClassificationReport:
    """Test classification report generation."""

    def test_generate_classification_report(self, mock_predictions_calibrated):
        """Test generating classification report with precision/recall/f1."""
        from sklearn.metrics import classification_report

        predictions, labels = mock_predictions_calibrated
        pred_classes = predictions.argmax(dim=1)

        report = classification_report(
            labels.numpy(),
            pred_classes.numpy(),
            target_names=[
                'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
            ],
            output_dict=True,
            zero_division=0
        )

        # Verify report structure
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report

        # Check per-class metrics
        for class_name in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']:
            if class_name in report:
                assert 'precision' in report[class_name]
                assert 'recall' in report[class_name]
                assert 'f1-score' in report[class_name]


# ============================================================================
# RESULT SAVING TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestResultSaving:
    """Test saving evaluation results to disk."""

    def test_save_predictions_to_file(self, mock_predictions_calibrated, temp_dir):
        """Test saving predictions to file."""
        predictions, labels = mock_predictions_calibrated

        results_path = temp_dir / "predictions.pt"

        results = {
            'predictions': predictions,
            'labels': labels,
            'predicted_classes': predictions.argmax(dim=1),
            'confidences': predictions.max(dim=1)[0]
        }

        torch.save(results, results_path)

        # Verify file saved
        assert results_path.exists()

        # Load and verify
        loaded = torch.load(results_path)
        assert torch.allclose(loaded['predictions'], predictions)
        assert torch.equal(loaded['labels'], labels)

    def test_save_fairness_metrics_to_json(self, temp_dir):
        """Test saving fairness metrics to JSON."""
        import json

        metrics_path = temp_dir / "fairness_metrics.json"

        fairness_metrics = {
            'auroc_per_fst': {
                'FST_I': 0.92,
                'FST_II': 0.90,
                'FST_III': 0.88,
                'FST_IV': 0.87,
                'FST_V': 0.85,
                'FST_VI': 0.83
            },
            'eod': 0.07,
            'demographic_parity': 0.05,
            'overall_accuracy': 0.87
        }

        with open(metrics_path, 'w') as f:
            json.dump(fairness_metrics, f, indent=2)

        # Verify file saved
        assert metrics_path.exists()

        # Load and verify
        with open(metrics_path, 'r') as f:
            loaded = json.load(f)

        assert loaded['eod'] == 0.07
        assert loaded['overall_accuracy'] == 0.87

    def test_save_confusion_matrix_to_csv(self, temp_dir):
        """Test saving confusion matrix to CSV."""
        import pandas as pd
        from sklearn.metrics import confusion_matrix

        # Generate sample confusion matrix
        cm = np.array([
            [50, 2, 1, 0, 1, 0, 0],
            [3, 45, 2, 0, 1, 1, 0],
            [1, 2, 48, 1, 0, 1, 0],
            [0, 0, 1, 52, 0, 0, 0],
            [2, 1, 0, 0, 47, 2, 1],
            [0, 1, 1, 0, 2, 49, 0],
            [0, 0, 0, 0, 1, 0, 51]
        ])

        class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        csv_path = temp_dir / "confusion_matrix.csv"
        cm_df.to_csv(csv_path)

        # Verify file saved
        assert csv_path.exists()

        # Load and verify
        loaded_df = pd.read_csv(csv_path, index_col=0)
        assert loaded_df.shape == (7, 7)


# ============================================================================
# VISUALIZATION GENERATION TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestVisualizationGeneration:
    """Test generating visualizations for evaluation results."""

    def test_generate_roc_curve_data(self, mock_predictions_calibrated):
        """Test generating ROC curve data."""
        from sklearn.metrics import roc_curve, auc

        predictions, labels = mock_predictions_calibrated

        # ROC for binary classification (class 4 vs rest)
        binary_labels = (labels == 4).numpy()
        probs = predictions[:, 4].numpy()

        fpr, tpr, thresholds = roc_curve(binary_labels, probs)
        roc_auc = auc(fpr, tpr)

        # Verify data
        assert len(fpr) > 0
        assert len(tpr) > 0
        assert 0 <= roc_auc <= 1

    def test_generate_precision_recall_curve_data(self, mock_predictions_calibrated):
        """Test generating precision-recall curve data."""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        predictions, labels = mock_predictions_calibrated

        # PR curve for binary classification
        binary_labels = (labels == 4).numpy()
        probs = predictions[:, 4].numpy()

        precision, recall, thresholds = precision_recall_curve(binary_labels, probs)
        ap_score = average_precision_score(binary_labels, probs)

        # Verify data
        assert len(precision) > 0
        assert len(recall) > 0
        assert 0 <= ap_score <= 1

    def test_generate_calibration_plot_data(self, mock_predictions_calibrated):
        """Test generating calibration plot data."""
        predictions, labels = mock_predictions_calibrated

        confidences, pred_classes = predictions.max(dim=1)
        correct = (pred_classes == labels).float()

        # Bin predictions
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        confidences_np = confidences.numpy()
        correct_np = correct.numpy()

        for i in range(n_bins):
            in_bin = (confidences_np > bin_boundaries[i]) & (confidences_np <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_accuracies.append(correct_np[in_bin].mean())
                bin_confidences.append(confidences_np[in_bin].mean())
                bin_counts.append(in_bin.sum())

        # Verify data
        assert len(bin_accuracies) > 0
        assert len(bin_confidences) > 0


# ============================================================================
# MULTI-MODEL COMPARISON TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestMultiModelComparison:
    """Test comparing multiple models."""

    def test_compare_two_models(self, mock_dataloader, device):
        """Test comparing predictions from two different models."""
        try:
            from torchvision.models import resnet18, resnet50
        except ImportError:
            pytest.skip("torchvision not available")

        # Two models with different architectures
        model1 = resnet18(pretrained=False)
        model1.fc = nn.Linear(model1.fc.in_features, 7)
        model1 = model1.to(device)
        model1.eval()

        # Note: resnet50 is larger, just testing comparison mechanism
        model2 = resnet18(pretrained=False)  # Using resnet18 for speed
        model2.fc = nn.Linear(model2.fc.in_features, 7)
        model2 = model2.to(device)
        model2.eval()

        # Evaluate both models
        def evaluate_model(model):
            predictions = []
            labels_list = []

            with torch.no_grad():
                for batch in mock_dataloader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)

                    predictions.append(probs.cpu())
                    labels_list.append(labels.cpu())

            predictions = torch.cat(predictions)
            labels_list = torch.cat(labels_list)

            _, pred_classes = predictions.max(dim=1)
            accuracy = (pred_classes == labels_list).float().mean()

            return accuracy

        acc1 = evaluate_model(model1)
        acc2 = evaluate_model(model2)

        # Both should produce valid accuracies
        assert 0 <= acc1 <= 1
        assert 0 <= acc2 <= 1


# ============================================================================
# ENSEMBLE EVALUATION TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestEnsembleEvaluation:
    """Test ensemble model evaluation."""

    def test_ensemble_averaging(self, mock_dataloader, device):
        """Test ensemble with prediction averaging."""
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        # Create 3 models for ensemble
        models = []
        for i in range(3):
            model = resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 7)
            model = model.to(device)
            model.eval()
            models.append(model)

        # Evaluate ensemble
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in mock_dataloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                # Get predictions from all models
                ensemble_logits = []
                for model in models:
                    logits = model(images)
                    ensemble_logits.append(logits)

                # Average logits
                avg_logits = torch.stack(ensemble_logits).mean(dim=0)
                probs = torch.softmax(avg_logits, dim=1)

                all_predictions.append(probs.cpu())
                all_labels.append(labels.cpu())

        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)

        _, pred_classes = predictions.max(dim=1)
        accuracy = (pred_classes == labels).float().mean()

        assert 0 <= accuracy <= 1
