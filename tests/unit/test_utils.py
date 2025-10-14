"""
Unit tests for utility functions.

Tests logging, configuration loading, checkpoint management, and other utilities.
"""

import pytest
import torch
import yaml
import json
from pathlib import Path


# ============================================================================
# CONFIGURATION LOADING TESTS
# ============================================================================

@pytest.mark.unit
class TestConfigurationLoading:
    """Test configuration file loading and parsing."""

    def test_load_yaml_config(self, temp_dir):
        """Test loading YAML configuration."""
        config_path = temp_dir / "config.yaml"

        config_data = {
            'model': {'architecture': 'resnet50', 'num_classes': 7},
            'training': {'epochs': 10, 'batch_size': 32}
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        # Load config
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config['model']['architecture'] == 'resnet50'
        assert loaded_config['training']['epochs'] == 10

    def test_load_json_config(self, temp_dir):
        """Test loading JSON configuration."""
        config_path = temp_dir / "config.json"

        config_data = {
            'model': {'architecture': 'resnet50'},
            'data': {'batch_size': 32}
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        assert loaded_config['model']['architecture'] == 'resnet50'

    def test_config_validation(self, mock_config):
        """Test configuration validation."""
        # Check required fields exist
        assert 'model' in mock_config
        assert 'data' in mock_config
        assert 'training' in mock_config

        # Check types
        assert isinstance(mock_config['model']['num_classes'], int)
        assert isinstance(mock_config['training']['learning_rate'], float)

    def test_config_merge(self):
        """Test merging configuration dictionaries."""
        base_config = {
            'model': {'architecture': 'resnet50', 'pretrained': True},
            'training': {'epochs': 10}
        }

        override_config = {
            'model': {'pretrained': False},
            'training': {'epochs': 20, 'lr': 0.001}
        }

        # Simple merge (override_config takes precedence)
        merged = {**base_config}
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        assert merged['model']['pretrained'] == False
        assert merged['training']['epochs'] == 20
        assert merged['training']['lr'] == 0.001


# ============================================================================
# CHECKPOINT MANAGEMENT TESTS
# ============================================================================

@pytest.mark.unit
class TestCheckpointManagement:
    """Test model checkpoint saving and loading utilities."""

    def test_save_checkpoint(self, temp_dir):
        """Test saving checkpoint with metadata."""
        checkpoint_path = temp_dir / "checkpoint.pth"

        checkpoint = {
            'epoch': 5,
            'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
            'optimizer_state_dict': {},
            'loss': 0.5,
            'metrics': {'accuracy': 0.85, 'auroc': 0.92}
        }

        torch.save(checkpoint, checkpoint_path)

        assert checkpoint_path.exists()

    def test_load_checkpoint(self, temp_dir):
        """Test loading checkpoint."""
        checkpoint_path = temp_dir / "checkpoint.pth"

        # Save
        checkpoint = {
            'epoch': 5,
            'loss': 0.5,
            'accuracy': 0.85
        }
        torch.save(checkpoint, checkpoint_path)

        # Load
        loaded = torch.load(checkpoint_path)

        assert loaded['epoch'] == 5
        assert loaded['loss'] == 0.5
        assert loaded['accuracy'] == 0.85

    def test_best_checkpoint_selection(self, temp_dir):
        """Test selecting best checkpoint based on metric."""
        # Create multiple checkpoints
        checkpoints = []
        for i in range(5):
            ckpt_path = temp_dir / f"checkpoint_epoch{i}.pth"
            checkpoint = {
                'epoch': i,
                'accuracy': 0.7 + i * 0.05  # Increasing accuracy
            }
            torch.save(checkpoint, ckpt_path)
            checkpoints.append(ckpt_path)

        # Find best checkpoint (highest accuracy)
        best_checkpoint = None
        best_accuracy = 0

        for ckpt_path in checkpoints:
            ckpt = torch.load(ckpt_path)
            if ckpt['accuracy'] > best_accuracy:
                best_accuracy = ckpt['accuracy']
                best_checkpoint = ckpt_path

        # Best should be last checkpoint (highest accuracy)
        assert best_checkpoint.name == "checkpoint_epoch4.pth"
        assert best_accuracy == 0.9

    def test_checkpoint_directory_creation(self, temp_dir):
        """Test creating checkpoint directory if it doesn't exist."""
        checkpoint_dir = temp_dir / "checkpoints" / "run1"

        # Create directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()


# ============================================================================
# LOGGING TESTS
# ============================================================================

@pytest.mark.unit
class TestLogging:
    """Test logging utilities."""

    def test_tensorboard_logging(self, mock_tensorboard_dir):
        """Test TensorBoard logging setup."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            pytest.skip("tensorboard not available")

        writer = SummaryWriter(log_dir=str(mock_tensorboard_dir))

        # Log scalar
        writer.add_scalar('loss/train', 0.5, 0)
        writer.add_scalar('accuracy/train', 0.85, 0)

        writer.close()

        # Check log directory exists
        assert mock_tensorboard_dir.exists()

    def test_metrics_logging_dict(self):
        """Test logging metrics in dictionary format."""
        metrics = {
            'epoch': 1,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'train_acc': 0.85,
            'val_acc': 0.82
        }

        # Check all metrics are numeric
        for key, value in metrics.items():
            if key != 'epoch':
                assert isinstance(value, (int, float))

    def test_csv_logging(self, temp_dir):
        """Test logging metrics to CSV file."""
        import csv

        log_path = temp_dir / "training_log.csv"

        # Write logs
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'accuracy'])
            writer.writeheader()
            for i in range(5):
                writer.writerow({'epoch': i, 'loss': 0.5 - i*0.05, 'accuracy': 0.7 + i*0.05})

        # Read and verify
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5
        assert float(rows[0]['loss']) == 0.5
        assert float(rows[4]['accuracy']) == 0.9


# ============================================================================
# RANDOM SEED TESTS
# ============================================================================

@pytest.mark.unit
class TestRandomSeed:
    """Test random seed setting for reproducibility."""

    def test_set_torch_seed(self, random_seed):
        """Test setting PyTorch random seed."""
        torch.manual_seed(random_seed)

        a = torch.randn(10)

        torch.manual_seed(random_seed)
        b = torch.randn(10)

        assert torch.allclose(a, b)

    def test_set_numpy_seed(self, random_seed):
        """Test setting NumPy random seed."""
        import numpy as np

        np.random.seed(random_seed)
        a = np.random.randn(10)

        np.random.seed(random_seed)
        b = np.random.randn(10)

        assert np.allclose(a, b)

    def test_reproducible_data_split(self, random_seed):
        """Test reproducible data splitting with seed."""
        from sklearn.model_selection import train_test_split
        import numpy as np

        data = np.arange(100)
        labels = np.random.randint(0, 2, 100)

        # Split 1
        train1, test1 = train_test_split(
            data, test_size=0.2, random_state=random_seed, stratify=labels
        )

        # Split 2 (should be identical)
        train2, test2 = train_test_split(
            data, test_size=0.2, random_state=random_seed, stratify=labels
        )

        assert np.array_equal(train1, train2)
        assert np.array_equal(test1, test2)


# ============================================================================
# FILE I/O TESTS
# ============================================================================

@pytest.mark.unit
class TestFileIO:
    """Test file input/output utilities."""

    def test_create_directory_if_not_exists(self, temp_dir):
        """Test creating directory if it doesn't exist."""
        new_dir = temp_dir / "experiments" / "run1"

        new_dir.mkdir(parents=True, exist_ok=True)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_save_predictions_to_file(self, temp_dir):
        """Test saving predictions to file."""
        predictions = torch.rand(100, 7)
        labels = torch.randint(0, 7, (100,))

        pred_path = temp_dir / "predictions.pt"
        torch.save({'predictions': predictions, 'labels': labels}, pred_path)

        # Load and verify
        loaded = torch.load(pred_path)
        assert torch.allclose(loaded['predictions'], predictions)
        assert torch.equal(loaded['labels'], labels)

    def test_list_checkpoint_files(self, temp_dir):
        """Test listing checkpoint files in directory."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Create some checkpoints
        for i in range(3):
            ckpt_path = checkpoint_dir / f"model_epoch{i}.pth"
            torch.save({'epoch': i}, ckpt_path)

        # List checkpoint files
        checkpoints = sorted(checkpoint_dir.glob("*.pth"))

        assert len(checkpoints) == 3
        assert checkpoints[0].name == "model_epoch0.pth"


# ============================================================================
# DATA VALIDATION TESTS
# ============================================================================

@pytest.mark.unit
class TestDataValidation:
    """Test data validation utilities."""

    def test_check_tensor_shape(self):
        """Test tensor shape validation."""
        tensor = torch.randn(8, 3, 224, 224)

        # Valid shape
        assert tensor.shape == (8, 3, 224, 224)
        assert tensor.ndim == 4

    def test_check_label_range(self):
        """Test label range validation."""
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        # Check all labels in valid range
        assert torch.all((labels >= 0) & (labels < 7))

    def test_check_probability_distribution(self):
        """Test probability distribution validation."""
        probs = torch.softmax(torch.randn(10, 7), dim=1)

        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-6)

        # Check all probabilities in [0, 1]
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_check_nan_inf(self):
        """Test checking for NaN and Inf values."""
        # Valid tensor
        valid = torch.randn(10, 10)
        assert torch.isfinite(valid).all()

        # Tensor with NaN
        with_nan = torch.randn(10, 10)
        with_nan[0, 0] = float('nan')
        assert not torch.isfinite(with_nan).all()

        # Tensor with Inf
        with_inf = torch.randn(10, 10)
        with_inf[0, 0] = float('inf')
        assert not torch.isfinite(with_inf).all()


# ============================================================================
# METRIC COMPUTATION TESTS
# ============================================================================

@pytest.mark.unit
class TestMetricComputation:
    """Test metric computation utilities."""

    def test_compute_accuracy(self):
        """Test accuracy computation."""
        predictions = torch.tensor([0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 1, 1])

        accuracy = (predictions == labels).float().mean()

        assert accuracy == 0.8  # 4 out of 5 correct

    def test_compute_top_k_accuracy(self):
        """Test top-k accuracy computation."""
        logits = torch.tensor([
            [10, 2, 1, 0, 0, 0, 0],  # Top-3: [0, 1, 2]
            [1, 10, 2, 0, 0, 0, 0],  # Top-3: [1, 2, 0]
            [1, 2, 10, 0, 0, 0, 0],  # Top-3: [2, 1, 0]
        ])
        labels = torch.tensor([0, 2, 2])  # Correct: [0], [2], [2]

        # Top-1 accuracy: 2/3
        _, top1_pred = logits.max(dim=1)
        top1_acc = (top1_pred == labels).float().mean()
        assert top1_acc == 2/3

        # Top-3 accuracy: 3/3 (all correct labels in top-3)
        _, top3_pred = logits.topk(3, dim=1)
        top3_acc = sum((labels[i] in top3_pred[i]) for i in range(len(labels))) / len(labels)
        assert top3_acc == 1.0

    def test_compute_class_wise_accuracy(self):
        """Test class-wise accuracy computation."""
        predictions = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        labels = torch.tensor([0, 0, 1, 2, 2, 1, 1, 1])

        # Class 0: 2/2 = 1.0
        # Class 1: 2/4 = 0.5
        # Class 2: 1/2 = 0.5

        from sklearn.metrics import classification_report
        report = classification_report(labels, predictions, output_dict=True)

        assert report['0']['precision'] == 2/3  # 2 correct out of 3 predicted as 0
        assert report['1']['recall'] == 2/4  # 2 correct out of 4 actual class 1


# ============================================================================
# VISUALIZATION HELPER TESTS
# ============================================================================

@pytest.mark.unit
class TestVisualizationHelpers:
    """Test visualization utility functions."""

    def test_unnormalize_image(self, imagenet_mean_std):
        """Test unnormalizing image for visualization."""
        # Normalized image
        normalized = torch.randn(3, 224, 224)

        # Unnormalize
        mean = torch.tensor(imagenet_mean_std['mean']).view(3, 1, 1)
        std = torch.tensor(imagenet_mean_std['std']).view(3, 1, 1)
        unnormalized = normalized * std + mean

        # Check shape preserved
        assert unnormalized.shape == normalized.shape

    def test_tensor_to_numpy_image(self):
        """Test converting tensor to numpy for visualization."""
        tensor_img = torch.rand(3, 224, 224)

        # Convert to HWC format for matplotlib
        numpy_img = tensor_img.permute(1, 2, 0).numpy()

        assert numpy_img.shape == (224, 224, 3)

    def test_batch_to_grid(self):
        """Test converting batch of images to grid."""
        try:
            from torchvision.utils import make_grid
        except ImportError:
            pytest.skip("torchvision not available")

        batch = torch.rand(16, 3, 64, 64)
        grid = make_grid(batch, nrow=4)

        # Grid should have correct shape
        assert grid.shape[0] == 3  # RGB channels
        assert grid.shape[1] > 64  # Height > single image
        assert grid.shape[2] > 64  # Width > single image


# ============================================================================
# TIMER UTILITY TESTS
# ============================================================================

@pytest.mark.unit
class TestTimerUtility:
    """Test timer utility for performance tracking."""

    def test_time_function_execution(self):
        """Test timing function execution."""
        import time

        start = time.time()
        time.sleep(0.1)  # Sleep for 100ms
        elapsed = time.time() - start

        assert elapsed >= 0.1
        assert elapsed < 0.2  # Should be close to 0.1s

    def test_context_manager_timer(self):
        """Test context manager for timing."""
        import time

        class Timer:
            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                self.end = time.time()
                self.elapsed = self.end - self.start

        with Timer() as t:
            time.sleep(0.05)

        assert t.elapsed >= 0.05
        assert t.elapsed < 0.1
