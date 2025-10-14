"""
Shared pytest fixtures for testing skin cancer detection system.

This module provides reusable fixtures for datasets, models, metrics,
and other testing utilities.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import tempfile
import shutil

from tests.fixtures.sample_data import (
    MockHAM10000,
    MockMedNodeDataset,
    create_mock_dataloader,
    generate_mock_predictions,
    generate_mock_fst_stratified_data
)


# ============================================================================
# SESSION-LEVEL FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing (CPU or GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp(prefix="test_skin_cancer_")
    yield Path(temp_path)
    # Cleanup after all tests
    shutil.rmtree(temp_path, ignore_errors=True)


# ============================================================================
# DATASET FIXTURES
# ============================================================================

@pytest.fixture
def mock_ham10000_dataset(random_seed):
    """Create a small mock HAM10000 dataset for testing."""
    return MockHAM10000(
        num_samples=100,
        num_classes=7,
        image_size=224,
        include_fst=True,
        balanced=True,
        seed=random_seed
    )


@pytest.fixture
def mock_ham10000_small(random_seed):
    """Create a very small mock HAM10000 dataset (10 samples) for quick tests."""
    return MockHAM10000(
        num_samples=10,
        num_classes=7,
        image_size=224,
        include_fst=True,
        balanced=True,
        seed=random_seed
    )


@pytest.fixture
def mock_mednode_dataset(random_seed):
    """Create a mock MedNode dataset for multi-source testing."""
    return MockMedNodeDataset(
        num_samples=50,
        num_classes=7,
        image_size=224,
        seed=random_seed
    )


@pytest.fixture
def mock_dataloader(random_seed):
    """Create a mock DataLoader for testing."""
    return create_mock_dataloader(
        dataset_type='ham10000',
        num_samples=32,
        batch_size=8,
        shuffle=False,
        seed=random_seed
    )


@pytest.fixture
def fst_stratified_data(random_seed):
    """Generate FST-stratified data for fairness testing."""
    return generate_mock_fst_stratified_data(
        num_samples=600,
        num_classes=7,
        ensure_balance=True,
        seed=random_seed
    )


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def mock_resnet_model(device):
    """
    Create a mock ResNet model for testing.

    Note: Uses a minimal architecture to speed up tests.
    """
    try:
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 7)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        pytest.skip("torchvision not available")


@pytest.fixture
def mock_trained_model_weights(temp_dir, random_seed):
    """Create mock trained model weights for testing loading/saving."""
    torch.manual_seed(random_seed)
    weights = {
        'epoch': 10,
        'model_state_dict': {
            'conv1.weight': torch.randn(64, 3, 7, 7),
            'fc.weight': torch.randn(7, 512),
            'fc.bias': torch.randn(7)
        },
        'optimizer_state_dict': {},
        'loss': 0.345,
        'accuracy': 0.876
    }

    checkpoint_path = temp_dir / "mock_checkpoint.pth"
    torch.save(weights, checkpoint_path)
    return checkpoint_path


# ============================================================================
# PREDICTION & LABEL FIXTURES
# ============================================================================

@pytest.fixture
def mock_predictions_calibrated(random_seed):
    """Generate well-calibrated mock predictions."""
    return generate_mock_predictions(
        num_samples=100,
        num_classes=7,
        calibrated=True,
        seed=random_seed
    )


@pytest.fixture
def mock_predictions_uncalibrated(random_seed):
    """Generate poorly calibrated (overconfident) mock predictions."""
    return generate_mock_predictions(
        num_samples=100,
        num_classes=7,
        calibrated=False,
        seed=random_seed
    )


@pytest.fixture
def binary_predictions():
    """Create simple binary classification predictions for metric testing."""
    predictions = torch.tensor([
        [0.9, 0.1],  # Confident class 0
        [0.7, 0.3],  # Likely class 0
        [0.3, 0.7],  # Likely class 1
        [0.1, 0.9],  # Confident class 1
        [0.6, 0.4],  # Slightly class 0
    ])
    labels = torch.tensor([0, 0, 1, 1, 0])
    return predictions, labels


# ============================================================================
# FAIRNESS TESTING FIXTURES
# ============================================================================

@pytest.fixture
def fst_groups():
    """Fitzpatrick Skin Type group definitions."""
    return {
        'light': [1, 2],      # FST I-II (fair skin)
        'medium': [3, 4],     # FST III-IV (medium skin)
        'dark': [5, 6]        # FST V-VI (dark skin)
    }


@pytest.fixture
def mock_fairness_metrics_data(random_seed):
    """
    Generate mock data specifically for fairness metrics testing.

    Returns predictions, labels, and FST for 3 groups with known disparities.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Group 1 (FST I-II): High accuracy
    pred_g1 = torch.softmax(torch.randn(100, 7) + 2, dim=1)
    labels_g1 = torch.randint(0, 7, (100,))
    fst_g1 = torch.randint(1, 3, (100,))

    # Group 2 (FST III-IV): Medium accuracy
    pred_g2 = torch.softmax(torch.randn(100, 7) + 1, dim=1)
    labels_g2 = torch.randint(0, 7, (100,))
    fst_g2 = torch.randint(3, 5, (100,))

    # Group 3 (FST V-VI): Lower accuracy (simulating bias)
    pred_g3 = torch.softmax(torch.randn(100, 7), dim=1)
    labels_g3 = torch.randint(0, 7, (100,))
    fst_g3 = torch.randint(5, 7, (100,))

    predictions = torch.cat([pred_g1, pred_g2, pred_g3])
    labels = torch.cat([labels_g1, labels_g2, labels_g3])
    fst = torch.cat([fst_g1, fst_g2, fst_g3])

    return predictions, labels, fst


# ============================================================================
# IMAGE PREPROCESSING FIXTURES
# ============================================================================

@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for preprocessing tests."""
    return torch.rand(3, 224, 224)


@pytest.fixture
def sample_image_batch():
    """Create a batch of sample images for preprocessing tests."""
    return torch.rand(8, 3, 224, 224)


@pytest.fixture
def imagenet_mean_std():
    """ImageNet normalization statistics."""
    return {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary for testing."""
    return {
        'model': {
            'architecture': 'resnet50',
            'num_classes': 7,
            'pretrained': True
        },
        'data': {
            'batch_size': 32,
            'image_size': 224,
            'num_workers': 0
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'weight_decay': 1e-4
        },
        'fairness': {
            'fst_groups': {
                'light': [1, 2],
                'medium': [3, 4],
                'dark': [5, 6]
            },
            'metrics': ['auroc', 'eod', 'ece']
        }
    }


@pytest.fixture
def mock_tensorboard_dir(temp_dir):
    """Create a temporary TensorBoard log directory."""
    tb_dir = temp_dir / "tensorboard_logs"
    tb_dir.mkdir(exist_ok=True)
    return tb_dir


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def capture_warnings():
    """Fixture to capture warnings during tests."""
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


@pytest.fixture(autouse=True)
def reset_random_state(random_seed):
    """Auto-use fixture to reset random state before each test."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


# ============================================================================
# PYTEST CONFIGURATION HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (>5 seconds)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if no GPU available."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
