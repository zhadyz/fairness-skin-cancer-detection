"""
Unit Tests for FairSkin Diffusion Augmentation.

Comprehensive test suite covering:
- FairSkinDiffusionModel: Stable Diffusion + LoRA
- LoRATrainer: Fine-tuning pipeline
- Quality metrics: FID, LPIPS, diversity
- Synthetic dataset: Loading and mixing
- Integration tests: End-to-end generation

Framework: MENDICANT_BIAS - Phase 2
Agent: HOLLOWED_EYES
Version: 0.3.0
Date: 2025-10-13

Run with:
    pytest tests/unit/test_fairskin.py -v
    pytest tests/unit/test_fairskin.py -v --cov=src/augmentation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

# Import modules to test
from src.augmentation.fairskin_diffusion import FairSkinDiffusionModel, DIAGNOSIS_LABELS, FST_DESCRIPTIONS
from src.augmentation.lora_trainer import LoRATrainingConfig
from src.augmentation.quality_metrics import (
    pil_to_tensor, tensor_to_pil, compute_diversity_score, QualityFilter
)
from src.augmentation.synthetic_dataset import SyntheticDermoscopyDataset, MixedDataset


# ==================== Fixtures ====================

@pytest.fixture
def device():
    """Get device for tests (prefer CPU for CI)."""
    return "cpu"  # Use CPU for tests to avoid GPU requirements


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image():
    """Create sample RGB image."""
    image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


@pytest.fixture
def sample_images():
    """Create batch of sample images."""
    return [
        Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        for _ in range(10)
    ]


@pytest.fixture
def mock_ham10000_dataset(temp_dir):
    """Create mock HAM10000 dataset for testing."""
    class MockDataset:
        def __init__(self):
            self.samples = []
            for i in range(100):
                self.samples.append({
                    'image': torch.rand(3, 512, 512),
                    'label': i % 7,
                    'fst': (i % 6) + 1,
                    'is_synthetic': 0,
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    return MockDataset()


@pytest.fixture
def mock_synthetic_dir(temp_dir):
    """Create mock synthetic image directory."""
    synthetic_dir = Path(temp_dir) / "synthetic"
    synthetic_dir.mkdir()

    # Create dummy images
    diagnoses = ['melanoma', 'nevus', 'basal_cell_carcinoma']
    fsts = [5, 6]

    for fst in fsts:
        for dx in diagnoses:
            for i in range(5):
                filename = f"synthetic_fst{fst}_{dx}_{i:05d}.png"
                img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                img.save(synthetic_dir / filename)

    return str(synthetic_dir)


# ==================== Test FairSkinDiffusionModel ====================

class TestFairSkinDiffusionModel:
    """Tests for FairSkinDiffusionModel."""

    @pytest.mark.skip(reason="Requires Stable Diffusion download (large model)")
    def test_model_initialization(self, device):
        """Test model initialization."""
        model = FairSkinDiffusionModel(device=device)
        assert model.device.type == device
        assert model.pipe is not None

    def test_prompt_generation(self):
        """Test text prompt generation."""
        model_mock = FairSkinDiffusionModel.__new__(FairSkinDiffusionModel)

        # Test with diagnosis int
        prompt = model_mock.create_prompt(diagnosis=4, fst=6)
        assert 'melanoma' in prompt.lower()
        assert 'type VI' in prompt or 'type 6' in prompt

        # Test with diagnosis string
        prompt = model_mock.create_prompt(diagnosis='nevus', fst=1)
        assert 'nevus' in prompt.lower()

        # Test different styles
        for style in ['clinical', 'dermoscopic', 'medical']:
            prompt = model_mock.create_prompt(diagnosis=0, fst=3, style=style)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_negative_prompt(self):
        """Test negative prompt generation."""
        model_mock = FairSkinDiffusionModel.__new__(FairSkinDiffusionModel)
        negative_prompt = model_mock.create_negative_prompt()

        assert isinstance(negative_prompt, str)
        assert 'blurry' in negative_prompt.lower()
        assert 'watermark' in negative_prompt.lower()

    def test_diagnosis_labels(self):
        """Test diagnosis label mapping."""
        assert len(DIAGNOSIS_LABELS) == 7
        assert 0 in DIAGNOSIS_LABELS
        assert 6 in DIAGNOSIS_LABELS
        assert 'melanoma' in DIAGNOSIS_LABELS.values()

    def test_fst_descriptions(self):
        """Test FST descriptions."""
        assert len(FST_DESCRIPTIONS) == 6
        assert 1 in FST_DESCRIPTIONS
        assert 6 in FST_DESCRIPTIONS
        assert 'very dark' in FST_DESCRIPTIONS[6].lower()


# ==================== Test LoRATrainingConfig ====================

class TestLoRATrainingConfig:
    """Tests for LoRA training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LoRATrainingConfig()

        assert config.lora_rank == 16
        assert config.lora_alpha == 16
        assert config.num_train_steps == 10000
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4
        assert config.target_modules is not None
        assert len(config.target_modules) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = LoRATrainingConfig(
            lora_rank=32,
            num_train_steps=5000,
            batch_size=8,
            learning_rate=5e-5,
        )

        assert config.lora_rank == 32
        assert config.num_train_steps == 5000
        assert config.batch_size == 8
        assert config.learning_rate == 5e-5

    def test_config_validation(self):
        """Test configuration validation."""
        config = LoRATrainingConfig(
            lora_rank=16,
            lora_alpha=32,  # Typically 2x rank
            gradient_accumulation_steps=4,
        )

        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        assert effective_batch_size == 16


# ==================== Test Quality Metrics ====================

class TestQualityMetrics:
    """Tests for quality metrics."""

    def test_pil_to_tensor(self, sample_image):
        """Test PIL Image to tensor conversion."""
        tensor = pil_to_tensor(sample_image, normalize=True)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 512, 512)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_tensor_to_pil(self):
        """Test tensor to PIL Image conversion."""
        tensor = torch.rand(1, 3, 512, 512)
        image = tensor_to_pil(tensor)

        assert isinstance(image, Image.Image)
        assert image.size == (512, 512)

    def test_diversity_score(self, sample_images):
        """Test diversity score computation."""
        # Skip if no GPU and requires model download
        pytest.skip("Requires model download")

        diversity = compute_diversity_score(sample_images, device="cpu")

        assert isinstance(diversity, float)
        assert 0.0 <= diversity <= 1.0

    def test_quality_filter_brightness(self):
        """Test brightness check in quality filter."""
        quality_filter = QualityFilter(device="cpu")

        # Normal brightness image
        normal_img = Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8))
        assert quality_filter.check_brightness(normal_img) == True

        # Pure black image
        black_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        assert quality_filter.check_brightness(black_img) == False

        # Pure white image
        white_img = Image.fromarray(np.full((512, 512, 3), 255, dtype=np.uint8))
        assert quality_filter.check_brightness(white_img) == False

    def test_quality_filter_resolution(self):
        """Test resolution check in quality filter."""
        quality_filter = QualityFilter(device="cpu")

        # Valid resolution
        valid_img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        assert quality_filter.check_resolution(valid_img) == True

        # Too small resolution
        small_img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        assert quality_filter.check_resolution(small_img) == False


# ==================== Test Synthetic Dataset ====================

class TestSyntheticDataset:
    """Tests for SyntheticDermoscopyDataset."""

    def test_dataset_loading(self, mock_synthetic_dir):
        """Test loading synthetic dataset."""
        dataset = SyntheticDermoscopyDataset(synthetic_dir=mock_synthetic_dir)

        assert len(dataset) > 0
        assert len(dataset) == 30  # 2 FSTs × 3 diagnoses × 5 images

    def test_dataset_getitem(self, mock_synthetic_dir):
        """Test dataset __getitem__."""
        dataset = SyntheticDermoscopyDataset(synthetic_dir=mock_synthetic_dir)

        sample = dataset[0]

        assert 'image' in sample
        assert 'label' in sample
        assert 'fst' in sample
        assert 'is_synthetic' in sample

        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape[0] == 3  # RGB
        assert sample['is_synthetic'] == 1

    def test_class_distribution(self, mock_synthetic_dir):
        """Test class distribution."""
        dataset = SyntheticDermoscopyDataset(synthetic_dir=mock_synthetic_dir)

        dist = dataset.get_class_distribution()

        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_fst_distribution(self, mock_synthetic_dir):
        """Test FST distribution."""
        dataset = SyntheticDermoscopyDataset(synthetic_dir=mock_synthetic_dir)

        dist = dataset.get_fst_distribution()

        assert isinstance(dist, dict)
        assert 5 in dist
        assert 6 in dist


# ==================== Test Mixed Dataset ====================

class TestMixedDataset:
    """Tests for MixedDataset."""

    def test_mixed_dataset_creation(self, mock_ham10000_dataset, mock_synthetic_dir):
        """Test mixed dataset creation."""
        synthetic_dataset = SyntheticDermoscopyDataset(synthetic_dir=mock_synthetic_dir)

        mixed_dataset = MixedDataset(
            real_dataset=mock_ham10000_dataset,
            synthetic_dataset=synthetic_dataset,
            balance_fst=False,  # Don't balance for simpler test
        )

        assert len(mixed_dataset) > 0
        assert len(mixed_dataset) <= len(mock_ham10000_dataset) + len(synthetic_dataset)

    def test_mixed_dataset_getitem(self, mock_ham10000_dataset, mock_synthetic_dir):
        """Test mixed dataset __getitem__."""
        synthetic_dataset = SyntheticDermoscopyDataset(synthetic_dir=mock_synthetic_dir)

        mixed_dataset = MixedDataset(
            real_dataset=mock_ham10000_dataset,
            synthetic_dataset=synthetic_dataset,
            balance_fst=False,
        )

        sample = mixed_dataset[0]

        assert 'image' in sample
        assert 'label' in sample
        assert 'fst' in sample
        assert 'is_synthetic' in sample

        # is_synthetic should be 0 or 1
        assert sample['is_synthetic'] in [0, 1]

    def test_fst_dependent_ratios(self, mock_ham10000_dataset, mock_synthetic_dir):
        """Test FST-dependent synthetic ratios."""
        synthetic_dataset = SyntheticDermoscopyDataset(synthetic_dir=mock_synthetic_dir)

        synthetic_ratios = {
            1: 0.2, 2: 0.2, 3: 0.3,
            4: 0.5, 5: 0.7, 6: 0.8,
        }

        mixed_dataset = MixedDataset(
            real_dataset=mock_ham10000_dataset,
            synthetic_dataset=synthetic_dataset,
            synthetic_ratio_by_fst=synthetic_ratios,
            balance_fst=False,
        )

        assert len(mixed_dataset) > 0


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for end-to-end workflows."""

    @pytest.mark.skip(reason="Requires full model download and GPU")
    def test_generation_pipeline(self, device, temp_dir):
        """Test full generation pipeline."""
        # Initialize model
        model = FairSkinDiffusionModel(device=device)

        # Generate single image
        image = model.generate_image(
            diagnosis=4,  # Melanoma
            fst=6,
            num_inference_steps=20,  # Fast for testing
            seed=42,
        )

        assert isinstance(image, Image.Image)
        assert image.size == (512, 512)

        # Save image
        output_path = Path(temp_dir) / "test_output.png"
        image.save(output_path)
        assert output_path.exists()

    def test_config_loading(self):
        """Test configuration file loading."""
        config_path = Path("configs/fairskin_config.yaml")

        if not config_path.exists():
            pytest.skip("Config file not found")

        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'stable_diffusion' in config
        assert 'lora' in config
        assert 'training' in config
        assert 'generation' in config

        # Validate key parameters
        assert config['lora']['rank'] > 0
        assert config['training']['num_train_steps'] > 0
        assert config['generation']['target_fsts']


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance and memory tests."""

    def test_prompt_generation_performance(self):
        """Test prompt generation speed."""
        import time

        model_mock = FairSkinDiffusionModel.__new__(FairSkinDiffusionModel)

        start_time = time.time()
        for _ in range(1000):
            prompt = model_mock.create_prompt(
                diagnosis=np.random.randint(0, 7),
                fst=np.random.randint(1, 7)
            )
        elapsed = time.time() - start_time

        # Should be very fast (<1 second for 1000 prompts)
        assert elapsed < 1.0

    def test_dataset_loading_performance(self, mock_synthetic_dir):
        """Test dataset loading speed."""
        import time

        start_time = time.time()
        dataset = SyntheticDermoscopyDataset(
            synthetic_dir=mock_synthetic_dir,
            load_to_memory=False,
        )
        elapsed = time.time() - start_time

        # Should load quickly (<5 seconds)
        assert elapsed < 5.0


# ==================== Test Summary ====================

def test_import_all_modules():
    """Test that all modules can be imported."""
    try:
        from src.augmentation import (
            FairSkinDiffusionModel,
            LoRATrainer,
            SyntheticDermoscopyDataset,
            MixedDataset,
            compute_fid,
            compute_lpips,
            compute_diversity_score,
            QualityFilter,
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


if __name__ == "__main__":
    """Run tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
