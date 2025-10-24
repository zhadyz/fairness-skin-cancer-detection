# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **research-driven, production-grade AI system for equitable skin cancer detection** across Fitzpatrick skin types I-VI. The project addresses critical healthcare disparities where existing models show 15-30% performance drops on darker skin tones.

**Current Version**: v0.5.0-dev (Phase 4 - Production Hardening, 70% complete)
**Mission**: Serve humanity through equitable AI for skin cancer detection

## Core Architecture

### Three-Tier Fairness Methodology

The system implements a sophisticated pipeline with three integrated fairness interventions:

1. **FairDisCo (v0.2.1)**: Adversarial debiasing using gradient reversal layer
   - `src/models/fairdisco_model.py` - FST discriminator + supervised contrastive loss
   - `src/training/fairdisco_trainer.py` - Three-loss training system
   - Expected: 65% EOD reduction

2. **CIRCLe (v0.2.2)**: Color-invariant learning via LAB transformations
   - `src/fairness/color_transforms.py` - RGB↔LAB conversion with FST statistics
   - `src/fairness/circle_regularization.py` - Multi-target regularization
   - `src/models/circle_model.py` - Four-loss training (extends FairDisCo)
   - Expected: 33% additional AUROC gap reduction

3. **FairSkin (v0.3.0)**: Diffusion-based synthetic augmentation
   - `src/augmentation/fairskin_diffusion.py` - Stable Diffusion v1.5 + LoRA
   - `src/augmentation/lora_trainer.py` - Rank-16 U-Net adaptation
   - `src/augmentation/quality_metrics.py` - FID/LPIPS validation
   - Expected: +18-21% FST VI AUROC

### Hybrid Model Architecture (v0.4.0)

**ConvNeXtV2-Swin Transformer** with multi-scale pyramid fusion:
- `src/models/convnextv2.py` - Local texture extraction (3 stages)
- `src/models/swin_transformer.py` - Global context modeling (2 stages)
- `src/models/hybrid_model.py` - 4-scale pyramid fusion with channel attention
- Target: 91-93% AUROC, <2% gap across FST groups

### Production Hardening (v0.5.0-dev)

- **Compression**: `src/compression/fairprune.py` (fairness-aware pruning), `src/compression/quantization.py` (INT8), `src/compression/onnx_export.py`
- **Explainability**: `src/explainability/shap_explainer.py` (GradientSHAP, IntegratedGradients, Saliency)
- **API**: `api/main.py` (FastAPI with 5 endpoints), `api/inference.py` (ONNX/PyTorch engine)
- Target: 27MB model, <100ms inference, 91% AUROC, 1.5% gap

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Testing

```bash
# Run all tests with coverage
pytest

# Run specific test markers
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
pytest -m fairness          # Fairness-specific tests

# Run specific test file
pytest tests/unit/test_fairdisco.py -v

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run single test
pytest tests/unit/test_fairprune.py::TestFairPrune::test_structured_pruning -v
```

### Code Quality

```bash
# Format code (Black, line length 100)
black src/ tests/ --line-length 100

# Sort imports (isort)
isort src/ tests/ --profile black

# Lint (Flake8)
flake8 src/ tests/ --max-line-length 100

# Type checking (mypy)
mypy src/ --ignore-missing-imports

# Security scanning (Bandit)
bandit -r src/ -ll
```

### Training

```bash
# Baseline training (ResNet50)
python experiments/baseline/train_resnet50.py --config configs/baseline_config.yaml

# FairDisCo adversarial debiasing
python experiments/fairness/train_fairdisco.py --config configs/fairdisco_config.yaml

# CIRCLe color-invariant learning
python experiments/fairness/train_circle.py --config configs/circle_config.yaml

# Hybrid architecture
python experiments/baseline/train_resnet50.py --config configs/hybrid_config.yaml --model hybrid

# Training with FairSkin augmentation
python experiments/augmentation/train_with_fairskin.py --config configs/fairskin_config.yaml
```

### Synthetic Data Generation

```bash
# LoRA fine-tuning on HAM10000
python experiments/augmentation/train_lora.py --config configs/fairskin_config.yaml

# Generate synthetic images
python experiments/augmentation/generate_fairskin.py \
    --lora-checkpoint outputs/lora/checkpoint-10000 \
    --num-samples 60000 \
    --fst-ratios 0.1 0.1 0.15 0.15 0.25 0.25
```

### Model Compression

```bash
# FairPrune compression
python scripts/compress_model.py \
    --model-path models/hybrid_model.pth \
    --method fairprune \
    --sparsity 0.6

# INT8 quantization
python scripts/compress_model.py \
    --model-path models/hybrid_model.pth \
    --method quantize \
    --precision int8

# ONNX export
python scripts/compress_model.py \
    --model-path models/hybrid_model.pth \
    --method onnx \
    --opset-version 17
```

### API Server

```bash
# Start development server
python api/start_server.py --host 0.0.0.0 --port 8000

# Start with hot reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test API endpoints
python api/test_api.py

# Production deployment (with Docker)
docker-compose -f docker-compose.prod.yml up -d
```

### Docker

```bash
# Build development image
docker build -t skin-cancer-detection:dev .

# Build production image
docker build -f Dockerfile.production -t skin-cancer-detection:prod .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### HAM10000 Dataset Setup

```bash
# Automated setup (downloads, FST annotation, splits)
python scripts/setup_ham10000.py

# Manual steps:
# 1. Generate FST annotations
python scripts/generate_ham10000_fst.py --data-dir data/HAM10000

# 2. Create stratified splits
python scripts/create_ham10000_splits.py --data-dir data/HAM10000

# 3. Verify dataset
python scripts/verify_ham10000.py --data-dir data/HAM10000
```

## Key Architecture Patterns

### Configuration Management

All experiments use YAML configurations in `configs/`:
- `baseline_config.yaml` - ResNet50/EfficientNet baselines
- `fairdisco_config.yaml` - Adversarial debiasing
- `circle_config.yaml` - Color-invariant learning
- `fairskin_config.yaml` - Diffusion augmentation
- `hybrid_config.yaml` - ConvNeXtV2-Swin architecture
- `production_config.yaml` - Compression + explainability + API

### Training Pipeline Flow

1. **Data Loading**: `src/data/ham10000_dataset.py` with FST annotations
2. **Model Creation**: Factory pattern in `src/models/__init__.py`
3. **Training**: Specialized trainers in `src/training/` (base, FairDisCo, CIRCLe)
4. **Evaluation**: `src/evaluation/fairness_metrics.py` computes per-FST metrics
5. **Checkpointing**: Best model saved based on validation AUROC
6. **Logging**: TensorBoard integration for real-time monitoring

### Fairness Evaluation

The `FairnessMetrics` class (`src/evaluation/fairness_metrics.py`) computes:
- **AUROC per FST**: Performance for each Fitzpatrick skin type (I-VI)
- **AUROC Gap**: max(AUROC) - min(AUROC) across groups
- **EOD (Equal Opportunity Difference)**: Max TPR difference between FST groups
- **ECE (Expected Calibration Error)**: Per-FST calibration quality
- **Demographic Parity**: Positive prediction rate differences

Always evaluate fairness when modifying models or training procedures.

### Multi-Loss Training

FairDisCo and CIRCLe use multiple loss components with lambda scheduling:

```python
# FairDisCo (3 losses)
total_loss = L_cls + lambda_adv * L_adv + lambda_con * L_con

# CIRCLe (4 losses)
total_loss = L_cls + lambda_adv * L_adv + lambda_con * L_con + lambda_reg * L_reg
```

Lambda scheduling (warmup → ramp-up → full) is critical - see trainer implementations.

### Gradient Reversal Layer (GRL)

FairDisCo uses a custom GRL implementation (`src/models/fairdisco_model.py`):
- Forward pass: Identity
- Backward pass: Negates gradients (multiplies by -lambda)
- Forces encoder to learn FST-invariant features

## Code Style

- **Line Length**: 100 characters (not 79)
- **Imports**: Standard library → Third-party → Local (separated by blank lines)
- **Naming**: snake_case (functions/vars), PascalCase (classes), UPPER_SNAKE_CASE (constants)
- **Type Hints**: Use for function signatures (enforced by mypy)
- **Docstrings**: Google-style for all public functions/classes

Example:
```python
def compute_fairness_gap(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    fst_groups: torch.Tensor
) -> Dict[str, float]:
    """
    Compute AUROC gap across Fitzpatrick skin types.

    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        fst_groups: Fitzpatrick skin type labels (1-6)

    Returns:
        Dictionary with AUROC per FST and gap metrics
    """
```

## Testing Requirements

- **Unit Tests**: 80%+ coverage minimum
- **Integration Tests**: For multi-component workflows (fairness pipeline, compression)
- **GPU Tests**: Mark with `@pytest.mark.requires_gpu`, skip if GPU unavailable
- **Fixtures**: Use `tests/conftest.py` for shared test data

Example test structure:
```python
@pytest.mark.unit
class TestFairDisCoModel:
    def test_gradient_reversal_layer(self):
        """Test GRL negates gradients during backward pass."""
        # Implementation

    @pytest.mark.requires_gpu
    def test_discriminator_convergence(self):
        """Test discriminator reaches equilibrium (~20-25% accuracy)."""
        # Implementation
```

## Important Constraints

### Fairness First
When modifying models or training:
1. Always evaluate per-FST metrics (not just overall accuracy)
2. Check AUROC gap (<4% target) and EOD (<0.05 target)
3. Ensure FST V-VI representation in calibration/validation sets
4. Document fairness impact in commit messages

### Model Compression
When using FairPrune or quantization:
- Skip final classifier layers and fusion modules
- Attention layers are sensitive - avoid pruning/quantizing
- Always fine-tune after pruning (5 epochs minimum)
- Validate fairness preservation (<2% AUROC gap increase)

### Production Deployment
- Model size: <50MB (27MB target achieved)
- Inference time: <100ms per image
- API rate limits: 10 req/min (predict), 5 req/min (batch)
- SHAP explanations: <2 seconds per image

## Framework Attribution

This project was developed using the **MENDICANT_BIAS Multi-Agent Framework**:
- **the_didact**: Research and literature analysis
- **hollowed_eyes**: Software architecture and implementation
- **loveless**: QA, security validation, and testing
- **zhadyz**: DevOps, Docker, and infrastructure

Research foundation by **Jasmin Flores** and **Dr. Nabeel Alzahrani** (CSUSB).

## Documentation References

Key docs in `docs/`:
- `architecture.md` - System design overview
- `fairdisco_training_guide.md` - Adversarial debiasing details
- `circle_training_guide.md` - Color-invariant learning guide
- `fairskin_usage_guide.md` - Diffusion augmentation workflow
- `api_documentation.md` - FastAPI endpoint reference
- `explainability_setup.md` - SHAP integration guide
- `code_quality_standards.md` - Full style guide

## Performance Benchmarks

Baseline (ResNet50, no fairness interventions):
- AUROC FST I-III: 91.3%
- AUROC FST V-VI: 75.4%
- Gap: **-15.9%** ❌

Target (Hybrid + FairDisCo + CIRCLe + FairSkin):
- AUROC FST I-III: 91-93%
- AUROC FST IV-VI: 89-92%
- Gap: **<4%** ✅
