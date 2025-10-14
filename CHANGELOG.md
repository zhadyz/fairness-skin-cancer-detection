# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-14

### Added
- Initial project structure and foundation
- Baseline model implementations (ResNet50, EfficientNet B4, InceptionV3)
- Comprehensive fairness evaluation framework (AUROC per FST, EOD, ECE)
- Training pipeline with TensorBoard logging and checkpointing
- Testing infrastructure with 129 tests (unit + integration)
- DevOps infrastructure (Docker, CI/CD, pre-commit hooks)
- Dataset acquisition documentation and preprocessing pipeline
- FST annotation protocol with ITA calculation
- Synthetic augmentation research (FairSkin, DermDiff, LoRA)
- Comprehensive documentation (10+ guides)

### Documentation
- Implementation roadmap (32-week plan, 5 phases)
- Architecture overview with model comparisons
- Dataset access guide (6 datasets with FST annotations)
- Environment setup guide (Windows/Linux/macOS)
- Testing guide (pytest, coverage, CI/CD)
- Code quality standards (PEP 8, type hints, docstrings)
- Docker deployment guide
- Experiment tracking guide (TensorBoard, W&B)
- FST annotation protocol with ITA calculation
- Synthetic augmentation techniques comparison
- Baseline results documentation

### Infrastructure
- Pre-commit hooks (Black, isort, Flake8, mypy, Bandit)
- GitHub Actions CI/CD (testing, building, releasing)
- Docker multi-stage builds (dev, training, inference)
- TensorBoard experiment tracking integration
- Pytest with 80%+ coverage target
- Comprehensive .gitignore for ML projects
- pyproject.toml for build configuration
- .coveragerc for test coverage configuration

### Research Foundation
- Survey paper citation: Flores & Alzahrani (2025) - CSUSB
- Literature review: 100+ papers on fairness-aware skin cancer detection
- Benchmarks: NHS DERM (97% sensitivity), DermaSensor (FDA cleared), BiaslessNAS
- Target metrics: <4% AUROC gap, 95%+ sensitivity (all FST groups)
- Clinical validation protocols defined

### Attribution
- Research: Jasmin Flores & Dr. Nabeel Alzahrani (CSUSB)
- Framework: MENDICANT_BIAS Multi-Agent System
- Agents: the_didact (Research), hollowed_eyes (Development), loveless (QA/Security), zhadyz (DevOps)

## [Unreleased]

### Planned (Phase 1.5)
- HAM10000 dataset integration with FST annotations
- ISIC 2019/2020 dataset integration
- Additional baseline model experiments (DenseNet, MobileNetV3)
- Ensemble baseline evaluation

### Planned (Phase 2)
- FairSkin diffusion-based augmentation implementation
- FairDisCo adversarial debiasing
- CIRCLe color-invariant representation learning
- Comprehensive fairness intervention comparison

### Planned (Phase 3)
- Hybrid ConvNeXtV2-Swin Transformer architecture
- Multi-scale feature fusion
- Attention mechanism integration
- Domain adaptation techniques

### Planned (Phase 4)
- FairPrune pruning for model compression
- Post-training quantization (INT8/FP16)
- SHAP-based explainability dashboard
- Production hardening and optimization

### Planned (Phase 5)
- Clinical validation study design
- Regulatory documentation (FDA, CE Mark)
- Deployment on edge devices
- Real-world teledermatology integration
- Community feedback integration

---

**Version Format**: MAJOR.MINOR.PATCH
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)
