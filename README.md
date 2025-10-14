# Fairness-Aware AI for Skin Cancer Detection

A research-driven, production-grade AI system for equitable skin cancer detection across all skin tones.

## Overview

This project implements state-of-the-art machine learning techniques to achieve fair diagnostic performance across Fitzpatrick skin types I-VI, addressing the critical healthcare disparity where existing models show 15-30% performance drops on darker skin tones.

**Research Foundation**: Inspired by the comprehensive survey by Flores & Alzahrani (2025) analyzing 100+ experimental studies on fairness-aware skin cancer detection.

## Key Features

- **Fairness-First Architecture**: Hybrid ConvNeXtV2-Swin Transformer with three-tier fairness methodology
- **Proven Techniques**: FairSkin diffusion augmentation, FairDisCo adversarial debiasing, CIRCLe color-invariant learning
- **Clinical-Grade**: Target performance benchmarks from deployed systems (NHS DERM: 97% sensitivity across all FST)
- **Transparent & Ethical**: Comprehensive model cards with disaggregated subgroup metrics, patient co-design
- **Edge-Optimized**: <50MB model, <100ms inference for teledermatology applications

## Documentation

- [Implementation Roadmap](docs/roadmap.md) - Detailed 32-week development plan
- [Architecture](docs/architecture.md) - Technical specifications (coming soon)
- [Datasets](docs/datasets.md) - Data sources and preprocessing (coming soon)

## Project Status

**Phase**: Foundation (Week 1)
**Progress**: Repository initialized, baseline experiments in progress

## Citation

If you use this work, please cite the foundational survey:

```
Flores, J., & Alzahrani, N. (2025). AI Skin Cancer Detection Across Skin Tones:
A Survey of Experimental Advances, Fairness Techniques, and Dataset Limitations.
Computers (MDPI). [Submitted]
```

## License

Apache 2.0

## Acknowledgments

Special thanks to **Jasmin Flores** and **Dr. Nabeel Alzahrani** (California State University, San Bernardino) for their comprehensive survey that provided the research foundation for this implementation.

---

Developed with MENDICANT_BIAS Multi-Agent Framework
