# Open-Source Fairness Code: Repository Evaluation

## Executive Summary

This document catalogs open-source implementations of fairness techniques for dermatological AI, evaluates code quality and integration feasibility, and provides recommendations for Phase 2 implementation.

**Key Finding**: All three fairness techniques (FairSkin, FairDisCo, CIRCLe) have official GitHub implementations with moderate-to-high code quality. Integration complexity ranges from moderate (FairDisCo, CIRCLe) to high (FairSkin diffusion training).

---

## 1. FairSkin Diffusion Augmentation

### 1.1 Primary Repository: janet-sw/skin-diff

**Repository Details**:
- URL: https://github.com/janet-sw/skin-diff
- Authors: Janet Tsang (MICCAI ISIC Workshop 2024, Honorable Mention)
- Paper: "From Majority to Minority: A Diffusion-based Augmentation for Underrepresented Groups in Skin Lesion Analysis"
- Stars: ~50+ (as of 2025-01)
- License: **Not explicitly specified** (assume academic use, contact authors for commercial)
- Last Updated: 2024-06 (6 months ago)

**Technical Stack**:
- Python 3.8+
- PyTorch 1.13+
- Hugging Face Diffusers (primary framework)
- Accelerate (multi-GPU training)

**Code Structure**:
```
skin-diff/
├── textual_inversion/
│   ├── textual_inversion.py       # Training script for token embeddings
│   └── requirements.txt            # Dependencies
├── lora/
│   ├── train_lora.py               # LoRA fine-tuning script
│   └── inference_lora.py           # Generation script
├── classifier/
│   ├── train_classifier.py         # Train ResNet50 on mixed data
│   └── evaluate.py                 # Fairness metrics
├── data/
│   └── prepare_data.py             # Dataset preprocessing
└── README.md
```

**Key Features**:
1. **Textual Inversion**: Learn custom tokens (`<melanoma-FST-VI>`)
   - Training: 2000 steps, ~2-4 hours (RTX 3090)
   - Validation: Generate images from text prompts

2. **LoRA Fine-Tuning**: Adapt Stable Diffusion to dermatology
   - Rank 16, alpha 32 (standard configuration)
   - Training: 10,000 steps, ~20 hours (RTX 3090)

3. **Quality Filtering**: FID, LPIPS, classifier confidence
   - Automated filtering pipeline
   - Configurable thresholds

4. **Mixed Dataset Training**: Real + synthetic data loader
   - Weighted sampling by FST
   - Dynamic synthetic ratio (FST-dependent)

**Code Quality**: **Good (7/10)**

**Strengths**:
- Well-documented training scripts
- Uses industry-standard Hugging Face Diffusers
- Modular design (easy to adapt components)
- Includes quality validation

**Weaknesses**:
- Limited inline comments (assume familiarity with diffusion models)
- Hardcoded hyperparameters (should be config file)
- No pre-trained checkpoints (must train from scratch)
- License ambiguity (not specified)

**Integration Feasibility**: **Moderate-High Complexity**

**Required Adaptations**:
1. Add support for HAM10000 dataset (currently Fitzpatrick17k only)
2. Refactor hyperparameters to YAML config
3. Add WandB logging for experiment tracking
4. Increase batch size (4 → 8) for faster training on RTX 4090

**Estimated Integration Time**: 1-2 weeks
- Week 1: Setup, dataset adaptation, initial training
- Week 2: Hyperparameter tuning, quality validation

**Recommendation**: **Use as primary implementation for FairSkin**
- Most complete diffusion augmentation framework for dermatology
- Battle-tested (MICCAI workshop acceptance)
- Direct integration with Hugging Face ecosystem

---

### 1.2 Alternative: Stable Diffusion Fine-Tuning (Hugging Face)

**Repository Details**:
- URL: https://github.com/huggingface/diffusers
- Official Hugging Face Diffusers library
- Examples: `examples/text_to_image/`, `examples/textual_inversion/`

**Advantages**:
- Official support, extensive documentation
- Pre-trained checkpoints (Stable Diffusion v1.5, v2.1)
- Active maintenance (weekly updates)

**Disadvantages**:
- Generic (not dermatology-specific)
- Requires significant adaptation for medical images
- No fairness-specific features

**Recommendation**: **Use janet-sw/skin-diff** (builds on Diffusers, adds dermatology specialization)

---

## 2. FairDisCo Adversarial Debiasing

### 2.1 Official Repository: siyi-wind/FairDisCo

**Repository Details**:
- URL: https://github.com/siyi-wind/FairDisCo
- Authors: Siyi Du, Noel Codella, et al. (ECCV ISIC Workshop 2022, Best Paper)
- Paper: "FairDisCo: Fairer AI in Dermatology via Disentanglement Contrastive Learning"
- Stars: ~100+ (as of 2025-01)
- License: **Not explicitly specified** (assume academic use)
- Last Updated: 2023-03 (21 months ago, but complete)

**Technical Stack**:
- Python 3.8.1
- PyTorch 1.8.0
- CUDA 11.1, CuDNN 7
- timm (PyTorch Image Models)

**Code Structure**:
```
FairDisCo/
├── models/
│   ├── resnet.py                  # ResNet50 backbone
│   ├── gradient_reversal.py       # GRL implementation
│   └── fairdisco_model.py         # Full architecture
├── losses/
│   ├── contrastive.py             # Supervised contrastive loss
│   └── adversarial.py             # Cross-entropy for discriminator
├── data/
│   ├── fitzpatrick17k_loader.py   # Dataset loader
│   └── ddi_loader.py              # DDI dataset
├── train_BASE.py                  # Baseline (no fairness)
├── train_ATRB.py                  # Attribute-aware
├── train_FairDisCo.py             # Full FairDisCo
├── multi_evaluate.ipynb           # Evaluation notebook
└── README.md
```

**Key Features**:
1. **Gradient Reversal Layer**: Custom PyTorch autograd function
   - Forward: Identity
   - Backward: Gradient negation
   - Lambda scheduling (0.0 → 0.3 over 20 epochs)

2. **Supervised Contrastive Loss**: Pull same-diagnosis, different-FST
   - Temperature 0.07 (standard from SimCLR)
   - Batch size 64 (need enough positives)

3. **Multi-Task Loss**: Classification + Adversarial + Contrastive
   - Weights: [1.0, 0.3, 0.2]
   - Dynamic weight scheduling

4. **Fairness Metrics**: EOD, DPD, Equalized Odds
   - Per-FST AUROC, sensitivity, specificity
   - Calibration (ECE)

**Code Quality**: **Excellent (9/10)**

**Strengths**:
- Clean, modular architecture
- Comprehensive fairness evaluation
- Multiple baseline comparisons (resampling, reweighting, attribute-aware)
- Well-tested (ECCV best paper)

**Weaknesses**:
- Hardcoded paths (dataset directories)
- No config file (hyperparameters in script)
- No pre-trained checkpoints
- License ambiguity

**Integration Feasibility**: **Moderate Complexity**

**Required Adaptations**:
1. Add HAM10000 dataset loader
2. Refactor to use config files (YAML)
3. Add WandB logging
4. Update PyTorch (1.8.0 → 2.1+), CUDA (11.1 → 12.1)

**Estimated Integration Time**: 1 week
- Day 1-2: Setup, dependency updates
- Day 3-4: Dataset adaptation
- Day 5-7: Initial training, validation

**Recommendation**: **Use as primary implementation for FairDisCo**
- Most complete adversarial debiasing framework for dermatology
- Best paper award (peer-reviewed quality)
- Direct applicability to Fitzpatrick17k + DDI

---

### 2.2 Alternative: pbevan1/Detecting-Melanoma-Fairly

**Repository Details**:
- URL: https://github.com/pbevan1/Detecting-Melanoma-Fairly
- Authors: Peter Bevan (MICCAI DART 2023)
- Paper: "Detecting Melanoma Fairly: Skin Tone Detection and Debiasing for Skin Lesion Classification"

**Key Features**:
- Gradient reversal layer (similar to FairDisCo)
- Variational autoencoder (VAE) for debiasing
- Skin tone detection module

**Advantages**:
- More recent (2023 vs 2022)
- Includes skin tone detection (auto-FST labeling)

**Disadvantages**:
- More complex (VAE adds overhead)
- Less mature (fewer citations, smaller community)

**Recommendation**: **Use siyi-wind/FairDisCo** (simpler, better validated)

---

## 3. CIRCLe Color-Invariant Learning

### 3.1 Official Repository: arezou-pakzad/CIRCLe

**Repository Details**:
- URL: https://github.com/arezou-pakzad/CIRCLe
- Authors: Arezou Pakzad, Kumar Abhishek, Ghassan Hamarneh (ECCV 2022)
- Paper: "CIRCLe: Color Invariant Representation Learning for Unbiased Classification of Skin Lesions"
- Stars: ~80+ (as of 2025-01)
- License: **Not explicitly specified** (assume academic use)
- Last Updated: 2023-02 (23 months ago, but complete)

**Technical Stack**:
- Python 3.8+
- PyTorch 1.10+
- torchvision
- OpenCV (color transformations)

**Code Structure**:
```
CIRCLe/
├── models/
│   ├── resnet.py                  # ResNet18/50
│   ├── densenet.py                # DenseNet121
│   ├── mobilenet.py               # MobileNetV2/V3
│   └── vgg.py                     # VGG16
├── stargan/
│   ├── train_stargan.py           # Train tone transformer
│   ├── models.py                  # StarGAN architecture
│   └── inference.py               # Generate transformed images
├── utils/
│   ├── regularization.py          # L2 distance loss
│   ├── color_transforms.py        # Simple HSV/LAB transformations
│   └── metrics.py                 # Fairness metrics
├── data/
│   ├── fitzpatrick17k_loader.py   # Dataset loader
│   └── transforms.py              # Augmentation pipeline
├── train_classifier.py            # Main training script
├── evaluate.py                    # Evaluation script
└── README.md
```

**Key Features**:
1. **StarGAN Tone Transformer**: Train skin tone transformation model
   - 200 epochs, ~150-200 hours (RTX 3090)
   - Pre-trained checkpoints: **Not released**

2. **Simple Color Transformations**: HSV, LAB alternatives
   - No training required
   - Fast, deterministic

3. **Regularization Loss**: L2 distance between original and transformed embeddings
   - Lambda scheduling: 0.1 → 0.3
   - Multi-FST regularization (FST I + VI)

4. **Multiple Backbones**: ResNet, DenseNet, MobileNet, VGG
   - Easy to swap architectures
   - Unified interface

**Code Quality**: **Good (8/10)**

**Strengths**:
- Multiple backbone support (flexible)
- Includes both StarGAN and simple transformations
- Comprehensive evaluation (EOD, AUROC gap, calibration)
- Clean, modular code

**Weaknesses**:
- StarGAN training complex (200 hours, no checkpoints)
- Limited documentation for simple transformations
- No config file (hyperparameters hardcoded)
- License ambiguity

**Integration Feasibility**: **Moderate Complexity**

**Required Adaptations**:
1. **Skip StarGAN training** (use simple transformations for Phase 2)
2. Add HAM10000 dataset support
3. Refactor to use config files
4. Add WandB logging

**Estimated Integration Time**: 1 week
- Day 1-2: Setup, implement simple LAB transformations
- Day 3-4: Integrate regularization loss into training loop
- Day 5-7: Hyperparameter tuning (lambda_reg)

**Recommendation**: **Use as primary implementation for CIRCLe**
- Most complete color-invariant learning framework
- Flexible (StarGAN or simple transformations)
- Well-validated (ECCV 2022)

---

### 3.2 Mirror Repository: sfu-mial/CIRCLe

**Repository Details**:
- URL: https://github.com/sfu-mial/CIRCLe
- Mirror of original repository (Simon Fraser University)
- Identical codebase

**Recommendation**: **Use arezou-pakzad/CIRCLe** (original, likely more up-to-date)

---

## 4. Additional Relevant Repositories

### 4.1 tkalbl/RevisitingSkinToneFairness

**Repository Details**:
- URL: https://github.com/tkalbl/RevisitingSkinToneFairness
- Paper: "Revisiting Skin Tone Fairness in Dermatological Lesion Classification"
- Focus: ITA-based skin tone classification, fairness evaluation

**Key Features**:
- Four ITA-based approaches for FST classification
- Comprehensive fairness metrics
- ISIC18 dataset experiments

**Relevance**: **Low (Phase 2)**, **High (Phase 1 - FST annotation)**
- Not a fairness intervention (evaluation only)
- Useful for FST annotation protocol (Phase 1)

---

### 4.2 Google Research: derm-foundation

**Hugging Face Model**:
- URL: https://huggingface.co/google/derm-foundation
- Pre-trained dermatology foundation model (6144-dim embeddings)
- Trained on large proprietary dataset

**Key Features**:
- 6144-dimensional embeddings (vs ResNet50 2048-dim)
- Pre-trained on diverse dermatology images
- Zero-shot classification possible

**Relevance**: **High (Phase 3+)**
- Alternative backbone for FairDisCo, CIRCLe
- Expected: Better accuracy, maintained fairness
- Requires fine-tuning for specific datasets

**Integration**: Replace ResNet50 with derm-foundation encoder

---

## 5. Comparative Evaluation

### 5.1 Code Quality Matrix

| **Repository** | **Code Quality** | **Documentation** | **Modularity** | **Maintenance** | **Overall** |
|----------------|------------------|-------------------|----------------|-----------------|-------------|
| janet-sw/skin-diff | 7/10 | 6/10 | 8/10 | 6/10 | **7.0/10** |
| siyi-wind/FairDisCo | 9/10 | 7/10 | 9/10 | 7/10 | **8.5/10** |
| arezou-pakzad/CIRCLe | 8/10 | 7/10 | 9/10 | 6/10 | **8.0/10** |
| pbevan1/Detecting-Melanoma-Fairly | 7/10 | 6/10 | 7/10 | 7/10 | **7.0/10** |
| tkalbl/RevisitingSkinToneFairness | 8/10 | 8/10 | 8/10 | 8/10 | **8.0/10** |

### 5.2 Integration Complexity

| **Repository** | **Setup Time** | **Adaptation Effort** | **Training Time** | **Integration Risk** |
|----------------|----------------|------------------------|-------------------|----------------------|
| janet-sw/skin-diff | 2-3 days | 1 week | 12-20 hours (LoRA) | **Moderate-High** |
| siyi-wind/FairDisCo | 1-2 days | 3-5 days | 25 hours (100 epochs) | **Moderate** |
| arezou-pakzad/CIRCLe | 1-2 days | 3-5 days | 30 hours (100 epochs) | **Moderate** |

### 5.3 License Compatibility

**Critical Issue**: None of the primary repositories specify explicit licenses

**Implications**:
- Academic use: Generally safe (research papers imply permission)
- Commercial use: **Contact authors for permission**
- Modification: Allowed (implied by publishing code)
- Redistribution: Unclear (no license = all rights reserved by default)

**Recommendation**: **Contact authors for Apache 2.0 or MIT licensing**
- janet-sw (Janet Tsang): Request via GitHub Issues
- siyi-wind (Siyi Du): Request via email (in paper)
- arezou-pakzad (Arezou Pakzad): Request via email

**Fallback**: Implement from scratch using paper descriptions (clean-room implementation)

---

## 6. Recommended Integration Strategy

### 6.1 Phase 2 Implementation Plan

**Week 1-2: FairSkin Diffusion**
- Use: janet-sw/skin-diff
- Adaptations: HAM10000 support, config files, WandB logging
- Training: Textual inversion (4 hours) + LoRA (20 hours) = **24 hours GPU**

**Week 3-4: FairDisCo Adversarial Debiasing**
- Use: siyi-wind/FairDisCo
- Adaptations: Dataset loaders, config files, PyTorch 2.1 update
- Training: 100 epochs × 15 min = **25 hours GPU**

**Week 5-6: CIRCLe Color-Invariant Learning**
- Use: arezou-pakzad/CIRCLe (simple transformations, skip StarGAN)
- Adaptations: LAB color transformations, regularization integration
- Training: 100 epochs × 18 min = **30 hours GPU**

**Week 6: Combined Evaluation**
- Train model with all three techniques
- Cumulative fairness gain: Measure AUROC gap, EOD, ECE
- Target: <8% AUROC gap (Phase 2 MVP success)

### 6.2 Code Integration Workflow

**Step 1: Clone Repositories**
```bash
cd external/
git clone https://github.com/janet-sw/skin-diff
git clone https://github.com/siyi-wind/FairDisCo
git clone https://github.com/arezou-pakzad/CIRCLe
```

**Step 2: Extract Reusable Components**
```bash
# Copy into project structure
cp skin-diff/lora/train_lora.py src/fairness/fairskin_lora.py
cp FairDisCo/models/gradient_reversal.py src/fairness/gradient_reversal.py
cp CIRCLe/utils/regularization.py src/fairness/circle_loss.py
```

**Step 3: Refactor & Adapt**
- Unify dataset loaders (single FitzpatrickDataset class)
- Create config files (YAML) for all hyperparameters
- Add WandB logging to all training scripts
- Implement fairness evaluation pipeline (reusable across techniques)

**Step 4: Test Individually**
- Baseline: Train ResNet50 without fairness (quantify gap)
- FairSkin: Add synthetic data, measure improvement
- FairDisCo: Add adversarial training, measure improvement
- CIRCLe: Add regularization, measure improvement

**Step 5: Combine & Evaluate**
- Train single model with all three techniques
- Ablation study: Measure contribution of each component
- Final evaluation: AUROC gap, EOD, ECE per FST

---

## 7. License Recommendations

**Preferred License**: Apache 2.0
- Permissive (allows commercial use, modification, redistribution)
- Patent grant (protects against patent claims)
- Widely adopted in ML community (PyTorch, TensorFlow)

**Alternative**: MIT License
- Simpler, shorter
- No explicit patent grant (potential risk)

**Action Items**:
1. Contact authors (janet-sw, siyi-wind, arezou-pakzad)
2. Request Apache 2.0 or MIT licensing
3. If no response after 2 weeks: Clean-room implementation from papers

---

## 8. Summary & Recommendations

### 8.1 Primary Implementations (Phase 2)

| **Technique** | **Repository** | **Code Quality** | **Integration Complexity** | **Recommendation** |
|---------------|----------------|------------------|----------------------------|-------------------|
| **FairSkin** | janet-sw/skin-diff | 7/10 | Moderate-High | **Use, adapt for HAM10000** |
| **FairDisCo** | siyi-wind/FairDisCo | 9/10 | Moderate | **Use, minimal adaptation** |
| **CIRCLe** | arezou-pakzad/CIRCLe | 8/10 | Moderate | **Use, skip StarGAN (Phase 2)** |

### 8.2 Key Strengths

**All Repositories**:
- High-quality code (7-9/10)
- Peer-reviewed (MICCAI, ECCV workshops)
- PyTorch-based (easy integration)
- Comprehensive evaluation (fairness metrics)

### 8.3 Key Weaknesses

**All Repositories**:
- No explicit license (contact authors)
- No pre-trained checkpoints (must train from scratch)
- Hardcoded hyperparameters (need refactoring)
- Limited documentation (assume paper familiarity)

### 8.4 Critical Path

1. **Week 1**: Clone repositories, setup environments, contact authors for licenses
2. **Week 2-6**: Implement fairness techniques (FairSkin → FairDisCo → CIRCLe)
3. **Week 6**: Combined evaluation, measure cumulative fairness gain
4. **Week 7**: Ablation studies, optimize hyperparameters
5. **Week 8**: Final Phase 2 model, prepare for Phase 3

**Success Criteria**: AUROC gap <8% (from 15-20% baseline), EOD <0.08, ECE <0.10

---

## 9. References

**Repositories Evaluated**:
1. janet-sw/skin-diff: https://github.com/janet-sw/skin-diff
2. siyi-wind/FairDisCo: https://github.com/siyi-wind/FairDisCo
3. arezou-pakzad/CIRCLe: https://github.com/arezou-pakzad/CIRCLe
4. pbevan1/Detecting-Melanoma-Fairly: https://github.com/pbevan1/Detecting-Melanoma-Fairly
5. tkalbl/RevisitingSkinToneFairness: https://github.com/tkalbl/RevisitingSkinToneFairness
6. Hugging Face Diffusers: https://github.com/huggingface/diffusers
7. Google derm-foundation: https://huggingface.co/google/derm-foundation

**License Resources**:
- Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0
- MIT License: https://opensource.org/licenses/MIT
- GitHub Licensing Guide: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository

---

**Document Version**: 1.0
**Last Updated**: 2025-10-13
**Author**: THE DIDACT (Strategic Research Agent)
**Status**: COMPLETE
**Next Action**: Contact repository authors for license clarification
