# Fairness-Aware AI for Skin Cancer Detection: Implementation Roadmap

**Research Foundation**
This project is inspired by and builds upon the comprehensive survey:

> Flores, J., & Alzahrani, N. (2025). *AI Skin Cancer Detection Across Skin Tones: A Survey of Experimental Advances, Fairness Techniques, and Dataset Limitations*. Submitted to *Computers* (MDPI).

**Authors**: Jasmin Flores & Dr. Nabeel Alzahrani
**Institution**: School of Computer Science & Engineering, California State University, San Bernardino, USA

---

## Project Vision

Develop a production-grade, fairness-aware AI system for skin cancer detection that achieves equitable diagnostic performance across all Fitzpatrick skin types (FST I-VI), addressing the critical healthcare disparity where existing models show 15-30% performance drops on darker skin tones.

**Core Principles**:
1. **Fairness-First Development**: Equity across skin tones is not an afterthought—it's embedded from Phase 1
2. **Evidence-Based Design**: Every architectural and methodological decision grounded in peer-reviewed research
3. **Clinical Viability**: Target performance benchmarks from deployed systems (NHS DERM: 97% sensitivity across all FST)
4. **Open Science**: Transparent methodology, reproducible experiments, public model cards with subgroup metrics
5. **Ethical AI**: Patient co-design, informed consent, continuous fairness monitoring

---

## Implementation Phases

### **Phase 1: Foundation (Weeks 1-4)**

**Objective**: Establish baseline infrastructure, quantify fairness gap, validate evaluation framework

**Key Activities**:
1. **Dataset Acquisition**
   - Primary: Fitzpatrick17k, DDI (Diverse Dermatology Images), MIDAS, SCIN
   - Baseline: HAM10000, ISIC 2019 (for comparison)
   - Target FST distribution: Minimum 25% FST IV-VI (vs <5% in standard datasets)

2. **Baseline Model Training**
   - ResNet50, EfficientNet B4 (transfer learning)
   - Quantify fairness gap: AUROC per FST group
   - Expected: 15-20% AUROC drop for FST V-VI (literature benchmark)

3. **Evaluation Framework**
   - Metrics: AUROC, Sensitivity, Specificity, Equal Opportunity Difference (EOD), Expected Calibration Error (ECE)
   - Per-FST reporting: Disaggregate ALL metrics by skin tone
   - Visualization: ROC curves per FST, calibration plots, confusion matrices

4. **Tone Annotation Pipeline**
   - Monk Skin Tone (MST) scale (10-point, superior to Fitzpatrick 6-point)
   - Dual annotation (2 annotators per image, adjudication protocol)
   - ITA (Individual Typology Angle) validation

**Success Criteria**:
- Dataset access confirmed (4/5 datasets)
- Baseline AUROC gap quantified: 15-20% (FST I-III vs V-VI)
- Evaluation pipeline operational: Automated subgroup metrics
- 5,000 images annotated with MST labels

**Deliverables**:
- `src/data/datasets.py`: Dataset loaders (Fitzpatrick17k, DDI, HAM10000)
- `src/evaluation/metrics.py`: Fairness metrics (AUROC per FST, EOD, ECE)
- `experiments/baseline/`: Baseline model training scripts + results
- `docs/datasets.md`: Dataset documentation with FST distributions

---

### **Phase 2: Fairness MVP (Weeks 5-10)**

**Objective**: Implement core fairness techniques, reduce AUROC gap to <8%

**Key Techniques** (Priority Order):

1. **Data-Level: FairSkin Diffusion Augmentation**
   - Train tone-conditioned diffusion model (Stable Diffusion architecture)
   - Generate 60,000 synthetic images with balanced FST distribution
   - Quality validation: FID <20, LPIPS <0.1, expert dermatologist review
   - **Expected Impact**: +18-21% AUROC improvement for FST VI (literature benchmark)

2. **Algorithm-Level: FairDisCo Adversarial Debiasing**
   - Auxiliary discriminator: Predict FST from latent embeddings
   - Adversarial training: Maximize classification accuracy, minimize FST predictability
   - Contrastive loss: Pull same-diagnosis, different-FST embeddings together
   - **Expected Impact**: 65% EOD reduction (Equal Opportunity Difference)

3. **Algorithm-Level: CIRCLe Color-Invariant Learning**
   - Regularization: Latent embeddings invariant across tone transformations
   - Prioritize lesion morphology (shape, border) over pixel color
   - **Expected Impact**: Improved calibration (ECE reduction 3-5%), better OOD generalization

4. **Metadata Fusion**
   - Attention-based encoder: Integrate FST, age, anatomical site
   - Late fusion with CNN features
   - **Expected Impact**: +5-10% AUROC for FST IV-VI (Muffin architecture benchmark)

**Success Criteria**:
- AUROC gap reduced: <8% (from baseline 15-20%)
- EOD <0.08 (Equal Opportunity Difference across FST groups)
- Overall accuracy maintained: >88%
- Synthetic dataset quality validated: FID <20, expert scores >5/7

**Deliverables**:
- `src/fairness/fairskin_diffusion.py`: Tone-conditioned diffusion model
- `src/fairness/fairdisco.py`: Adversarial debiasing implementation
- `src/fairness/circle.py`: Color-invariant regularization
- `src/models/metadata_fusion.py`: Attention-based metadata encoder
- `experiments/fairness_mvp/`: Training scripts + comparative results

---

### **Phase 3: Hybrid Architecture (Weeks 11-18)**

**Objective**: Implement state-of-the-art hybrid model, achieve 93%+ accuracy with <4% gap

**Architecture: ConvNeXtV2-Swin Transformer Hybrid**

**Components**:
1. **ConvNeXtV2-Base** (first 2 stages)
   - Local feature extraction (lesion borders, texture)
   - 36.44M parameters, efficient (80-100ms inference)

2. **Swin Transformer V2 Small** (later stages)
   - Global attention (tone-invariant contextual features)
   - Hierarchical windows (7×7, 14×14, 28×28)

3. **Attentional Feature Fusion (AFF)**
   - Merge ConvNeXt + Swin features
   - Learned attention weights per branch

4. **Metadata Encoder**
   - Attention-MLP: FST, age, anatomical site → 64-dim embedding
   - Late fusion with image features

**Training Strategy**:
- Pre-train: Synthetic-augmented dataset (Phase 2 output)
- Fine-tune: Real data (Fitzpatrick17k + DDI + MIDAS)
- Multi-task loss: Classification + adversarial + color-invariant + metadata
- Loss weights: [1.0, 0.3, 0.2, 0.1] (classification dominant)

**Hyperparameters** (from literature synthesis):
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingWarmRestarts
- Batch size: 32
- Epochs: 100
- Augmentation: RandAugment + FairSkin synthetic

**Success Criteria**:
- Overall accuracy: 93-95% (ISIC 2019 benchmark)
- AUROC gap: <4% (FST I-III vs FST V-VI)
- EOD: <0.05, ECE: <0.08 (ALL FST groups)
- Inference time: <120ms (clinical acceptability)

**Deliverables**:
- `src/models/hybrid_convnext_swin.py`: Hybrid architecture implementation
- `src/models/attention_fusion.py`: AFF module
- `experiments/hybrid_architecture/`: Training pipeline + ablation studies
- `docs/architecture.md`: Technical documentation with diagrams

---

### **Phase 4: Production Hardening (Weeks 19-24)**

**Objective**: Optimize for edge deployment, add explainability, prepare for clinical validation

**Key Activities**:

1. **FairPrune Edge Optimization**
   - Analyze activation saliency per FST subgroup
   - Prune 30-40% of filters contributing to light-tone overfitting
   - Fine-tune pruned model
   - **Target**: <50MB model, <80ms inference on mobile, maintained fairness

2. **Quantization**
   - INT8 quantization (TensorFlow Lite, ONNX Runtime)
   - Platform-specific: Core ML (iOS), TF Lite (Android)
   - **Target**: <12MB model (75% size reduction), <1% accuracy loss

3. **Grad-CAM Explainability**
   - Heatmap overlays: Show model attention regions
   - Clinical feature alignment: ABCD rule (Asymmetry, Border, Color, Diameter)
   - Tone-specific failure mode analysis: Qualitative review per FST

4. **Model Card Documentation**
   - Dataset composition: FST distribution, disease categories
   - Subgroup metrics: AUROC, Sensitivity, Specificity per FST
   - Limitations: Intermediate tone subjectivity, OOD performance
   - Intended use: Teledermatology decision support (not standalone diagnosis)

5. **Fairness Monitoring Dashboard**
   - Real-time metrics: AUROC, EOD, ECE per FST (updated daily)
   - Model drift detection: KL divergence, population stability index
   - Alert system: Email/SMS when fairness thresholds exceeded

**Success Criteria**:
- Edge model deployed: <50MB, <80ms inference on mobile
- Grad-CAM validated: Dermatologist feedback (qualitative)
- Model card complete: 10+ pages, public disclosure
- Monitoring dashboard operational: Real-time fairness tracking

**Deliverables**:
- `src/optimization/fairprune.py`: Pruning implementation
- `src/explainability/gradcam.py`: Grad-CAM visualization
- `docs/model_card.md`: Comprehensive model documentation
- `scripts/monitoring_dashboard.py`: Fairness monitoring (Streamlit/Gradana)

---

### **Phase 5: Deployment & Validation (Weeks 25-32)**

**Objective**: Deploy to teledermatology platform, conduct prospective clinical trial

**Key Activities**:

1. **Teledermatology API**
   - RESTful API: OpenAPI 3.0 specification
   - SDK: Python, JavaScript clients
   - EHR integration: HL7 FHIR messaging
   - **Deployment**: Cloud (AWS/Azure/GCP), auto-scaling (100+ concurrent users)

2. **Prospective Clinical Trial**
   - Design: Multi-site (2-3 hospitals), 500+ patients, all FST types
   - Comparator: Dermatologist diagnosis (gold standard: biopsy)
   - Primary outcome: Sensitivity and specificity per FST (non-inferiority: 5% margin)
   - Secondary: Calibration (ECE), time-to-diagnosis, patient satisfaction
   - Duration: 4-6 months

3. **Continual Learning Pipeline**
   - Weekly model updates: Incremental learning on new labeled data
   - Bayesian generative approach: Store statistics, not raw images (privacy)
   - Drift monitoring: Trigger retraining when EOD or ECE exceed thresholds

4. **Regulatory Documentation**
   - FDA: De Novo submission (breakthrough device pathway)
   - EU: CE marking (MDR Class IIa/IIb or Class III)
   - Clinical data: Prospective trial results
   - Risk analysis: FMEA (Failure Mode and Effects Analysis)

**Success Criteria**:
- API deployed: 99.9% uptime, <200ms response time
- Clinical trial completed: 500+ patients, all FST represented
- Non-inferiority demonstrated: Sensitivity >95%, Specificity >80% for ALL FST groups
- Regulatory submission prepared: FDA De Novo or EU CE application

**Deliverables**:
- `src/api/`: RESTful API implementation (FastAPI/Flask)
- `src/continual_learning/`: Incremental learning pipeline
- `docs/clinical_trial_protocol.md`: Trial design, statistical analysis plan
- `docs/regulatory/`: FDA submission documentation

---

## Target Performance Benchmarks

**Literature-Derived Targets** (from 100+ papers surveyed):

| **Metric** | **FST I-III** | **FST IV-VI** | **Gap** | **Benchmark Source** |
|------------|---------------|---------------|---------|---------------------|
| **AUROC** | 91-93% | 89-92% | <4% | NHS DERM (deployed), BiaslessNAS |
| **Sensitivity (Melanoma)** | >95% | >95% | 0% | NHS DERM (97% across all FST) |
| **Specificity** | >80% | >80% | 0% | Clinical acceptability threshold |
| **EOD** | --- | --- | <0.05 | Fairness standard (5% max disparity) |
| **ECE** | <0.08 | <0.08 | 0% | Calibration quality (clinical trust) |

**Baseline (No Fairness)**:
- ResNet50 on ISIC 2020: 91.3% (FST I-III), 75.4% (FST V-VI) = **-15.9% gap**
- InceptionV3 on HAM10000: 90.1% (FST I-III), 78.3% (FST V-VI) = **-11.8% gap**

**Phase 2 Target (Fairness MVP)**:
- AUROC gap: <8% (50% reduction from baseline)
- EOD: <0.08

**Phase 3 Target (Hybrid Architecture)**:
- AUROC gap: <4% (match BiaslessNAS, NHS DERM)
- EOD: <0.05, ECE: <0.08 (all FST groups)

**Phase 5 Target (Clinical Deployment)**:
- Non-inferiority to dermatologist: Within 5% sensitivity, 5% specificity
- Patient satisfaction: >80% (NHS DERM achieved 85%)

---

## Key Datasets

**Primary Training Datasets** (FST Diversity):

1. **Fitzpatrick17k**: 16,577 images, ~8% FST V-VI, dual annotation (FST + ITA)
2. **DDI (Stanford)**: 656 images, 34% FST V-VI, clinician-rated (gold standard)
3. **MIDAS**: Biopsy-confirmed, ~28% FST V-VI, multi-modal (clinical + dermoscopic)
4. **SCIN (Google)**: 10,000+ images, ~33% FST V-VI, triple annotation (eFST, eMST, CST)
5. **SkinCon (MIT)**: Built on Fitzpatrick17k + DDI, ~30% FST V-VI, meta-concept tags

**Baseline Datasets** (For Comparison):
- HAM10000: 10,015 images, <5% FST V-VI (high quality, tone-imbalanced)
- ISIC 2020: 33,126 images, <3% FST V-VI (no tone labels, benchmark)

**Synthetic Augmentation**:
- FairSkin/DermDiff: Generate 60,000 synthetic images with balanced FST distribution

---

## Fairness Techniques Summary

**Three-Tier Fairness Methodology**:

| **Stage** | **Technique** | **Expected Impact** | **Complexity** |
|-----------|---------------|---------------------|----------------|
| **Data-Level** | FairSkin Diffusion Augmentation | +18-21% FST VI AUROC | High (48-72 hrs GPU) |
| **Algorithm-Level** | FairDisCo Adversarial Debiasing | 65% EOD reduction | Moderate |
| **Algorithm-Level** | CIRCLe Color-Invariant Loss | 3-5% ECE reduction | Moderate |
| **Post-Processing** | FairPrune Selective Pruning | 3-6% AUROC gap reduction | Low |

**Trade-offs**:
- Accuracy cost: 1-3% (mitigated by contrastive loss in FairDisCo)
- Computational cost: Diffusion training (48-72 hrs), NAS (7-14 days if used)
- Calibration: May degrade 5-10% ECE with synthetic data (mitigate: temperature scaling)

---

## Technology Stack

**Deep Learning**:
- PyTorch 2.0+ (primary framework)
- timm (ConvNeXt, Swin Transformer pre-trained models)
- Diffusers (Hugging Face, for FairSkin diffusion)

**Data Science**:
- NumPy, pandas, scikit-learn
- OpenCV, Albumentations (image preprocessing, augmentation)

**Fairness**:
- Fairlearn (fairness metrics)
- Custom implementations: FairDisCo, CIRCLe, FairPrune

**Evaluation & Visualization**:
- Matplotlib, Seaborn (plots, calibration curves)
- TensorBoard (training monitoring)
- Grad-CAM (explainability)

**Deployment**:
- FastAPI (RESTful API)
- TensorFlow Lite / ONNX Runtime (edge deployment)
- Docker (containerization)
- AWS/Azure/GCP (cloud hosting)

---

## Regulatory Pathway

**FDA (United States)**:
- **Pathway**: De Novo (Class II) or 510(k) if predicate exists
- **Precedent**: DermaSensor (FDA cleared Jan 2024, breakthrough device)
- **Requirements**: Prospective multi-site trial (500+ patients), subgroup metrics, software documentation (IEC 62304)

**EU (Europe)**:
- **Pathway**: CE marking (MDR 2017/745)
- **Classification**: Class IIa/IIb (decision support) or Class III (autonomous diagnosis)
- **Precedent**: DERM (CE Class III approved 2024, 99.8% accuracy)

**Timeline**:
- FDA De Novo: 18-24 months (with breakthrough designation: 12-18 months)
- EU CE: 6-18 months (depends on class)

---

## Ethical Considerations

**Patient Consent**:
- Transparent disclosure: Model uses skin tone for fairness-aware training
- Opt-out mechanism: Tone-blind inference available (with performance caveat)

**Co-Design**:
- Patient advisory board: Diverse FST representation
- Iterative feedback: Incorporate patient concerns (privacy, bias, transparency)

**Model Card Transparency**:
- Dataset composition: FST distribution, annotation methods
- Subgroup metrics: AUROC, Sensitivity, Specificity per FST
- Limitations: Intermediate tone subjectivity, OOD performance, synthetic data artifacts

**Post-Market Surveillance**:
- Quarterly fairness audits: EOD, ECE per FST
- Incident reporting: Misdiagnosis cases, tone-related failures
- Continual learning: Model updates based on real-world feedback

---

## Success Metrics

**Technical**:
- AUROC gap <4% (FST I-III vs FST V-VI)
- EOD <0.05 (Equal Opportunity Difference)
- ECE <0.08 per FST (Expected Calibration Error)
- Sensitivity >95% for melanoma (ALL FST groups)
- Inference time <100ms (edge deployment)

**Clinical**:
- Non-inferiority to dermatologist (within 5% sensitivity, 5% specificity)
- Patient satisfaction >80%
- Time-to-diagnosis reduction (vs standard referral pathway)
- Cost-effectiveness (QALY analysis)

**Regulatory**:
- FDA clearance or EU CE marking
- Model card with subgroup metrics (public disclosure)
- Post-market surveillance plan approved

---

## References

**Foundational Survey**:
- Flores, J., & Alzahrani, N. (2025). AI Skin Cancer Detection Across Skin Tones: A Survey of Experimental Advances, Fairness Techniques, and Dataset Limitations. *Computers* (MDPI). [Submitted]

**Key Techniques**:
- **FairSkin**: Ju et al. (2024). Diffusion-based synthetic augmentation for skin tone fairness.
- **FairDisCo**: Daneshjou et al. (2022). Adversarial debiasing with contrastive learning. *Science Advances*, 8(25), eabq6147.
- **CIRCLe**: Pakzad et al. (2022). Color-invariant representation learning. *ECCV 2022*.
- **BiaslessNAS**: Pacal et al. (2025). Neural architecture search for fairness. *Biomedical Signal Processing and Control*, 104, 107627.

**Deployed Systems**:
- **NHS DERM**: Skin Analytics (2022-2023). 9,649 patients, 97% melanoma sensitivity, 85% patient satisfaction.
- **DermaSensor**: FDA cleared Jan 2024. 96% malignancy sensitivity in primary care.

**Datasets**:
- **Fitzpatrick17k**: Groh et al. (2021). 16,577 images with FST labels.
- **DDI**: Daneshjou et al. (2022). 656 diverse dermatology images.
- **MIDAS**: Stanford AIMI. Multimodal biopsy-confirmed dataset.

---

## Acknowledgments

This project builds upon the comprehensive survey by **Jasmin Flores** and **Dr. Nabeel Alzahrani** from California State University, San Bernardino. Their systematic analysis of 100+ experimental studies (2022-2025) on fairness-aware skin cancer detection provided the foundational research that informs every phase of this implementation roadmap.

Special recognition to:
- **The research community**: Authors of FairSkin, FairDisCo, CIRCLe, BiaslessNAS, and other fairness techniques
- **Dataset creators**: Fitzpatrick17k, DDI, MIDAS, SCIN, SkinCon teams for enabling diverse-tone research
- **Clinical pioneers**: NHS DERM, DermaSensor teams for demonstrating real-world viability

**Project Status**: Foundation Phase
**Last Updated**: 2025-10-13
**Framework**: MENDICANT_BIAS Multi-Agent System
**License**: Apache 2.0

---

**Strategic Research**: the_didact
**Core Development**: hollowed_eyes
**QA & Security**: loveless
**DevOps & Deployment**: zhadyz
**Supreme Orchestrator**: mendicant_bias
