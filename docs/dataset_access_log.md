# Dataset Access Log

**Mission**: Phase 1 Dataset Acquisition for Fairness-Aware Skin Cancer Detection
**Initiated**: 2025-10-13
**Status**: IN PROGRESS
**Agent**: the_didact (MENDICANT_BIAS framework)

---

## Primary Datasets Status

### 1. HAM10000 (Human Against Machine with 10,000 Dermoscopic Images)

**Status**: ✅ ACCESS CONFIRMED - PUBLIC DATASET

**Details**:
- **Size**: 10,015 dermoscopic images
- **Classes**: 7 diagnostic categories (melanoma, basal cell carcinoma, melanocytic nevi, etc.)
- **Metadata**: Age, sex, localization
- **Skin Tone Limitation**: <5% FST V-VI (majority lighter skin)

**Access Methods**:
1. **Harvard Dataverse** (Primary):
   - URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
   - DOI: 10.7910/DVN/DBW86T
   - License: Non-commercial use only
   - Method: Direct download (requires Terms of Use confirmation)

2. **Kaggle** (Alternative):
   - URL: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
   - Method: Kaggle account required, free download

3. **ISIC Archive API**:
   - URL: https://isic-archive.com/api/v1
   - Method: Programmatic access via API calls

**Download Action**: Use Kaggle CLI or Harvard Dataverse direct download
**Target Directory**: `data/raw/ham10000/`
**Next Steps**:
- Download dataset (10,015 images + metadata CSV)
- Verify file integrity
- Parse metadata for FST distribution analysis

---

### 2. ISIC 2019 Challenge Dataset

**Status**: ✅ ACCESS CONFIRMED - PUBLIC DATASET

**Details**:
- **Size**: 25,331 training images (ISIC 2019), 33,126 total (ISIC 2020)
- **Classes**: 8 diagnostic categories
- **Metadata**: Patient demographics, lesion location
- **Skin Tone Limitation**: <3% FST V-VI, no explicit FST labels

**Access Methods**:
1. **ISIC Challenge Website** (Primary):
   - URL: https://challenge.isic-archive.com/data/
   - Method: Direct download (training data ~2.6GB)
   - Registration: Free account required

2. **AWS Open Data Registry**:
   - URL: https://registry.opendata.aws/isic-archive/
   - Method: S3 bucket access (public)

3. **Hugging Face** (Fed-ISIC-2019):
   - URL: https://huggingface.co/datasets/flwrlabs/fed-isic2019
   - Method: Datasets library download

**Download Action**: ISIC Challenge direct download for ISIC 2019 training set
**Target Directory**: `data/raw/isic2019/`
**Next Steps**:
- Download ISIC 2019 training images + ground truth CSV
- Parse metadata (age, sex, anatomical site)
- Note: Will require FST annotation in Phase 1 (annotation protocol in progress)

---

### 3. Fitzpatrick17k

**Status**: ⏳ ACCESS REQUEST REQUIRED

**Details**:
- **Size**: 16,577 clinical images
- **FST Distribution**: ~8% FST V-VI (better than baseline but still imbalanced)
- **Metadata**: Fitzpatrick skin type (dual annotation), diagnosis, ITA values
- **Source**: DermaAmin and Atlas Dermatologico atlases

**Access Methods**:
1. **GitHub Repository** (Metadata only):
   - URL: https://github.com/mattgroh/fitzpatrick17k
   - Available: Fitzpatrick17k.csv (annotations)

2. **Image Access** (Request Required):
   - Method: Fill out access request form (linked in GitHub README)
   - Contact: Matthew Groh (MIT researcher)
   - Alternative: Download from original source URLs (provided in CSV)

**Citation**:
- Groh, M., Harris, C., Daneshjou, R., et al. (2021). Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset. arXiv:2104.09957

**Access Request Draft**:

```
Subject: Research Access Request - Fitzpatrick17k Dataset

Dear Dr. Groh,

I am writing to request access to the Fitzpatrick17k dataset for academic research purposes.

Project: Fairness-Aware AI for Skin Cancer Detection Across All Fitzpatrick Skin Types
Institution: [California State University, San Bernardino / Independent Research]
Principal Investigator: [Dr. Nabeel Alzahrani / Student: Jasmin Flores]

Research Objective:
We are developing a fairness-aware skin cancer detection system to address the 15-30%
performance gap observed in existing AI models for darker skin tones (FST IV-VI). Our
project builds upon the comprehensive survey "AI Skin Cancer Detection Across Skin Tones"
(Flores & Alzahrani, 2025) and aims to implement state-of-the-art fairness techniques
(FairSkin diffusion, FairDisCo adversarial debiasing, CIRCLe color-invariant learning).

Dataset Usage:
- Training fairness-aware deep learning models (hybrid ConvNeXt-Swin Transformer architecture)
- Stratified evaluation across FST groups (quantifying AUROC, EOD, ECE per skin tone)
- Comparative analysis with baseline datasets (HAM10000, ISIC) to demonstrate fairness improvement

We commit to:
1. Using the dataset solely for non-commercial academic research
2. Proper attribution in all publications and presentations
3. Sharing subgroup performance metrics transparently (model card documentation)
4. Not re-distributing the dataset without authorization

Timeline: Phase 1 baseline training (4 weeks), full project completion (24 weeks)

Contact Information:
- Email: [your_email@csusb.edu]
- GitHub: https://github.com/[username]/skin-cancer-fairness
- Project Framework: MENDICANT_BIAS multi-agent system

Thank you for advancing fairness-aware dermatology AI research. Your Fitzpatrick17k
dataset is foundational to addressing healthcare disparities.

Sincerely,
[Your Name]
[Affiliation]
```

**Target Directory**: `data/raw/fitzpatrick17k/`
**Next Steps**:
- Submit access request via GitHub form
- Follow up in 3-5 business days
- Alternative: Use CSV URLs to download from original atlases (if permitted)

---

### 4. DDI (Diverse Dermatology Images)

**Status**: ✅ ACCESS AVAILABLE - RESEARCH USE AGREEMENT REQUIRED

**Details**:
- **Size**: 656 images (570 unique patients)
- **FST Distribution**: 34% FST V-VI (EXCELLENT diversity)
- **Metadata**: Pathologically confirmed diagnoses, clinician-rated FST (gold standard)
- **Source**: Stanford Clinics (2010-2020 retrospective selection)

**Access Methods**:
1. **Stanford AIMI Shared Datasets Portal**:
   - URL: https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images
   - URL: https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965
   - DOI: https://doi.org/10.71718/kqee-3z39
   - License: Research Use Agreement (non-commercial, no re-identification)

**Research Use Agreement Key Terms**:
- Personal, non-commercial research only
- No commercial use, sale, or monetization
- No attempt to re-identify individual data subjects
- Indemnification of Stanford from claims/damages

**Access Action**:
- Navigate to Stanford AIMI portal
- Accept Research Use Agreement
- Download dataset directly (no institutional approval needed for research use)

**Citation**:
- Daneshjou, R., Barata, C., Betz-Stablein, B., et al. (2022). Disparities in dermatology AI performance on a diverse, curated clinical image set. Science Advances, 8(25), eabq6147.

**Target Directory**: `data/raw/ddi/`
**Next Steps**:
- Accept Research Use Agreement on Stanford AIMI portal
- Download 656 images + metadata
- Verify pathology-confirmed labels
- Prioritize for fairness evaluation (highest FST V-VI representation)

---

### 5. MIDAS (MRA-MIDAS: Multimodal Image Dataset for AI-based Skin Cancer)

**Status**: ✅ ACCESS AVAILABLE - STANFORD AIMI PORTAL

**Details**:
- **Size**: Dual-center, prospectively recruited dataset
- **Modalities**: Paired dermoscopic + clinical images
- **FST Distribution**: ~28% FST V-VI (estimated based on Stanford diversity metrics)
- **Metadata**: Patient-level clinical metadata, histopathologic confirmation
- **Unique Feature**: Prospective recruitment (higher real-world fidelity vs retrospective)

**Access Methods**:
1. **Stanford AIMI Portal**:
   - URL: https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer
   - DOI: https://doi.org/10.71718/15nz-jv40
   - License: Non-commercial research use

**Citation**:
- McCoy, L. G., Naik, B., Saunders, H., et al. (2024). Multimodal Image Dataset for AI-based Skin Cancer (MIDAS) Benchmarking. medRxiv. DOI: 10.1101/2024.06.27.24309562

**Access Action**:
- Navigate to Stanford AIMI portal
- Accept Research Use Agreement (same as DDI)
- Download multimodal dataset (dermoscopic + clinical pairs)

**Target Directory**: `data/raw/midas/`
**Next Steps**:
- Download dataset from Stanford AIMI
- Parse multimodal structure (separate dermoscopic/clinical folders)
- Leverage clinical images for metadata fusion architecture (Phase 3)

---

### 6. SCIN (Skin Condition Image Network)

**Status**: ✅ ACCESS CONFIRMED - OPEN GITHUB REPOSITORY

**Details**:
- **Size**: 10,000+ images
- **FST Distribution**: ~33% FST V-VI, balanced FST distribution (major advantage)
- **Metadata**: Self-reported demographics, symptoms, dermatologist labels, estimated FST (eFST), estimated MST (eMST)
- **Source**: Crowdsourced from US internet users via Google Search Ads
- **Unique Feature**: Real-world, in-the-wild images (not clinical dermoscopy)

**Access Methods**:
1. **GitHub Repository**:
   - URL: https://github.com/google-research-datasets/scin
   - License: Open access for research, education, development
   - Method: Git clone or direct download

**Key Features**:
- Dermatologist estimates of Fitzpatrick Skin Type (eFST)
- Layperson labeler estimates of Monk Skin Tone (eMST)
- Common allergic, inflammatory, and infectious conditions (not just tumors)
- Crowdsourced with informed consent

**Citation**:
- Jain, A., Lipman, M., Liu, Y., et al. (2024). Crowdsourcing Dermatology Images with Google Search Ads: Creating a Real-World Skin Condition Dataset. arXiv:2402.18545

**Access Action**: Git clone repository
**Target Directory**: `data/raw/scin/`
**Next Steps**:
- Clone GitHub repository: `git clone https://github.com/google-research-datasets/scin.git`
- Review dataset schema (dataset_schema.md in repo)
- Explore scin_demo.ipynb for loading examples
- Prioritize for Phase 2 training (excellent FST balance + real-world diversity)

---

## Synthetic Augmentation Datasets (Phase 2)

### 7. FairSkin / DermDiff Synthetic Generation

**Status**: ⏳ IMPLEMENTATION RESEARCH IN PROGRESS

**Objective**: Generate 60,000 synthetic images with balanced FST distribution

**Models Identified**:

1. **FairSkin** (Oct 2024):
   - Paper: https://arxiv.org/abs/2410.22551
   - Method: Three-level resampling, class diversity loss, balanced sampling
   - Code: NOT YET AVAILABLE (paper just published)
   - Implementation: Will require custom development using Hugging Face Diffusers

2. **DermDiff** (March 2025):
   - Paper: https://arxiv.org/abs/2503.17536
   - Method: Skin tone detector + race-conditioned diffusion + multimodal text-image learning
   - Implementation: PyTorch with HuggingFace Diffusers + OpenAI APIs
   - Generated Dataset: 60k synthetic images (30k benign, 30k malignant)
   - Code: NOT EXPLICITLY AVAILABLE (check arXiv code links)

3. **From Majority to Minority** (June 2024):
   - Paper: https://arxiv.org/html/2406.18375
   - GitHub: https://github.com/janet-sw/skin-diff
   - Award: MICCAI ISIC Workshop 2024 Honorable Mention
   - Method: Stable Diffusion via Textual Inversion + LoRA
   - Implementation: AVAILABLE (Hugging Face Diffusers)
   - **RECOMMENDED FOR PHASE 2 IMPLEMENTATION**

**Next Steps for Phase 2**:
- Review janet-sw/skin-diff GitHub repository
- Implement tone-conditioned Stable Diffusion using HuggingFace Diffusers
- Train on HAM10000 + ISIC 2019 + Fitzpatrick17k (once acquired)
- Generate 60k synthetic images with target FST distribution: 25% FST V-VI
- Validate quality: FID <20, LPIPS <0.1, expert dermatologist review

**Documentation**: See `docs/synthetic_augmentation.md`

---

## Summary Statistics

| Dataset | Size | FST V-VI % | Access Status | Priority |
|---------|------|------------|---------------|----------|
| HAM10000 | 10,015 | <5% | ✅ Public | HIGH (baseline) |
| ISIC 2019 | 25,331 | <3% | ✅ Public | HIGH (baseline) |
| Fitzpatrick17k | 16,577 | ~8% | ⏳ Request | HIGH (FST labels) |
| DDI | 656 | 34% | ✅ Available | CRITICAL (diversity) |
| MIDAS | Variable | ~28% | ✅ Available | HIGH (multimodal) |
| SCIN | 10,000+ | ~33% | ✅ Public | HIGH (real-world) |
| Synthetic (Phase 2) | 60,000 | 25% target | ⏳ Pending | MEDIUM (augmentation) |

**Total Training Dataset Target (Phase 2)**: ~130,000 images with 25%+ FST V-VI representation

---

## Action Items

**IMMEDIATE (Week 1)**:
- [ ] Download HAM10000 from Kaggle (using Kaggle API)
- [ ] Download ISIC 2019 from challenge website
- [ ] Submit Fitzpatrick17k access request to Dr. Matthew Groh
- [ ] Download DDI from Stanford AIMI portal
- [ ] Download MIDAS from Stanford AIMI portal
- [ ] Clone SCIN from GitHub repository

**SHORT-TERM (Week 2-3)**:
- [ ] Verify all dataset file integrity (checksums, image loading)
- [ ] Parse and merge metadata CSVs
- [ ] Analyze FST distribution across datasets
- [ ] Implement FST annotation for ISIC 2019 (no native labels)
- [ ] Create stratified train/val/test splits (balanced FST)

**MEDIUM-TERM (Week 4-6)**:
- [ ] Implement FST annotation protocol using ITA + Monk Skin Tone
- [ ] Annotate ISIC 2019 images (automated ITA + manual validation)
- [ ] Prepare Phase 2 synthetic data generation pipeline

---

## Risk Assessment

**High Risk**:
- Fitzpatrick17k access delay (mitigation: use CSV URLs to download from original atlases if form delayed)

**Medium Risk**:
- ISIC 2019 download size (2.6GB+, may require bandwidth/storage management)
- FST annotation quality for datasets without native labels (mitigation: dual annotation + ITA validation)

**Low Risk**:
- Public datasets (HAM10000, ISIC, SCIN) - minimal access barriers
- Stanford AIMI datasets (DDI, MIDAS) - streamlined Research Use Agreement

---

## Contact Information

**Dataset Curators**:
- HAM10000: Peter Tschandl (Medical University of Vienna)
- Fitzpatrick17k: Matthew Groh (MIT Media Lab)
- DDI: Roxana Daneshjou (Stanford Dermatology)
- MIDAS: Leo McCoy, Bhavik Naik (Stanford AIMI)
- SCIN: Abhishek Jain (Google Health)

**Institutional Contact**:
- PI: Dr. Nabeel Alzahrani (CSUSB)
- Researcher: Jasmin Flores (CSUSB)

---

**Last Updated**: 2025-10-13
**Next Review**: 2025-10-14 (daily updates during acquisition phase)
**Maintained by**: the_didact (MENDICANT_BIAS framework)
