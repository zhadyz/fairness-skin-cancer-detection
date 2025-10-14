# FairSkin Diffusion Augmentation: Implementation Plan

## Executive Summary

FairSkin is a diffusion-based data augmentation technique that generates synthetic skin lesion images with balanced Fitzpatrick Skin Type (FST) representation. This document provides a comprehensive implementation plan including architecture details, training requirements, integration strategy, and quality validation protocols.

**Expected Impact**: +18-21% AUROC improvement for FST V-VI groups (Ju et al., 2024)

---

## 1. Architecture Overview

### 1.1 Base Model: Stable Diffusion v1.5

**Foundation**:
- Pre-trained Stable Diffusion v1.5 (CompVis/Stability AI)
- 860M parameters (U-Net: 860M, VAE: 83M, CLIP text encoder: 123M)
- Trained on LAION-5B (general domain images)

**Why Stable Diffusion**:
- State-of-the-art image generation quality (FID <20)
- Efficient inference (20-50 steps, 3-6s on RTX 3090)
- Extensive community support and tooling (Hugging Face Diffusers)
- Parameter-efficient fine-tuning via LoRA (Low-Rank Adaptation)

### 1.2 Fine-Tuning Strategy: LoRA + Textual Inversion

**LoRA (Low-Rank Adaptation)**:
- Freeze base Stable Diffusion weights (860M params)
- Train low-rank decomposition matrices: ΔW = BA (where B is rank×original_dim, A is original_dim×rank)
- Typical rank r=4-16 (reduces trainable params from 860M to ~3-10M)
- Only update cross-attention layers in U-Net (most semantically relevant)

**Textual Inversion**:
- Learn new token embeddings for medical concepts
- Example tokens: `<melanoma-FST-VI>`, `<nevus-FST-I>`, `<basal-cell-FST-IV>`
- Embedding dimension: 768 (CLIP text encoder dimension)
- Freezes all weights except new token embeddings (~500K params)

**Combined Approach** (from janet-sw/skin-diff):
1. Phase 1: Textual Inversion (find optimal token embeddings)
   - Train 1000-2000 steps (~2-4 hours on RTX 3090)
   - Validate: Can generate basic lesion concepts from text prompts
2. Phase 2: LoRA fine-tuning (adapt U-Net to medical domain)
   - Train 5000-10000 steps (~10-20 hours on RTX 3090)
   - Integrate new tokens learned in Phase 1
3. Phase 3: Joint refinement (optional)
   - Co-train both LoRA and token embeddings
   - 2000-5000 additional steps (~4-10 hours)

### 1.3 Conditioning Mechanism

**Multi-Level Conditioning**:
1. **Text Prompt** (primary):
   - Format: `"A dermoscopic image of {diagnosis} on Fitzpatrick skin type {FST}, high quality medical photograph"`
   - Example: `"A dermoscopic image of melanoma on Fitzpatrick skin type VI, high quality medical photograph"`

2. **Class Labels** (optional, via classifier-free guidance):
   - Diagnosis class (7 classes: MEL, NV, BCC, AK, BKL, DF, VASC)
   - FST class (6 classes: I-VI)
   - Encoded as additional conditioning vectors

3. **Image Conditioning** (for controlled generation):
   - Reference image from minority group (FST V-VI)
   - Extract CLIP embeddings, add to text embeddings
   - Enables style transfer: "Generate MEL with texture from this image"

**Three-Level Resampling** (Ju et al., 2024):
1. **Balanced Sampling**: Oversample minority FST classes during training
   - FST I-III: 40% of batches
   - FST IV: 20% of batches
   - FST V-VI: 40% of batches (vs <5% in original datasets)

2. **Class Diversity Loss**: Add auxiliary loss to encourage intra-class diversity
   - L_diversity = -log(1/N × Σ cosine_distance(embedding_i, embedding_j))
   - Penalizes mode collapse (all FST VI images looking identical)

3. **Imbalance-Aware Augmentation**: During classifier training, dynamically weight synthetic vs real
   - FST I-III: 20% synthetic, 80% real (abundant data)
   - FST V-VI: 80% synthetic, 20% real (scarce data)

---

## 2. Training Requirements

### 2.1 Dataset Specification

**Minimum Viable Dataset**:
- Size: 500-1000 dermatology images (per diagnosis class)
- FST distribution: Minimum 100 images per FST class (150-200 preferred)
- Quality: High-resolution dermoscopic images (512x512 minimum, 1024x1024 preferred)
- Annotations:
  - Diagnosis label (7 classes: MEL, NV, BCC, AK, BKL, DF, VASC)
  - FST label (I-VI, dual annotation preferred)
  - Optional: ITA (Individual Typology Angle) for validation

**Recommended Training Datasets**:
1. **Fitzpatrick17k**: 16,577 images, ~8% FST V-VI
   - Use ALL images for textual inversion (broad concept learning)
   - Use FST V-VI subset for LoRA fine-tuning (focus on minority groups)

2. **DDI (Stanford)**: 656 images, 34% FST V-VI (gold standard quality)
   - Primary dataset for LoRA fine-tuning
   - Clinician-annotated, biopsy-confirmed

3. **HAM10000**: 10,015 images, <5% FST V-VI (high quality, tone-imbalanced)
   - Use only for textual inversion (learn lesion morphology)
   - Do NOT use for LoRA (would bias toward light tones)

**Data Preprocessing**:
- Resize: 512x512 (Stable Diffusion v1.5 native resolution)
- Normalization: [0, 1] range (diffusion models trained on normalized images)
- Augmentation: ONLY for real data (horizontal flip, rotation, color jitter)
  - Do NOT augment during diffusion training (hurts generation quality)
- Hair removal: Apply automated hair removal (DullRazor algorithm)
  - Prevents model from learning "hair = dark skin" spurious correlation

### 2.2 GPU Requirements

**Hardware Specifications**:
- Minimum: 1x RTX 3090 (24GB VRAM)
- Recommended: 2x RTX 4090 (48GB total VRAM)
- Optimal: 4x A100 (160GB total VRAM, reduced training time by 4x)

**VRAM Breakdown** (512x512 resolution, batch size 4):
- Model weights: 3.4GB (Stable Diffusion v1.5)
- LoRA adapters: 0.8GB (rank 16, all attention layers)
- Optimizer state (AdamW): 4.2GB (2x model params for momentum + variance)
- Activations (forward pass): 2.1GB per image × 4 = 8.4GB
- Gradient checkpointing: Reduces to 4.2GB (-50% VRAM, +20% time)
- **Total: 16.8GB** (fits on RTX 3090 with gradient checkpointing)

**Training Time Estimates** (RTX 3090, 1000 training images):
- Textual Inversion: 2000 steps × 1.2s/step = 40 minutes
- LoRA Training: 10000 steps × 2.8s/step = 7.8 hours
- Joint Refinement: 3000 steps × 3.1s/step = 2.6 hours
- **Total: ~12-14 hours** (single GPU, sequential training)

**Multi-GPU Scaling**:
- 2 GPUs: 6-7 hours (1.9x speedup, communication overhead)
- 4 GPUs: 3-4 hours (3.2x speedup)
- 8 GPUs: 2-3 hours (4.8x speedup, diminishing returns)

### 2.3 Hyperparameters (Optimized from Literature)

**Textual Inversion**:
```python
learning_rate = 5e-4  # Higher than LoRA (only ~500K params)
num_steps = 2000
batch_size = 4
gradient_accumulation = 2  # Effective batch size: 8
lr_scheduler = "constant_with_warmup"
warmup_steps = 100
optimizer = "AdamW"
weight_decay = 0.01
max_grad_norm = 1.0
```

**LoRA Training**:
```python
learning_rate = 1e-4  # Lower than textual inversion (more params)
num_steps = 10000
batch_size = 4
gradient_accumulation = 2
lora_rank = 16  # Balance: 8 (underfits), 32 (overfits)
lora_alpha = 32  # Scaling factor (typically 2×rank)
lora_dropout = 0.1  # Regularization
target_modules = ["to_q", "to_k", "to_v", "to_out"]  # Cross-attention
lr_scheduler = "cosine_with_restarts"
num_cycles = 3  # Escape local minima
optimizer = "AdamW"
weight_decay = 0.01
max_grad_norm = 1.0
```

**Diffusion-Specific**:
```python
noise_scheduler = "DDPM"  # Denoising Diffusion Probabilistic Model
num_train_timesteps = 1000  # Stable Diffusion default
beta_schedule = "scaled_linear"  # Noise schedule
prediction_type = "epsilon"  # Predict noise (vs velocity or x0)
snr_gamma = 5.0  # Signal-to-noise ratio weighting (improves quality)
```

**Class Diversity Loss** (FairSkin-specific):
```python
lambda_diversity = 0.1  # Weight for diversity loss
diversity_metric = "cosine_distance"  # vs L2 distance
sample_pool_size = 16  # Number of embeddings to compare
```

---

## 3. Inference Pipeline

### 3.1 Generation Protocol

**Single Image Generation**:
```python
from diffusers import StableDiffusionPipeline
import torch

# Load fine-tuned model
model_id = "CompVis/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Half precision (2x faster, -50% VRAM)
).to("cuda")

# Load LoRA weights
pipe.unet.load_attn_procs("path/to/lora_weights")

# Load custom tokens
pipe.tokenizer.add_tokens(["<melanoma-FST-VI>", "<nevus-FST-I>", ...])
pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
pipe.text_encoder.load_state_dict(torch.load("path/to/token_embeddings.pt"))

# Generate image
prompt = "A dermoscopic image of melanoma on Fitzpatrick skin type VI, high quality"
negative_prompt = "blurry, low quality, text, watermark, duplicated"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,  # Trade-off: 20 (fast, lower quality), 100 (slow, best)
    guidance_scale=7.5,  # Classifier-free guidance (higher = more prompt adherence)
    height=512,
    width=512,
).images[0]
```

**Batch Generation** (for 60k synthetic dataset):
```python
# Configuration
target_images = 60000
diagnoses = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]
fst_classes = ["I", "II", "III", "IV", "V", "VI"]

# Balanced sampling: Oversample FST V-VI
fst_distribution = {
    "I": 0.10, "II": 0.10, "III": 0.15,  # 35% light tones
    "IV": 0.15,                           # 15% intermediate
    "V": 0.25, "VI": 0.25                 # 50% dark tones (vs <5% in original)
}

# Generate with quality filtering
high_quality_images = []
for diagnosis in diagnoses:
    for fst in fst_classes:
        num_images = int(target_images * fst_distribution[fst] / len(diagnoses))

        for i in range(num_images * 1.5):  # Generate 50% extra for filtering
            image = generate_image(diagnosis, fst)

            # Quality checks (see section 3.2)
            if passes_quality_checks(image):
                high_quality_images.append((image, diagnosis, fst))
                if len([x for x in high_quality_images if x[2] == fst]) >= num_images:
                    break

# Result: 60,000 high-quality synthetic images with balanced FST distribution
```

### 3.2 Quality Validation

**Automatic Quality Metrics**:

1. **FID (Frechet Inception Distance)**: <20 (threshold from literature)
   - Measures distribution similarity: synthetic vs real images
   - Calculate per FST class: FID_FST-VI should be <25 (slightly higher acceptable for rare groups)
   - **Implementation**: Use `pytorch-fid` library, 2048-dim Inception-v3 features

2. **LPIPS (Learned Perceptual Image Patch Similarity)**: <0.15 (vs real images)
   - Measures perceptual similarity using deep features
   - Lower = more realistic (but not identical, which would indicate memorization)
   - **Implementation**: Use `lpips` library, AlexNet or VGG backbone

3. **Classifier Confidence** (diagnosis prediction):
   - Train ResNet50 on real data, evaluate on synthetic
   - Synthetic images should yield confident predictions (softmax >0.7)
   - Low confidence = unrealistic lesion morphology

4. **Diversity Score** (intra-class):
   - CLIP embeddings for all synthetic images of same (diagnosis, FST)
   - Average pairwise cosine distance: >0.3 (avoid mode collapse)
   - Low diversity = model generating identical images

**Expert Dermatologist Review** (Quality Assurance):
- Sample 500 synthetic images (stratified by diagnosis × FST)
- 3 dermatologists rate each image (blinded, mixed with real images)
- Rating scale: 1-7 (1=clearly fake, 4=uncertain, 7=clinically realistic)
- **Acceptance criteria**: Mean score >5.0, <10% images rated <3

**Quality Filtering Pipeline**:
```python
def passes_quality_checks(image, diagnosis, fst):
    # 1. Resolution check
    if image.size != (512, 512):
        return False

    # 2. Brightness check (avoid pure black/white)
    mean_brightness = image.mean()
    if mean_brightness < 0.1 or mean_brightness > 0.9:
        return False

    # 3. FID check (per-image approximation using nearest neighbor)
    fid_score = compute_single_image_fid(image, real_dataset_fst=fst)
    if fid_score > 30:
        return False

    # 4. LPIPS check (vs 10 random real images of same FST)
    lpips_scores = [compute_lpips(image, real_img) for real_img in sample_real_images(fst, n=10)]
    if np.mean(lpips_scores) > 0.2:
        return False

    # 5. Classifier confidence check
    prediction = classifier.predict(image)
    if prediction["confidence"] < 0.6:
        return False

    # 6. Diversity check (vs previous synthetic images)
    if is_too_similar_to_existing(image, existing_synthetic_images):
        return False

    return True
```

---

## 4. Integration with Training Loop

### 4.1 Synthetic + Real Data Mixing

**Three Strategies** (from literature):

**Strategy 1: Pre-Generate Static Dataset** (Recommended for Phase 2)
- Generate 60,000 synthetic images BEFORE classifier training
- Store in disk (lossless PNG format)
- Mix with real data during DataLoader sampling

**Advantages**:
- Fast training (no generation overhead during epochs)
- Reproducible experiments (same synthetic dataset)
- Easy quality control (filter before training)

**Disadvantages**:
- Large storage (60k × 512×512×3 × 8bit = ~47GB)
- No dynamic adaptation (fixed synthetic dataset)

**Implementation**:
```python
# Pre-generation script (run once)
python scripts/generate_fairskin_dataset.py \
    --num_images 60000 \
    --output_dir data/synthetic/fairskin \
    --fst_distribution balanced \
    --quality_threshold high

# Training script (use mixed dataset)
from torch.utils.data import ConcatDataset

real_dataset = FitzpatrickDataset(root="data/real")
synthetic_dataset = FitzpatrickDataset(root="data/synthetic/fairskin")

# Weighted sampling: FST-dependent synthetic ratio
train_dataset = WeightedMixedDataset(
    real=real_dataset,
    synthetic=synthetic_dataset,
    synthetic_ratio_by_fst={
        "I": 0.2, "II": 0.2, "III": 0.3,  # 20-30% synthetic for light tones
        "IV": 0.5,                         # 50% for intermediate
        "V": 0.7, "VI": 0.8                # 70-80% for dark tones (scarce real data)
    }
)
```

**Strategy 2: On-the-Fly Generation** (Advanced, Phase 3+)
- Generate synthetic images during training (as augmentation)
- Cache last N generated images to avoid redundant generation

**Advantages**:
- No storage overhead
- Dynamic adaptation (generate images model struggles with)
- Infinite dataset size (never see same synthetic image twice)

**Disadvantages**:
- Slow training (3-6s generation time per image)
- Requires powerful GPU for simultaneous generation + training
- Harder to debug (non-reproducible)

**Strategy 3: Hybrid** (Best of Both)
- Pre-generate 30k synthetic images (core dataset)
- Generate additional 5-10% on-the-fly (for hard examples)
- Use classifier loss to guide generation: "Generate more FST VI melanoma images, current model struggles"

### 4.2 Training Protocol

**Phase 1: Pre-train on Synthetic (Optional)**
- Train ResNet50 on 60k synthetic images ONLY
- Goal: Learn FST-invariant representations
- 50 epochs, standard hyperparameters
- **Expected**: Lower accuracy (85-88%) but better fairness (AUROC gap <6%)

**Phase 2: Fine-tune on Real**
- Initialize from Phase 1 checkpoint
- Train on mixed dataset (real + synthetic)
- 100 epochs, lower learning rate (1e-4 → 1e-5 after 50 epochs)
- **Expected**: High accuracy (91-93%) + maintained fairness

**Phase 3: Domain Adaptation** (if synthetic artifacts detected)
- Use domain adversarial training (DANN)
- Auxiliary discriminator: Predict real vs synthetic
- Maximize classification accuracy, minimize domain predictability
- **Expected**: Further reduce AUROC gap (-1-2% improvement)

---

## 5. Implementation Timeline

### Week 1: Setup & Dataset Preparation
- Day 1-2: Install Hugging Face Diffusers, PyTorch, dependencies
- Day 3-4: Download Fitzpatrick17k, DDI, HAM10000
- Day 5: Preprocess datasets (resize, normalize, hair removal)
- Day 6-7: Create training splits, verify FST distributions

**Deliverables**: `data/processed/fitzpatrick17k/`, `data/processed/ddi/`, preprocessing scripts

### Week 2: Textual Inversion
- Day 1-2: Implement textual inversion training script
- Day 3: Train token embeddings (2000 steps, ~4 hours)
- Day 4: Validate: Generate images from text prompts, qualitative review
- Day 5-7: Iterate: Adjust learning rate, add more tokens if needed

**Deliverables**: `checkpoints/textual_inversion/token_embeddings.pt`, validation images

### Week 3-4: LoRA Training
- Day 1-2: Implement LoRA training script (integrate with textual inversion)
- Day 3-5: Train LoRA adapters (10k steps, ~20 hours)
- Day 6-7: Validate: Generate 100 images per (diagnosis × FST), compute FID/LPIPS
- Day 8-10: Iterate: Adjust rank, alpha, learning rate if quality insufficient

**Deliverables**: `checkpoints/lora/lora_weights.pt`, quality metrics report

### Week 5: Batch Generation
- Day 1-2: Implement batch generation script with quality filtering
- Day 3-5: Generate 60k synthetic images (~120 hours GPU time, run overnight/weekend)
- Day 6-7: Expert review: Sample 500 images, dermatologist rating

**Deliverables**: `data/synthetic/fairskin/` (60k images), quality report

### Week 6: Integration & Training
- Day 1-2: Implement mixed dataset loader (real + synthetic)
- Day 3-7: Train ResNet50 classifier with mixed data (100 epochs, ~48 hours)
- Evaluate: AUROC per FST, compare to baseline

**Deliverables**: `models/fairskin_resnet50.pth`, fairness metrics report

**Total Time: 6 weeks (42 days)**
- GPU-intensive tasks: ~170 hours (7 days continuous GPU usage)
- Human time: ~40 hours (1 week full-time equivalent)

---

## 6. Pre-Trained Models & Open-Source Resources

### 6.1 Available Checkpoints

**Option 1: janet-sw/skin-diff** (GitHub)
- Repository: https://github.com/janet-sw/skin-diff
- Paper: "From Majority to Minority" (MICCAI ISIC Workshop 2024, Honorable Mention)
- Base model: Stable Diffusion v1.5
- Pre-trained: Textual Inversion + LoRA on HAM10000 + ISIC 2019
- **Pros**: Ready to use, validated in peer-reviewed work
- **Cons**: Trained on tone-imbalanced datasets (may need re-training)
- **License**: Not specified in repo (contact authors)

**Option 2: Train from Scratch** (Recommended)
- Use Stable Diffusion v1.5 as base (open license: CreativeML Open RAIL-M)
- Train on Fitzpatrick17k + DDI (FST-diverse datasets)
- Full control over hyperparameters, data distribution
- **Estimated time**: 6 weeks (see section 5)

**Option 3: Pre-trained Dermatology Models** (Hugging Face Hub)
- Search query: "dermatology diffusion" OR "skin lesion generation"
- As of 2025-01, no dedicated dermatology diffusion models on Hub
- General Stable Diffusion v1.5/v2.1 models available
- **Recommendation**: Start with SD v1.5, fine-tune for dermatology

### 6.2 Alternative Architectures (Future Work)

**DermDiff** (Hypothetical, if released):
- Specialized dermatology diffusion model
- Pre-trained on 100k+ dermoscopic images
- FST-aware conditioning built-in
- **If released**: Use as base instead of SD v1.5 (faster training, better quality)

**Latent Diffusion with MedCLIP**:
- Replace CLIP text encoder with MedCLIP (medical domain-specific)
- Better understanding of clinical terminology
- **Implementation**: Swap `text_encoder` in Diffusers pipeline
- **Expected**: +2-3% generation quality (FID improvement)

---

## 7. Key Questions & Answers

### Q1: Can we use pre-trained dermatology diffusion models?
**Answer**: As of 2025-01, no pre-trained dermatology-specific diffusion models are publicly available on Hugging Face Hub. The janet-sw/skin-diff repository provides LoRA weights, but these were trained on tone-imbalanced datasets (HAM10000, ISIC 2019). **Recommendation**: Start with Stable Diffusion v1.5, fine-tune on Fitzpatrick17k + DDI for FST diversity.

### Q2: What's the minimum viable dataset for LoRA training?
**Answer**: 500-1000 images per diagnosis class, with minimum 100 images per FST class. DDI (656 images, 34% FST V-VI) is sufficient for initial experiments. For production, combine Fitzpatrick17k (16,577 images) + DDI (656 images) = **17,233 images total**, which enables robust training.

### Q3: How to ensure synthetic quality (FID <20, LPIPS <0.1)?
**Answer**: Multi-pronged approach:
1. **Data quality**: Use high-resolution (512x512+), clinician-annotated training data
2. **Hyperparameter tuning**: Rank 16, alpha 32, learning rate 1e-4 (see section 2.3)
3. **Class diversity loss**: Lambda 0.1 (prevents mode collapse)
4. **Quality filtering**: Generate 1.5x target images, keep only high-quality (see section 3.2)
5. **Expert review**: Dermatologist rating >5/7 on 500-image sample

**Empirical benchmarks**:
- janet-sw/skin-diff: FID 18.3 (HAM10000 test set)
- FairSkin paper: FID 16.7 (Fitzpatrick17k)
- **Target**: FID <20 per FST class (FST VI may reach ~22-25, acceptable)

### Q4: Pre-generate 60k images or on-the-fly during training?
**Answer**: **Pre-generate for Phase 2** (production hardening in Phase 4 can explore on-the-fly).

**Rationale**:
- Phase 2 focus: Validate fairness improvement (need reproducible experiments)
- Pre-generation: Fixed dataset enables direct comparison across runs
- Storage: 47GB (negligible on modern systems)
- Training speed: No generation overhead (faster epoch time)

**On-the-fly for Phase 3+** (if benefits justify complexity):
- Dynamic adaptation: Generate images model struggles with
- Infinite diversity: Never repeat synthetic image
- **Implementation complexity**: Requires multi-GPU setup (1 for generation, 1+ for training)

---

## 8. Risk Mitigation

### Risk 1: Low Synthetic Quality (FID >30)
**Mitigation**:
- Use higher-quality training data (DDI vs HAM10000)
- Increase LoRA rank (16 → 32) and training steps (10k → 20k)
- Add classifier-guided generation (use pre-trained ResNet to filter bad images)

### Risk 2: Mode Collapse (All FST VI Images Look Identical)
**Mitigation**:
- Class diversity loss (lambda 0.1-0.2)
- Increase diffusion steps during inference (50 → 100)
- Use temperature scaling in sampling (temperature 0.7-0.9)

### Risk 3: Memorization (Synthetic Images Identical to Training Data)
**Mitigation**:
- Check LPIPS <0.1 vs training set (high similarity = memorization)
- Use LoRA dropout 0.1 (regularization)
- Limit training steps if overfitting detected (reduce 10k → 5k)

### Risk 4: Domain Shift (Classifier Fails on Synthetic Images)
**Mitigation**:
- Domain adversarial training (DANN): Make classifier agnostic to real vs synthetic
- Gradually increase synthetic ratio (start 30%, increase to 80% over epochs)
- Hybrid strategy: Always keep 20% real data in batches

### Risk 5: Ethical Concerns (Synthetic Medical Data)
**Mitigation**:
- Expert validation: Dermatologist review mandatory (500+ images)
- Model card transparency: Disclose synthetic data usage, proportions
- Clinical trial: Validate on real-world prospective data (Phase 5)
- Regulatory guidance: Consult FDA/EMA on synthetic data acceptability

---

## 9. Success Criteria

### Technical Metrics
- FID <20 per FST class (FST VI: <25 acceptable)
- LPIPS <0.15 (vs real images of same FST)
- Classifier confidence >0.7 on synthetic images
- Intra-class diversity >0.3 (CLIP embedding distance)

### Fairness Impact
- AUROC gap reduction: 15-20% (baseline) → 8-12% (Phase 2) = **40-60% improvement**
- Expected AUROC gain for FST V-VI: +18-21% (literature benchmark)
- EOD reduction: >30% (from FairSkin data augmentation alone)

### Qualitative Validation
- Dermatologist rating: Mean >5.0/7.0 (500-image sample)
- Acceptance rate: >90% images rated ≥4/7
- Rejection rate: <10% images rated <3/7 (clearly synthetic)

### Operational
- Generation time: <6s per image (RTX 3090, 50 steps)
- Storage: <50GB (60k images, compressed PNG)
- Integration time: <1 week (implement mixed dataset loader)

---

## 10. References

**Primary Paper**:
- Ju, L., et al. (2024). "FairSkin: Fair Diffusion for Skin Disease Image Generation." arXiv:2410.22551

**Implementation Reference**:
- janet-sw. (2024). "skin-diff: From Majority to Minority." GitHub. https://github.com/janet-sw/skin-diff

**Diffusion Frameworks**:
- Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR.
- Hugging Face Diffusers: https://huggingface.co/docs/diffusers

**LoRA & Fine-Tuning**:
- Hu, E.J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
- Gal, R., et al. (2022). "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." arXiv:2208.01618

**Quality Metrics**:
- Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS. (FID metric)
- Zhang, R., et al. (2018). "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric." CVPR. (LPIPS metric)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-13
**Author**: THE DIDACT (Strategic Research Agent)
**Status**: IMPLEMENTATION-READY
**Next Review**: Post-Phase 2 (Week 10)
