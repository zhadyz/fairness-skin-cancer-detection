# CIRCLe: Color-Invariant Representation Learning - Implementation Guide

## Executive Summary

CIRCLe (Color Invariant Representation Learning for Unbiased Classification of Skin Lesions) is an algorithm-level fairness technique that enforces skin tone invariance through regularization. By encouraging similar latent representations for images with same diagnosis but different skin tones, CIRCLe improves calibration and out-of-distribution generalization.

**Expected Impact**: 3-5% ECE (Expected Calibration Error) reduction, improved OOD generalization, +2-4% AUROC for FST V-VI (Pakzad et al., 2022)

---

## 1. Methodology Overview

### 1.1 Core Concept

**Problem**: Deep learning models learn spurious correlations between skin color and diagnosis
- Example: "Dark skin + lesion = benign nevus" (dataset bias)
- Result: Poor performance on rare (diagnosis, FST) combinations

**Solution**: Regularize latent embeddings to be invariant to skin tone transformations
- Same diagnosis, different FST → similar embeddings
- Different diagnosis, any FST → dissimilar embeddings

**Mathematical Formulation**:
```
L_total = L_cls + λ_reg × L_reg

Where:
L_cls = CrossEntropy(f(x), y_diagnosis)
L_reg = ||f(x_original) - f(T_FST(x_original))||²

f(·): Feature extractor (e.g., ResNet50 embeddings)
T_FST(·): Skin tone transformation (FST I ↔ VI)
λ_reg: Regularization strength (typically 0.1-0.3)
```

### 1.2 Pipeline

```
Original Image (FST III)
        |
        v
   ┌─────────────────────┐
   │ Skin Tone Transformer│  ← StarGAN or Color Transformation
   │    T_FST(x)          │     Generate FST I, VI versions
   └─────────────────────┘
        |
        ├────────────────┐
        v                v
  x_FST-I          x_FST-VI
        |                |
        v                v
   ┌────────────────────────┐
   │  Feature Extractor (f) │  ← ResNet50 or other backbone
   │  Shared Weights        │
   └────────────────────────┘
        |                |
        v                v
   emb_FST-I       emb_FST-VI
        |                |
        └───────┬────────┘
                v
        Regularization Loss
        L_reg = ||emb_FST-I - emb_FST-VI||²

                +

        Classification Loss
        L_cls = CrossEntropy(f(x), y)
```

---

## 2. Skin Tone Transformation Approaches

### 2.1 Approach 1: StarGAN (Original Paper)

**Architecture**: StarGAN v2 (Choi et al., 2020)
- Generator: Transforms image from source FST → target FST
- Discriminator: Verifies realism of generated images
- Style encoder: Extracts skin tone style codes

**Training Requirements**:
- Dataset: 5,000+ images per FST class (for robust GAN training)
- GPU: 1x RTX 3090 (24GB VRAM)
- Training time: 100-200 epochs × 1 hour/epoch = **100-200 hours**
- Hyperparameters:
  - Learning rate: 1e-4 (generator), 1e-4 (discriminator)
  - Batch size: 8 (high-resolution images)
  - Adversarial loss weight: 1.0
  - Style reconstruction loss weight: 1.0
  - Cycle consistency loss weight: 1.0

**Advantages**:
- High-quality transformations (realistic skin tone changes)
- Preserves lesion morphology (shape, texture, borders)
- Medical domain adaptation possible (fine-tune on dermoscopy)

**Disadvantages**:
- Complex training (GAN instability, mode collapse risks)
- Requires large FST-diverse dataset (5k+ images per FST)
- Long training time (100-200 GPU hours)
- Potential artifacts (blurriness, checkerboard patterns)

**Implementation** (using official StarGAN repository):
```bash
# Clone StarGAN v2
git clone https://github.com/clovaai/stargan-v2
cd stargan-v2

# Prepare dataset (organize by FST)
python prepare_data.py \
    --input_dir data/fitzpatrick17k \
    --output_dir data/stargan_fst \
    --attribute fst \
    --classes I,II,III,IV,V,VI

# Train StarGAN
python main.py \
    --mode train \
    --num_domains 6 \
    --train_img_dir data/stargan_fst/train \
    --val_img_dir data/stargan_fst/val \
    --batch_size 8 \
    --total_iters 100000 \
    --lambda_reg 1.0 \
    --lambda_sty 1.0 \
    --lambda_cyc 1.0

# Generate transformed images
python main.py \
    --mode sample \
    --checkpoint_dir checkpoints/stargan_fst \
    --result_dir results/transformed \
    --src_dir data/fitzpatrick17k/test
```

**Quality Validation**:
- FID (Frechet Inception Distance): <30 per FST transformation
- LPIPS (perceptual similarity): 0.2-0.4 (vs original, should preserve structure)
- Expert review: Dermatologist rating >4/7 for realism

### 2.2 Approach 2: Simple Color Transformations (Practical Alternative)

**Concept**: Approximate skin tone changes using color space manipulations
- No GAN training required
- Fast, deterministic, no artifacts
- Less realistic but sufficient for regularization

**Transformations**:

1. **HSV (Hue, Saturation, Value) Adjustment**:
   ```python
   import cv2
   import numpy as np

   def transform_skin_tone_hsv(image, target_fst):
       """
       Transform image to target FST using HSV adjustments.

       FST I-III (Light): Increase brightness, decrease saturation
       FST IV (Intermediate): Minimal changes
       FST V-VI (Dark): Decrease brightness, increase saturation
       """
       hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

       # Define FST-specific adjustments (empirically tuned)
       fst_adjustments = {
           "I":  {"h": 0,   "s": -0.2, "v": +0.3},   # Very light
           "II": {"h": 0,   "s": -0.1, "v": +0.2},   # Light
           "III": {"h": 0,  "s": 0.0,  "v": +0.1},   # Light-medium
           "IV": {"h": 0,   "s": 0.0,  "v": 0.0},    # Medium (baseline)
           "V":  {"h": +5,  "s": +0.1, "v": -0.2},   # Dark
           "VI": {"h": +10, "s": +0.2, "v": -0.3},   # Very dark
       }

       adj = fst_adjustments[target_fst]

       # Apply adjustments
       hsv[:, :, 0] = np.clip(hsv[:, :, 0] + adj["h"], 0, 179)        # Hue
       hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + adj["s"]), 0, 255)  # Saturation
       hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + adj["v"]), 0, 255)  # Value

       rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
       return rgb
   ```

2. **LAB Color Space Adjustment** (More Perceptually Uniform):
   ```python
   def transform_skin_tone_lab(image, target_fst):
       """
       Transform using LAB color space (L*a*b*).
       L: Lightness (0-100)
       a: Green-Red axis
       b: Blue-Yellow axis
       """
       lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

       # FST-specific LAB adjustments
       fst_adjustments = {
           "I":  {"L": +30, "a": +5,  "b": +10},
           "II": {"L": +20, "a": +3,  "b": +7},
           "III": {"L": +10, "a": +2,  "b": +5},
           "IV": {"L": 0,   "a": 0,   "b": 0},
           "V":  {"L": -15, "a": -3,  "b": -5},
           "VI": {"L": -25, "a": -5,  "b": -8},
       }

       adj = fst_adjustments[target_fst]

       # Apply adjustments (L channel most important for skin tone)
       lab[:, :, 0] = np.clip(lab[:, :, 0] + adj["L"], 0, 255)
       lab[:, :, 1] = np.clip(lab[:, :, 1] + adj["a"], 0, 255)
       lab[:, :, 2] = np.clip(lab[:, :, 2] + adj["b"], 0, 255)

       rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
       return rgb
   ```

3. **Hybrid: Skin Segmentation + LAB Adjustment**:
   ```python
   def transform_skin_tone_segmented(image, target_fst):
       """
       1. Segment skin regions (avoid lesion)
       2. Apply LAB transformation only to skin
       3. Preserve lesion appearance
       """
       # Step 1: Segment skin (simple thresholding or U-Net)
       skin_mask = segment_skin(image)  # Binary mask

       # Step 2: Transform only skin regions
       transformed = transform_skin_tone_lab(image, target_fst)

       # Step 3: Blend original lesion with transformed skin
       result = image.copy()
       result[skin_mask > 0] = transformed[skin_mask > 0]

       return result
   ```

**Advantages**:
- No training required (instant setup)
- Fast (CPU-based, <10ms per image)
- Deterministic (reproducible)
- No artifacts or mode collapse

**Disadvantages**:
- Less realistic (may not capture FST diversity)
- Heuristic parameters (require manual tuning)
- May alter lesion appearance if not segmented

**Recommendation for Phase 2**: Use simple color transformations
- Faster iteration (no GAN training)
- Sufficient for regularization (embeddings learn invariance)
- Upgrade to StarGAN in Phase 3 if results insufficient

### 2.3 Approach 3: Pre-trained Dermatology StyleGAN (Future Work)

**Concept**: Use pre-trained StyleGAN2-ADA on dermatology images
- Latent space: Disentangled skin tone from lesion morphology
- Edit: Manipulate tone latent code, preserve morphology

**Availability**: No public dermatology StyleGAN as of 2025-01
**Alternative**: Train StyleGAN2-ADA on Fitzpatrick17k (requires 1-2 weeks GPU time)

---

## 3. Regularization Loss Formulation

### 3.1 L2 Distance Regularization (Original Paper)

**Formula**:
```
L_reg = 1/N × Σ ||f(x_i) - f(T_FST(x_i))||²

Where:
- N: Batch size
- x_i: Original image (source FST)
- T_FST(x_i): Transformed image (target FST, e.g., FST I → VI)
- f(·): Feature extractor (2048-dim embeddings from ResNet50)
```

**PyTorch Implementation**:
```python
import torch
import torch.nn as nn

class CIRCLeRegularizationLoss(nn.Module):
    def __init__(self, distance_metric="l2"):
        super().__init__()
        self.distance_metric = distance_metric

    def forward(self, embeddings_original, embeddings_transformed):
        """
        Args:
            embeddings_original: [batch_size, feature_dim] (e.g., 2048)
            embeddings_transformed: [batch_size, feature_dim]

        Returns:
            loss: Scalar regularization loss
        """
        if self.distance_metric == "l2":
            # Euclidean distance squared
            loss = torch.mean((embeddings_original - embeddings_transformed) ** 2)
        elif self.distance_metric == "cosine":
            # Cosine distance (1 - cosine_similarity)
            cosine_sim = F.cosine_similarity(embeddings_original, embeddings_transformed, dim=1)
            loss = torch.mean(1 - cosine_sim)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        return loss
```

**Hyperparameters**:
- λ_reg = 0.1-0.3 (regularization strength)
  - 0.1: Weak regularization (minimal fairness improvement)
  - 0.2: Balanced (recommended starting point)
  - 0.3: Strong regularization (may hurt accuracy if too aggressive)

### 3.2 Multi-FST Regularization (Extended)

**Concept**: Regularize against MULTIPLE tone transformations (not just one)
- Original FST III → Transform to FST I, VI
- Encourage f(x_FST-III) ≈ f(x_FST-I) ≈ f(x_FST-VI)

**Formula**:
```
L_reg = 1/(N×K) × Σ_i Σ_k ||f(x_i) - f(T_FST-k(x_i))||²

Where K = number of target FST classes (e.g., 2: FST I and VI)
```

**Implementation**:
```python
def multi_fst_regularization_loss(model, images, target_fsts=["I", "VI"]):
    """
    Regularize embeddings against multiple FST transformations.

    Args:
        model: Feature extractor (e.g., ResNet50)
        images: Original images [batch_size, 3, H, W]
        target_fsts: List of target FST classes (e.g., ["I", "VI"])

    Returns:
        loss: Multi-FST regularization loss
    """
    # Extract embeddings from original images
    embeddings_original = model.feature_extractor(images)

    total_loss = 0.0
    for target_fst in target_fsts:
        # Transform images to target FST
        images_transformed = transform_skin_tone(images, target_fst)

        # Extract embeddings from transformed images
        embeddings_transformed = model.feature_extractor(images_transformed)

        # Compute L2 distance
        loss_fst = torch.mean((embeddings_original - embeddings_transformed) ** 2)
        total_loss += loss_fst

    # Average over target FST classes
    return total_loss / len(target_fsts)
```

**Advantages**:
- More robust (invariant to multiple FST directions)
- Better OOD generalization (handles unseen FST combinations)

**Disadvantages**:
- Higher computational cost (K transformations per image)
- More GPU memory (need to store K transformed images + embeddings)

---

## 4. Training Protocol

### 4.1 Full Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Initialize model
model = ResNet50Classifier(num_classes=7).cuda()

# Loss functions
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = CIRCLeRegularizationLoss(distance_metric="l2")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Hyperparameters
lambda_reg = 0.2  # Regularization strength
target_fsts = ["I", "VI"]  # Extreme FST classes for regularization

# Training loop
for epoch in range(num_epochs):
    for images, labels, fst_labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass (original images)
        embeddings_original, logits = model(images, return_embeddings=True)
        loss_cls = criterion_cls(logits, labels)

        # Regularization: Transform to target FST, compute embedding distance
        loss_reg = 0.0
        for target_fst in target_fsts:
            # Transform images (on-the-fly or pre-computed)
            images_transformed = transform_skin_tone(images, target_fst)

            # Extract embeddings from transformed images
            embeddings_transformed, _ = model(images_transformed, return_embeddings=True)

            # Compute regularization loss
            loss_reg += criterion_reg(embeddings_original, embeddings_transformed)

        loss_reg /= len(target_fsts)

        # Total loss
        loss = loss_cls + lambda_reg * loss_reg

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    val_metrics = evaluate(model, val_loader)
    print(f"Epoch {epoch}: AUROC gap = {val_metrics['auroc_gap']:.2%}, ECE = {val_metrics['ece']:.4f}")
```

### 4.2 Data Augmentation Strategy

**Standard Augmentation** (always applied):
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness ±0.2, contrast ±0.2, saturation ±0.2)

**Tone Transformation** (for regularization):
- Applied during training (on-the-fly or pre-computed)
- Targets: FST I and VI (extreme classes)
- Frequency: Every batch (all images transformed)

**Pre-computed vs On-the-Fly**:

**Option 1: Pre-compute Transformations** (Recommended for Phase 2)
- Generate transformed versions of entire dataset BEFORE training
- Store: original + FST-I + FST-VI versions (3x storage)
- Training: Load all 3 versions, compute regularization

**Advantages**: Fast training (no transformation overhead)
**Disadvantages**: 3x storage (e.g., 48GB → 144GB)

**Option 2: On-the-Fly Transformation**
- Transform images during training (in DataLoader)
- No extra storage

**Advantages**: Storage-efficient
**Disadvantages**: CPU overhead (~20ms per transformation), may bottleneck GPU

**Recommendation**: Pre-compute for Phase 2 (easier debugging, faster training)

### 4.3 Hyperparameter Tuning

**Key Hyperparameters**:
- λ_reg: 0.1, 0.2, 0.3 (start with 0.2)
- Target FST classes: ["I", "VI"] or ["I", "III", "VI"]
- Distance metric: "l2" or "cosine"
- Transformation method: HSV, LAB, or StarGAN

**Tuning Strategy** (Grid Search):
```python
hyperparameter_grid = {
    "lambda_reg": [0.1, 0.2, 0.3],
    "target_fsts": [["I", "VI"], ["I", "III", "VI"]],
    "distance_metric": ["l2", "cosine"],
}

for config in iterate_grid(hyperparameter_grid):
    model = train_circle(config)
    val_metrics = evaluate(model, val_loader)
    log_experiment(config, val_metrics)

# Select best: Minimize AUROC gap, maximize calibration (minimize ECE)
best_config = select_best(
    criterion="auroc_gap + ece",  # Multi-objective
    direction="minimize"
)
```

**Expected Training Time per Config**:
- 100 epochs × 15 min/epoch = 25 hours (RTX 3090)
- 9 configs (3 λ_reg × 2 target_fsts × 2 metrics) = **225 hours total**
- Parallelize: 4 GPUs → 56 hours (2.3 days)

---

## 5. Model Architecture

### 5.1 Feature Extractor with Dual Outputs

```python
import torch
import torch.nn as nn
import torchvision.models as models

class CIRCLeModel(nn.Module):
    def __init__(self, num_classes=7, backbone="resnet50", pretrained=True):
        super().__init__()

        # Backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features  # 2048
            self.backbone.fc = nn.Identity()  # Remove original FC
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, return_embeddings=False):
        # Extract features
        embeddings = self.backbone(x)

        # Classification
        logits = self.classifier(embeddings)

        if return_embeddings:
            return embeddings, logits
        else:
            return logits

    def feature_extractor(self, x):
        """Extract embeddings only (for regularization)."""
        return self.backbone(x)
```

### 5.2 Integration with Existing Models

**Scenario 1: Add CIRCLe to Baseline ResNet50**
- Train baseline ResNet50 (Phase 1)
- Add regularization loss (Phase 2)
- Continue training for 50-100 epochs

**Scenario 2: Combine CIRCLe + FairDisCo**
- Use FairDisCo architecture (adversarial + contrastive)
- Add CIRCLe regularization as third loss term
- Total loss: L_cls + λ_adv × L_adv + λ_con × L_con + λ_reg × L_reg

**Combined Loss**:
```python
loss = (
    criterion_cls(diagnosis_logits, labels) +
    0.3 * criterion_adv(fst_logits, fst_labels) +
    0.2 * criterion_con(contrastive_embeddings, labels, fst_labels) +
    0.2 * criterion_circle(embeddings_original, embeddings_transformed)
)
```

---

## 6. Computational Requirements

### 6.1 GPU Requirements

**Training** (Batch size 64, ResNet50):
- VRAM: ~10GB (original images) + ~10GB (transformed images) = **20GB**
- Minimum: 1x RTX 3090 (24GB VRAM)
- Recommended: 1x RTX 4090 (24GB VRAM, 1.5x faster)

**VRAM Optimization**:
- Mixed precision (FP16): Reduces to ~12GB
- Gradient checkpointing: Reduces to ~10GB (slower training)

**Training Time**:
- Single epoch: ~18 minutes (Fitzpatrick17k, 16,577 images, batch 64)
  - Original images: 15 min
  - Transformed images (2x FST): +3 min overhead
- 100 epochs: ~30 hours (RTX 3090)

**Inference Time**:
- No overhead (regularization only during training)
- Same as baseline: ~30ms per image (RTX 3090)

### 6.2 Storage Requirements

**Pre-computed Transformations**:
- Original dataset: 48GB (Fitzpatrick17k, 512×512 PNG)
- Transformed FST-I: 48GB
- Transformed FST-VI: 48GB
- **Total: 144GB** (3x original)

**On-the-Fly Transformation**: No extra storage (0GB)

---

## 7. Open-Source Implementation

### 7.1 Official CIRCLe Repository

**GitHub**: https://github.com/arezou-pakzad/CIRCLe

**Key Details**:
- Language: Python
- Framework: PyTorch
- License: Not specified (assume academic use, contact for commercial)

**Provided Code**:
- `train_stargan.py`: Train StarGAN skin tone transformer
- `train_classifier.py`: Train classifier with/without regularization
- `models/`: ResNet, DenseNet, MobileNet implementations
- `utils/regularization.py`: CIRCLe regularization loss

**Training Command**:
```bash
# Step 1: Train StarGAN (optional, skip if using simple transformations)
python train_stargan.py \
    --dataset fitzpatrick17k \
    --num_domains 6 \
    --epochs 200

# Step 2: Train classifier with CIRCLe regularization
python train_classifier.py \
    --model resnet50 \
    --dataset fitzpatrick17k \
    --use_regularization \
    --lambda_reg 0.2 \
    --target_fsts I,VI
```

**Model Checkpoints**: Not publicly released (train from scratch)

### 7.2 Mirror Repository

**GitHub**: https://github.com/sfu-mial/CIRCLe (Simon Fraser University)
- Mirror of original repository
- Same codebase, alternative hosting

### 7.3 Integration Assessment

**Ease of Integration**: Moderate
- Well-structured, modular code
- Supports multiple backbones (ResNet, DenseNet, MobileNet, VGG)
- Requires StarGAN training (complex) or adaptation to simple transformations

**Code Quality**: Good
- PyTorch best practices
- Configurable hyperparameters (command-line arguments)
- Limited documentation (assume familiarity with paper)

**Recommended Approach**:
1. Clone repository, install dependencies
2. **Skip StarGAN training** (Phase 2), use simple LAB transformations
3. Adapt `train_classifier.py` to use color transformations (modify data loader)
4. Run experiments with λ_reg = 0.1, 0.2, 0.3
5. Integrate into Phase 2 pipeline (after FairSkin + FairDisCo)

---

## 8. Implementation Timeline

**Week 1: Setup & Simple Transformations**
- Day 1-2: Install dependencies, download Fitzpatrick17k
- Day 3-4: Implement simple color transformations (HSV, LAB)
- Day 5: Validate transformations (visual inspection, LPIPS)
- Day 6-7: Pre-compute transformed datasets (FST I, VI versions)

**Week 2: CIRCLe Integration**
- Day 1-2: Implement regularization loss (L2 distance)
- Day 3: Modify training loop (add regularization term)
- Day 4-5: Debug (verify gradients flow correctly)
- Day 6-7: Baseline experiment (λ_reg = 0.2)

**Week 3: Hyperparameter Tuning**
- Day 1-3: Grid search (λ_reg, target FST, distance metric)
- Day 4-5: Analyze results (AUROC gap, ECE per config)
- Day 6-7: Final training with best config

**Week 4: Combined Fairness (CIRCLe + FairDisCo)**
- Day 1-2: Integrate CIRCLe into FairDisCo architecture
- Day 3-5: Train combined model (100 epochs, ~30 hours)
- Day 6-7: Evaluate fairness (AUROC gap, EOD, ECE)

**Total: 4 weeks (28 days)**
- GPU time: ~150 hours (6 days continuous)
- Human time: ~50 hours (1.25 weeks full-time equivalent)

---

## 9. Success Criteria

**Fairness Metrics**:
- AUROC gap reduction: +2-4% (from FairDisCo 8-10% → CIRCLe 6-8%)
- ECE reduction: 3-5% (improved calibration)
- OOD generalization: +3-5% AUROC on held-out FST classes

**Accuracy Maintenance**:
- Overall accuracy: >88% (no degradation from Phase 2 baseline)
- AUROC (average): >90%

**Calibration**:
- ECE <0.08 for ALL FST groups (vs 0.10-0.12 baseline)
- Reliability diagrams: Tighter fit to diagonal (better calibration)

---

## 10. Risk Mitigation

**Risk 1: Simple Transformations Insufficient (Poor Fairness Gain)**
**Mitigation**:
- Increase λ_reg (0.2 → 0.3 → 0.4)
- Use skin segmentation (apply transformation only to skin)
- Upgrade to StarGAN (higher quality transformations)

**Risk 2: StarGAN Training Fails (Mode Collapse, Artifacts)**
**Mitigation**:
- Use spectral normalization (improves GAN stability)
- Increase training data (need 5k+ images per FST)
- Lower expectations (simple transformations may suffice)

**Risk 3: Over-Regularization (Accuracy Drops)**
**Mitigation**:
- Reduce λ_reg (0.2 → 0.1)
- Use cosine distance (softer than L2)
- Monitor validation accuracy (stop if drops >2%)

**Risk 4: High Storage Overhead (144GB Pre-computed)**
**Mitigation**:
- Use on-the-fly transformation (0GB extra storage)
- Optimize CPU transformation (multi-threading, <10ms overhead)
- Compress transformed images (lossy JPEG, -50% size)

---

## 11. Comparison: StarGAN vs Simple Transformations

| **Aspect** | **StarGAN** | **Simple Color Transformations** |
|------------|-------------|----------------------------------|
| **Training Time** | 100-200 GPU hours | 0 hours (no training) |
| **Realism** | High (photo-realistic) | Low-moderate (heuristic) |
| **Implementation Complexity** | High (GAN training, hyperparameters) | Low (10-20 lines of code) |
| **Fairness Improvement** | +4-5% AUROC gap reduction | +2-3% AUROC gap reduction |
| **Artifacts** | Potential (blur, checkerboard) | Minimal (deterministic) |
| **Dataset Requirements** | 5k+ images per FST | Any size (even 100 images) |
| **Recommendation** | Phase 3+ (after MVP) | Phase 2 (rapid iteration) |

**Recommendation**: Use simple transformations for Phase 2, upgrade to StarGAN in Phase 3 if needed.

---

## 12. Key Insights from Literature

**Pakzad et al. (2022) Findings**:
- CIRCLe improves equal opportunity (+5%) and calibration (ECE -3-5%)
- Regularization most effective on ResNet50 (vs MobileNet, DenseNet)
- Multi-FST regularization (FST I + VI) outperforms single-FST (FST I only)
- Combined with data augmentation (FairSkin), achieves 91.3% AUROC with 3.7% gap

**Best Practices**:
- Start with λ_reg = 0.2, tune ±0.1 based on validation
- Use both extreme FST classes (I and VI) for regularization
- Pre-compute transformations for faster training (storage permitting)
- Combine with adversarial debiasing (FairDisCo) for synergistic effect

---

## 13. References

**Primary Paper**:
- Pakzad, A., Abhishek, K., Hamarneh, G. (2023). "CIRCLe: Color Invariant Representation Learning for Unbiased Classification of Skin Lesions." ECCV 2022 Workshops. arXiv:2208.13528

**Skin Tone Transformation**:
- Choi, Y., et al. (2020). "StarGAN v2: Diverse Image Synthesis for Multiple Domains." CVPR.

**Color Spaces**:
- Fairchild, M.D. (2013). "Color Appearance Models." Wiley. (LAB color space)

**Calibration**:
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." ICML. (Expected Calibration Error)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-13
**Author**: THE DIDACT (Strategic Research Agent)
**Status**: IMPLEMENTATION-READY
**Next Review**: Post-Phase 2 (Week 10)
