# FairDisCo Training Guide

**Complete guide to training FairDisCo adversarial debiasing model for fairness-aware skin cancer detection**

Framework: MENDICANT_BIAS - Phase 2, Week 5-6
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-13

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Training Configuration](#training-configuration)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)
7. [Expected Performance](#expected-performance)
8. [Ablation Studies](#ablation-studies)
9. [Integration with Phase 2](#integration-with-phase-2)

---

## Quick Start

### Prerequisites

```bash
# Ensure you have HAM10000 dataset downloaded
# Expected directory structure:
# data/HAM10000/
#   ├── HAM10000_images_part_1/
#   ├── HAM10000_images_part_2/
#   └── HAM10000_metadata.csv

# Install dependencies (if not already done)
pip install torch torchvision albumentations pyyaml tensorboard scikit-learn
```

### Basic Training

```bash
# Train with default configuration
python experiments/fairness/train_fairdisco.py --config configs/fairdisco_config.yaml

# View training logs
tensorboard --logdir experiments/fairdisco/logs
```

### Expected Output

```
================================================================================
FairDisCo Training Pipeline
================================================================================

Loading configuration from configs/fairdisco_config.yaml...
Random seed set to 42

Creating data transforms...

Loading HAM10000 training dataset...
HAM10000 Dataset [train]
  Total samples: 7010
  Diagnosis distribution:
    nv    : 4461 (63.64%)
    mel   :  836 (11.93%)
    bkl   :  755 (10.77%)
    ...

Creating FairDisCo model...
Total parameters: 27,142,534
Trainable parameters: 27,142,534

Initializing FairDisCo trainer...

Starting training...
Epoch 1/100 - 892.34s
  Train: Loss=1.8234 (cls=1.7456, adv=0.0234, con=0.0544), Acc=0.3456
  Val:   Loss=1.7123, Acc=0.3789, AUROC=0.6234
  Fairness: AUROC Gap=0.1823, EOD=0.1567
  Discriminator Acc: 0.5234
  Lambda: adv=0.000, con=0.000, LR=0.000100
...
```

---

## Architecture Overview

FairDisCo uses a **three-branch architecture** with adversarial training:

### Components

```
Input Image (224×224×3)
        |
        v
┌──────────────────────┐
│  ResNet50 Backbone   │  ← Pre-trained on ImageNet
│  (Feature Extractor) │     Output: 2048-dim embeddings
└──────────────────────┘
        |
    ┌───┴───┬───────────┐
    v       v           v
┌────────┐ ┌────────┐ ┌──────────┐
│ Cls    │ │  GRL   │ │Contrast. │
│ Head   │ │  +     │ │ Head     │
│        │ │ Disc.  │ │          │
└────────┘ └────────┘ └──────────┘
    |         |           |
    v         v           v
Diagnosis   FST       Feature
(7 cls)    (6 cls)    Alignment
```

### Three Losses

1. **Classification Loss (L_cls)**: Standard cross-entropy for diagnosis prediction
   - Weight: λ_cls = 1.0 (implicit)
   - Target: Accurate diagnosis (melanoma, nevus, etc.)

2. **Adversarial Loss (L_adv)**: Cross-entropy for FST prediction with gradient reversal
   - Weight: λ_adv = 0.3 (configurable)
   - Target: Force embeddings to be FST-invariant
   - Mechanism: Gradient Reversal Layer (GRL)

3. **Contrastive Loss (L_con)**: Supervised contrastive learning
   - Weight: λ_con = 0.2 (configurable)
   - Target: Pull same-diagnosis embeddings together
   - Key: Only same-diagnosis, **different-FST** pairs are positives

**Total Loss**:
```
L_total = L_cls + λ_adv × L_adv + λ_con × L_con
```

### Gradient Reversal Layer (GRL)

The GRL is the core mechanism for adversarial debiasing:

**Forward Pass**: Identity operation (y = x)
**Backward Pass**: Reverse gradient (∂L/∂x = -λ × ∂L/∂y)

**Effect**:
- Discriminator learns to predict FST (normal gradient)
- Backbone learns to **prevent** FST prediction (reversed gradient)
- Equilibrium: Discriminator accuracy → random chance (~17% for 6 classes)

---

## Training Configuration

### Core Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 100 | 50-150 | Total training epochs |
| `batch_size` | 64 | 32-128 | **Minimum 64** for contrastive loss |
| `learning_rate` | 1e-4 | 1e-5 to 1e-3 | Initial learning rate |
| `lambda_adv` | 0.3 | 0.1-0.5 | Adversarial loss weight |
| `lambda_con` | 0.2 | 0.1-0.3 | Contrastive loss weight |
| `temperature` | 0.07 | 0.05-0.1 | Contrastive loss temperature |

### Lambda Scheduling

**Purpose**: Prevent training instability in early epochs

**Schedule**:
1. **Warmup (Epochs 1-20)**: λ_adv = 0.0, λ_con = 0.0
   - Learn basic classification features first
   - No adversarial or contrastive training

2. **Ramp-up (Epochs 20-40)**: λ_adv: 0.1 → 0.3, λ_con: 0.0 → 0.2
   - Linearly increase lambda values
   - Gradually introduce debiasing

3. **Full Training (Epochs 40+)**: λ_adv = 0.3, λ_con = 0.2
   - Use target lambda values
   - Full adversarial training

**Configuration**:
```yaml
training:
  use_lambda_schedule: true
  lambda_schedule_start_epoch: 20
  lambda_schedule_end_epoch: 40
  lambda_schedule_start_value: 0.1
  lambda_schedule_end_value: 0.3
```

### Optimizer & Scheduler

**AdamW Optimizer**:
```yaml
optimizer: "adamw"
learning_rate: 0.0001
weight_decay: 0.01  # L2 regularization
```

**Cosine Annealing with Warm Restarts**:
```yaml
scheduler: "cosine_warm"
scheduler_t0: 20    # Restart every 20 epochs
scheduler_t_mult: 2  # Double period after restart
scheduler_eta_min: 0.000001  # Min LR (1e-6)
```

**Learning Rate Schedule**:
```
LR = 1e-4

Epoch:  0----20---40---60---80--100
        |    /\   /  \  /    \
        |   /  \ /    \/      \
        |  /    X      X       \
        | /      \    /         \___
        |/        \__/              \___ (1e-6)
```

---

## Hyperparameter Tuning

### Tuning Lambda_adv (Adversarial Strength)

**Goal**: Balance fairness vs accuracy

| λ_adv | Fairness (EOD) | Accuracy | Use Case |
|-------|---------------|----------|----------|
| 0.1 | Moderate (0.10-0.12) | High (-0.5%) | Conservative, accuracy-first |
| 0.3 | Good (0.06-0.08) | Good (-1-2%) | **Default, balanced** |
| 0.5 | Excellent (0.04-0.06) | Lower (-3-5%) | Aggressive debiasing |

**Tuning Strategy**:

1. **Start with default** (λ_adv = 0.3)
2. **Monitor discriminator accuracy** (target: 20-25%)
   - If disc_acc > 50% after epoch 50: **Increase λ_adv** (0.3 → 0.4)
   - If disc_acc < 15%: **Decrease λ_adv** (0.3 → 0.2)

3. **Check accuracy trade-off**:
   - If accuracy drop > 3%: **Reduce λ_adv**
   - If EOD > 0.08: **Increase λ_adv**

### Tuning Lambda_con (Contrastive Strength)

**Goal**: Maintain feature quality while debiasing

| λ_con | Feature Quality | Accuracy | Use Case |
|-------|----------------|----------|----------|
| 0.1 | Moderate | Baseline | Minimal contrastive |
| 0.2 | Good | +0.5-1% | **Default** |
| 0.3 | Excellent | +1-2% | Strong feature learning |

**When to increase λ_con**:
- Accuracy dropping too much with adversarial training
- Need better feature representations
- Contrastive loss is decreasing well

### Tuning Temperature (τ)

**Goal**: Control separation between positive/negative pairs

| τ | Effect | Use Case |
|---|--------|----------|
| 0.05 | Strict separation | Very similar classes |
| 0.07 | **Default** | Standard |
| 0.10 | Relaxed separation | More diverse embeddings |

**Typically not tuned** - 0.07 is standard from SimCLR

### Batch Size Considerations

**Critical for contrastive loss**: Need enough positive pairs per batch

| Batch Size | Positive Pairs | Training Speed | Recommendation |
|-----------|----------------|----------------|----------------|
| 32 | Low (5-10) | Fast | Not recommended |
| 64 | Good (15-20) | Medium | **Minimum recommended** |
| 128 | Excellent (30-40) | Slow | Ideal if GPU memory allows |

**GPU Memory Requirements**:
- Batch 64: ~12GB VRAM (RTX 3090)
- Batch 128: ~24GB VRAM (A100 40GB)
- Use gradient accumulation if limited memory

---

## Monitoring Training

### Key Metrics to Track

1. **Discriminator Accuracy** (Most important for adversarial training)
   ```
   Target: 20-25% (near random chance 1/6 = 16.7%)

   Ideal progression:
   Epoch 1-10:   50-70% (discriminator learning)
   Epoch 10-30:  40-30% (backbone learning invariance)
   Epoch 30-50:  25-20% (equilibrium reached)
   Epoch 50+:    20-25% (stable)
   ```

2. **Three Losses**
   ```
   L_cls:  Should decrease steadily (1.8 → 0.4)
   L_adv:  Should increase initially, then plateau
   L_con:  Should decrease (1.5 → 0.5)
   L_total: Should decrease overall
   ```

3. **Fairness Metrics**
   ```
   AUROC Gap:  Target <0.08 (baseline ~0.18)
   EOD:        Target <0.06 (baseline ~0.18)
   ```

4. **Validation AUROC**
   ```
   Target: >0.88 (maintain accuracy)
   Acceptable drop: <3% from baseline
   ```

### TensorBoard Visualization

```bash
tensorboard --logdir experiments/fairdisco/logs
```

**Important Plots**:
- `Loss/train_adv` vs `Discriminator/accuracy` (should be inversely correlated)
- `Fairness/auroc_gap` (should decrease over time)
- `Fairness/eod` (should decrease to <0.08)
- `Lambda/adv` and `Lambda/con` (verify schedule)

### Example Good Training Run

```
Epoch 10:  Disc Acc = 0.68, AUROC Gap = 0.18, EOD = 0.17
Epoch 20:  Disc Acc = 0.55, AUROC Gap = 0.16, EOD = 0.15  ← Lambda ramp-up starts
Epoch 30:  Disc Acc = 0.40, AUROC Gap = 0.12, EOD = 0.11
Epoch 40:  Disc Acc = 0.28, AUROC Gap = 0.09, EOD = 0.08  ← Full lambda
Epoch 50:  Disc Acc = 0.23, AUROC Gap = 0.08, EOD = 0.06  ← Target reached
Epoch 60:  Disc Acc = 0.22, AUROC Gap = 0.08, EOD = 0.06  ← Stable
```

### Example Bad Training Run (Discriminator Too Strong)

```
Epoch 10:  Disc Acc = 0.72, AUROC Gap = 0.18, EOD = 0.17
Epoch 20:  Disc Acc = 0.68, AUROC Gap = 0.17, EOD = 0.16
Epoch 30:  Disc Acc = 0.65, AUROC Gap = 0.16, EOD = 0.15
Epoch 40:  Disc Acc = 0.61, AUROC Gap = 0.15, EOD = 0.14  ← Not improving!
Epoch 50:  Disc Acc = 0.58, AUROC Gap = 0.14, EOD = 0.13

ACTION: Increase λ_adv from 0.3 to 0.4
```

---

## Troubleshooting

### Issue 1: Discriminator Too Strong (Accuracy > 50% after epoch 50)

**Symptoms**:
- Discriminator accuracy stays high (>50%)
- AUROC gap not decreasing
- EOD not improving

**Solutions**:
1. **Increase λ_adv**: 0.3 → 0.4 → 0.5
2. **Start adversarial training earlier**: Set `lambda_schedule_start_epoch: 10`
3. **Use stronger discriminator**: Add 4th layer to FST_Discriminator
4. **Check data**: Ensure FST labels are correct

**Config Change**:
```yaml
training:
  lambda_adv: 0.4  # Increase from 0.3
  lambda_schedule_start_value: 0.2  # Start higher
```

### Issue 2: Discriminator Too Weak (Accuracy < 15%)

**Symptoms**:
- Discriminator accuracy drops too low (<15%)
- Accuracy dropping significantly (>5%)
- Training unstable

**Solutions**:
1. **Decrease λ_adv**: 0.3 → 0.2 → 0.1
2. **Slower ramp-up**: Extend `lambda_schedule_end_epoch: 60`
3. **Increase contrastive weight**: λ_con: 0.2 → 0.3
4. **Reduce gradient clip**: `gradient_clip_norm: 1.0 → 2.0`

**Config Change**:
```yaml
training:
  lambda_adv: 0.2
  lambda_con: 0.3
  gradient_clip_norm: 2.0
```

### Issue 3: Accuracy Dropping Too Much (>3%)

**Symptoms**:
- Validation AUROC <0.85 (vs baseline 0.88-0.90)
- Classification loss not decreasing well

**Solutions**:
1. **Reduce λ_adv**: 0.3 → 0.2
2. **Increase λ_con**: 0.2 → 0.3 (compensate with feature quality)
3. **Longer warmup**: `lambda_schedule_start_epoch: 30`
4. **Lower learning rate**: 1e-4 → 5e-5

### Issue 4: Contrastive Loss Not Decreasing

**Symptoms**:
- L_con plateaus at high value (>1.0)
- Feature embeddings not clustering

**Solutions**:
1. **Increase batch size**: 64 → 128 (more positive pairs)
2. **Check FST label distribution**: Need balanced FST in each batch
3. **Adjust temperature**: 0.07 → 0.10 (relax constraints)
4. **Verify embeddings**: Check if normalized correctly

### Issue 5: Training Diverges (NaN losses)

**Symptoms**:
- Loss becomes NaN
- Gradients explode

**Solutions**:
1. **Lower learning rate**: 1e-4 → 5e-5
2. **Stronger gradient clipping**: 1.0 → 0.5
3. **Disable AMP** (mixed precision): `use_amp: false`
4. **Check data**: Look for corrupted images or extreme values
5. **Reduce lambda values**: Start with λ_adv=0.1, λ_con=0.1

### Issue 6: Out of Memory (OOM)

**Solutions**:
1. **Reduce batch size**: 64 → 32
2. **Use gradient accumulation**:
   ```yaml
   training:
     batch_size: 32
     gradient_accumulation_steps: 2  # Effective batch: 64
   ```
3. **Enable mixed precision**: `use_amp: true` (saves ~50% memory)
4. **Reduce image size**: 224 → 192 (not recommended)

---

## Expected Performance

### Baseline (No Fairness Techniques)

| Metric | FST I-III | FST IV-VI | Gap |
|--------|----------|-----------|-----|
| AUROC | 0.91 | 0.73 | **0.18** |
| Sensitivity | 0.85 | 0.68 | 0.17 |
| EOD | - | - | **0.18** |

### FairDisCo (After 100 Epochs)

| Metric | FST I-III | FST IV-VI | Gap | Improvement |
|--------|----------|-----------|-----|-------------|
| AUROC | 0.89 | 0.83 | **0.06** | **65% reduction** |
| Sensitivity | 0.84 | 0.80 | 0.04 | 76% reduction |
| EOD | - | - | **0.06** | **65% reduction** |
| Overall Acc | 0.88 | 0.88 | 0.00 | -1% (acceptable) |

### Training Time

| Hardware | Batch Size | Time per Epoch | Total (100 epochs) |
|----------|-----------|---------------|-------------------|
| RTX 3090 (24GB) | 64 | 15 min | **25 hours** |
| RTX 4090 (24GB) | 64 | 10 min | 17 hours |
| A100 40GB | 128 | 8 min | 13 hours |
| A100 80GB | 256 | 6 min | 10 hours |

---

## Ablation Studies

Once training is complete, run ablation studies to understand component contributions:

### Study 1: Adversarial Only (No Contrastive)

```yaml
training:
  lambda_adv: 0.3
  lambda_con: 0.0  # Disable contrastive
```

**Expected**: EOD ~0.10 (44% reduction), Accuracy -2%

### Study 2: Contrastive Only (No Adversarial)

```yaml
training:
  lambda_adv: 0.0  # Disable adversarial
  lambda_con: 0.2
```

**Expected**: EOD ~0.12 (33% reduction), Accuracy +1%

### Study 3: Full FairDisCo (Both)

```yaml
training:
  lambda_adv: 0.3
  lambda_con: 0.2
```

**Expected**: EOD ~0.06 (65% reduction), Accuracy -0.5%

**Insight**: Adversarial + Contrastive are **complementary**
- Adversarial: Removes FST signal
- Contrastive: Maintains diagnostic signal
- Combined: Best fairness with minimal accuracy loss

---

## Integration with Phase 2

FairDisCo is **Week 5-6** of Phase 2. After completion:

### Week 7-8: Add CIRCLe

Integrate color-invariant regularization:
```python
# In fairdisco_trainer.py, add 4th loss term
loss_reg = color_invariant_loss(embeddings, transformed_images)
loss = loss_cls + λ_adv * loss_adv + λ_con * loss_con + λ_reg * loss_reg
```

### Week 9-11: Add FairSkin Augmentation

Train on mixed real + synthetic data:
```yaml
data:
  dataset: "ham10000_fairskin"  # Includes synthetic
  synthetic_ratio: 0.5  # 50% synthetic for FST V-VI
```

### Week 12: Final Evaluation

Train combined model and report:
- AUROC gap: Target <4%
- EOD: Target <0.05
- Overall accuracy: >88%

---

## Conclusion

FairDisCo provides a powerful adversarial debiasing approach for fair skin cancer detection. Key takeaways:

1. **Lambda scheduling is critical** - Don't start adversarial training too early
2. **Monitor discriminator accuracy** - Should reach ~20-25% equilibrium
3. **Batch size matters** - Minimum 64 for contrastive loss
4. **Expect 1-2% accuracy trade-off** - But 65% fairness improvement
5. **Combine with other techniques** (CIRCLe, FairSkin) for Phase 2 MVP

For questions or issues, refer to:
- Research documentation: `docs/fairdisco_architecture.md`
- Model code: `src/models/fairdisco_model.py`
- Trainer code: `src/training/fairdisco_trainer.py`

**Next Steps**: Proceed to Week 7-8 (CIRCLe implementation)
