# Synthetic Data Augmentation for Fairness-Aware Skin Cancer Detection

**Version**: 1.0
**Date**: 2025-10-13
**Framework**: MENDICANT_BIAS - the_didact research division
**Purpose**: Implement tone-conditioned diffusion models to generate synthetic dermoscopic images with balanced Fitzpatrick Skin Type (FST) distribution

---

## Executive Summary

**Problem**: Existing dermatology datasets exhibit severe FST imbalance (<5% FST V-VI), leading to 15-30% AUROC performance drops for darker skin tones.

**Solution**: Generate 60,000 high-fidelity synthetic images using diffusion models (FairSkin, DermDiff, or custom Stable Diffusion), targeting 25% FST V-VI representation.

**Expected Impact** (from literature):
- +18-21% AUROC improvement for FST VI (FairSkin benchmark)
- Reduced Equal Opportunity Difference (EOD): 65% reduction
- Improved calibration and generalization across all FST groups

**Challenges**:
- Quality validation (FID <20, LPIPS <0.1)
- Preservation of diagnostic features
- Avoiding synthetic artifacts that reduce clinical trust
- Computational cost (48-72 hours GPU training for diffusion models)

---

## 1. Synthetic Augmentation Strategy

### 1.1 Why Synthetic Data?

**Advantages**:
1. **Scalability**: Generate unlimited images without patient recruitment
2. **Control**: Precise FST distribution targeting
3. **Privacy**: No real patient data, no consent issues
4. **Speed**: Faster than multi-year prospective studies

**Limitations** (Must Address)**:
1. **Distribution Shift**: Synthetic images may not fully capture real-world diversity
2. **Artifacts**: GAN/Diffusion models can introduce unrealistic features
3. **Clinician Trust**: Medical professionals skeptical of synthetic data
4. **Validation Required**: Must prove synthetic data improves real-world performance

### 1.2 Literature Benchmarks

| Study | Method | Synthetic Data Size | FST Balance | Impact on FST V-VI AUROC |
|-------|--------|---------------------|-------------|--------------------------|
| **FairSkin** (Ju et al., 2024) | Diffusion (3-level resampling) | 60k images | 25% FST V-VI | +18-21% |
| **DermDiff** (2025) | Text-conditioned diffusion | 60k images | 30k benign, 30k malignant | Significant bias reduction |
| **From Majority to Minority** (Wang et al., 2024) | Stable Diffusion (LoRA) | Variable | Targeted minority augmentation | MICCAI Workshop Honorable Mention |
| **BiaslessNAS** (Pacal et al., 2025) | NAS + synthetic | 40k images | 30% FST V-VI | <4% AUROC gap |

**Consensus**: 40-60k synthetic images with balanced FST distribution is the sweet spot for fairness improvement.

---

## 2. Diffusion Model Architectures

### 2.1 FairSkin (Recommended for Quality)

**Paper**: Ju et al. (2024). *FairSkin: Fair Diffusion for Skin Disease Image Generation*. arXiv:2410.22551

**Key Innovation**: Three-level resampling mechanism
1. **Resampling**: Balanced sampling across racial/disease categories
2. **Class Diversity Loss**: Ensures quality representation of underrepresented groups
3. **Imbalance-Aware Augmentation**: Dynamic reweighting during training

**Architecture**: Stable Diffusion base with custom conditioning
- Condition 1: Disease class (melanoma, nevus, BCC, etc.)
- Condition 2: Fitzpatrick Skin Type (I-VI)
- Condition 3: Lesion characteristics (shape, border, color)

**Training Strategy**:
- Pre-train on HAM10000 + ISIC 2019 (70k real images)
- Fine-tune with resampling (oversample FST V-VI by 5x)
- Loss: L_total = L_diffusion + λ_1 * L_class_diversity + λ_2 * L_fairness

**Code Availability**: NOT YET RELEASED (paper Oct 2024)
- **Alternative**: Implement using Hugging Face Diffusers + custom conditioning

### 2.2 DermDiff (Recommended for Racial Bias Mitigation)

**Paper**: *DermDiff: Generative Diffusion Model for Mitigating Racial Biases in Dermatology Diagnosis*. arXiv:2503.17536

**Key Innovation**: Skin tone detector + multimodal conditioning
1. **Skin Tone Detector**: Automated FST classification (ResNet-based)
2. **Text Prompting**: "Melanoma on Fitzpatrick Type VI skin"
3. **Multimodal Learning**: Combines image features + text embeddings

**Architecture**: Latent Diffusion Model (LDM)
- Base: Stable Diffusion v1.5 or v2.1
- Conditioning: CLIP text encoder + FST embedding
- VAE: Encode images to latent space (4x downsampling)

**Implementation Details**:
- Framework: PyTorch + Hugging Face Diffusers
- Training: 48-72 hours on 4x A100 GPUs (80GB VRAM)
- Dataset: Fitzpatrick17k (16,577 images with FST labels)

**Generated Dataset**:
- 60k synthetic images (30k benign, 30k malignant)
- Balanced FST distribution: 10k per FST type (I-VI)

**Code Availability**: Likely available (check arXiv code link or contact authors)

### 2.3 From Majority to Minority (Recommended for Implementation Speed)

**Paper**: Wang et al. (2024). *From Majority to Minority: A Diffusion-based Augmentation for Underrepresented Groups in Skin Lesion Analysis*. arXiv:2406.18375

**GitHub**: ✅ https://github.com/janet-sw/skin-diff

**Key Innovation**: Textual Inversion + LoRA for minority augmentation
1. **Textual Inversion**: Learn token embeddings for specific FST + disease combinations
2. **LoRA (Low-Rank Adaptation)**: Fine-tune Stable Diffusion with minimal parameters
3. **Targeted Generation**: Generate ONLY underrepresented categories (FST V-VI)

**Advantages**:
- FASTEST implementation (code available, well-documented)
- Minimal computational cost (LoRA requires <10% parameters vs full fine-tuning)
- Proven effective (MICCAI 2024 Honorable Mention)

**Training Time**:
- Textual Inversion: 2-4 hours on single A100 GPU
- LoRA fine-tuning: 4-8 hours on single A100 GPU
- Total: <12 hours (vs 48-72 hours for full diffusion training)

**Recommended for Phase 2 MVP**: Use this approach to quickly generate minority samples.

---

## 3. Implementation Roadmap

### Phase 2A: Quick-Start Minority Augmentation (Week 5-6)

**Objective**: Generate 10,000 FST V-VI synthetic images using existing code

**Steps**:
1. **Clone Repository**:
   ```bash
   git clone https://github.com/janet-sw/skin-diff.git
   cd skin-diff
   pip install -r requirements.txt
   ```

2. **Prepare Training Data**:
   - Extract FST V-VI images from Fitzpatrick17k, DDI, SCIN
   - Target: 500-1,000 real FST V-VI images
   - Organize by diagnosis: melanoma, nevus, BCC, etc.

3. **Train Textual Inversion**:
   ```bash
   python train_textual_inversion.py \
       --dataset_path data/fst_v_vi/ \
       --token "fst_dark_skin" \
       --num_steps 3000 \
       --lr 5e-4
   ```

4. **Train LoRA**:
   ```bash
   python train_lora.py \
       --base_model "CompVis/stable-diffusion-v1-4" \
       --dataset_path data/fst_v_vi/ \
       --textual_inversion_token "fst_dark_skin" \
       --lora_rank 4 \
       --num_steps 5000
   ```

5. **Generate Synthetic Images**:
   ```bash
   python generate.py \
       --prompt "melanoma on [fst_dark_skin] skin" \
       --num_images 10000 \
       --guidance_scale 7.5 \
       --output_dir data/synthetic/fst_v_vi/
   ```

**Timeline**: 2 weeks (including setup, training, generation)
**GPU Requirements**: 1x A100 (40GB VRAM) or 1x V100 (32GB VRAM)

### Phase 2B: Full Fairness Augmentation (Week 7-10)

**Objective**: Generate 60,000 balanced synthetic images (10k per FST type)

**Approach 1: Extend LoRA Method**:
- Train 6 separate LoRA models (one per FST type)
- Generate 10k images per FST type
- Merge with real data for balanced dataset

**Approach 2: Implement FairSkin (If Time Permits)**:
- Custom Stable Diffusion conditioning with FST embeddings
- Three-level resampling mechanism
- Higher quality but longer training time

**Recommended**: Approach 1 (LoRA) for MVP, Approach 2 (FairSkin) for Phase 3 refinement.

---

## 4. Quality Validation

### 4.1 Quantitative Metrics

**Frechet Inception Distance (FID)**:
- **Definition**: Measures distance between real and synthetic image feature distributions
- **Target**: FID <20 (clinical acceptability threshold from literature)
- **Implementation**:
  ```python
  from pytorch_fid import fid_score
  fid_value = fid_score.calculate_fid_given_paths(
      [real_images_path, synthetic_images_path],
      batch_size=50,
      device='cuda',
      dims=2048
  )
  print(f"FID Score: {fid_value}")
  ```

**Learned Perceptual Image Patch Similarity (LPIPS)**:
- **Definition**: Perceptual similarity between real and synthetic images
- **Target**: LPIPS <0.1 (high perceptual similarity)
- **Implementation**:
  ```python
  import lpips
  loss_fn = lpips.LPIPS(net='alex')
  d = loss_fn(real_img, synthetic_img)
  ```

**Inception Score (IS)**:
- **Definition**: Measures diversity and quality of generated images
- **Target**: IS >5 (for dermatology images)

### 4.2 Qualitative Validation

**Expert Dermatologist Review**:
- Sample 100 random synthetic images
- Blind evaluation: Mix 50 real + 50 synthetic
- Task: Classify as "real" or "synthetic", rate realism (1-7 scale)
- **Target**: >60% cannot distinguish real vs synthetic, mean realism score >5

**Clinical Feature Checklist** (ABCD Rule Validation):
- **A**: Asymmetry preserved in synthetic images?
- **B**: Border irregularity realistic?
- **C**: Color variation matches real lesions?
- **D**: Diameter appropriate for lesion type?

**FST Fidelity Check**:
- Run ITA algorithm on synthetic images
- Verify FST label matches generated ITA value
- **Target**: >90% agreement (within 1 FST category)

### 4.3 Downstream Task Validation

**Critical Test**: Does synthetic data improve real-world performance?

**Experiment Design**:
1. **Baseline Model**: Train on real data only (HAM10000 + ISIC)
2. **Augmented Model**: Train on real + 60k synthetic (balanced FST)
3. **Test Set**: DDI (656 images, 34% FST V-VI) - NEVER seen during training

**Success Criteria**:
- Augmented model AUROC (FST V-VI) > Baseline model AUROC (FST V-VI) by 10%+
- No degradation in FST I-III performance (AUROC drop <2%)

**Failure Mode**: If synthetic data degrades performance, diagnose:
- FID too high (>30): Quality insufficient
- Distribution shift: Synthetic images too different from test set
- Overfitting to synthetic artifacts

---

## 5. Technical Implementation

### 5.1 Hugging Face Diffusers Setup

**Installation**:
```bash
pip install diffusers transformers accelerate safetensors
```

**Basic Stable Diffusion Pipeline**:
```python
from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Generate image
prompt = "melanoma on Fitzpatrick Type VI skin, dermoscopy, high quality"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("synthetic_melanoma_fst6.png")
```

### 5.2 FST-Conditioned Generation

**Custom Conditioning Module**:
```python
import torch.nn as nn

class FSTConditioningModule(nn.Module):
    def __init__(self, fst_embedding_dim=128):
        super().__init__()
        # Learnable FST embeddings (6 types: I-VI)
        self.fst_embeddings = nn.Embedding(6, fst_embedding_dim)

    def forward(self, fst_labels):
        """
        Args:
            fst_labels: Tensor of FST labels (0-5 for FST I-VI)
        Returns:
            fst_embeds: FST embeddings for conditioning
        """
        return self.fst_embeddings(fst_labels)
```

**Integrate with Stable Diffusion**:
- Add FST embeddings to cross-attention layers
- Concatenate with CLIP text embeddings
- Fine-tune with FST-labeled data

### 5.3 Training Loop (Simplified)

```python
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch.optim as optim

# Initialize models
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
fst_module = FSTConditioningModule()
noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

optimizer = optim.AdamW(list(unet.parameters()) + list(fst_module.parameters()), lr=1e-5)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images, fst_labels, text_prompts = batch

        # Add noise to images (diffusion forward process)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],))
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        # Get FST conditioning
        fst_embeds = fst_module(fst_labels)

        # Predict noise
        noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=fst_embeds).sample

        # Loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. Data Pipeline Integration

### 6.1 Mixing Real + Synthetic Data

**Strategy 1: Pre-Training on Synthetic**:
1. Pre-train model on 60k synthetic (balanced FST)
2. Fine-tune on real data (HAM10000, ISIC, Fitzpatrick17k)
3. **Advantage**: Model learns FST-invariant features early

**Strategy 2: Mixed Training**:
1. Combine real (70k) + synthetic (60k) = 130k total
2. Train single model on mixed dataset
3. **Advantage**: Simpler pipeline, better calibration

**Strategy 3: Two-Stage with Reweighting**:
1. Train on mixed data
2. Apply loss reweighting: Higher weight for real data
3. **Advantage**: Prioritizes real data while benefiting from synthetic diversity

**Recommended**: Strategy 2 (Mixed Training) for Phase 2 MVP.

### 6.2 Dataset Class Implementation

```python
class MixedDermoscopyDataset(torch.utils.data.Dataset):
    def __init__(self, real_data_path, synthetic_data_path, transform=None):
        self.real_images = load_images(real_data_path)
        self.synthetic_images = load_images(synthetic_data_path)
        self.transform = transform

        # Mark data source
        self.real_labels = [(img, label, 'real') for img, label in self.real_images]
        self.synthetic_labels = [(img, label, 'synthetic') for img, label in self.synthetic_images]

        # Combine
        self.data = self.real_labels + self.synthetic_labels

    def __getitem__(self, idx):
        image, label, source = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, source

    def __len__(self):
        return len(self.data)
```

---

## 7. Computational Requirements

### 7.1 GPU Resources

**Training**:
- **Textual Inversion + LoRA**: 1x A100 (40GB), 12 hours
- **Full FairSkin Diffusion**: 4x A100 (80GB), 48-72 hours
- **DermDiff**: 4x A100 (80GB), 48-72 hours

**Generation**:
- 10k images: 4-6 hours on 1x A100
- 60k images: 24-36 hours on 1x A100 (or 6-9 hours on 4x A100)

**Total Cost Estimate** (Cloud GPU):
- AWS p4d.24xlarge (8x A100): $32.77/hour
- Training (72 hours): ~$2,360
- Generation (36 hours): ~$1,180
- **Total**: ~$3,500 for full diffusion training

**Budget Alternative** (LoRA approach):
- Training (12 hours): ~$400
- Generation (36 hours): ~$1,180
- **Total**: ~$1,600 (60% cost reduction)

### 7.2 Storage Requirements

- Real data (HAM10000 + ISIC): ~5GB
- Synthetic data (60k images, 224x224): ~4GB (JPEG compressed)
- Model checkpoints (Stable Diffusion + LoRA): ~8GB
- **Total**: ~20GB storage

---

## 8. Risks & Mitigation

### Risk 1: Synthetic Artifacts Reduce Clinician Trust

**Mitigation**:
- Expert validation with dermatologists
- Transparent disclosure: Label synthetic images in training pipeline
- Ablation study: Report performance with/without synthetic data

### Risk 2: Distribution Shift (Synthetic ≠ Real)

**Mitigation**:
- FID/LPIPS monitoring during generation
- Test on held-out REAL data (DDI, MIDAS) never seen during training
- If performance degrades, reduce synthetic data ratio

### Risk 3: Insufficient Diversity (Mode Collapse)

**Mitigation**:
- Use class diversity loss (FairSkin approach)
- Generate from multiple random seeds
- Visual inspection: Check for repetitive patterns

### Risk 4: Computational Cost Overrun

**Mitigation**:
- Start with LoRA approach (cheaper)
- Use Google Colab Pro+ (A100 access for $50/month)
- Reduce generation to 40k images (still effective per literature)

---

## 9. Success Metrics

**Phase 2 Targets**:
- ✅ 60k synthetic images generated
- ✅ FID <20, LPIPS <0.1
- ✅ Expert realism score >5/7
- ✅ AUROC improvement (FST V-VI): +10% vs baseline
- ✅ No degradation in FST I-III performance (<2% drop)

**Go/No-Go Decision** (Week 10):
- If synthetic data improves fairness: Proceed to Phase 3 (hybrid architecture)
- If no improvement: Investigate failure mode, consider alternative augmentation

---

## 10. Alternative Augmentation Methods (If Diffusion Fails)

### 10.1 GAN-Based Augmentation

**StyleGAN3** (NVIDIA, 2021):
- Proven for high-fidelity image generation
- Faster training than diffusion (24-48 hours)
- Less control over FST conditioning

### 10.2 Traditional Augmentation (Baseline)

**RandAugment** + color jittering:
- No synthetic data generation
- Simple transformations: rotation, flip, brightness adjustment
- **Limitation**: Cannot create new FST V-VI samples

### 10.3 Conditional VAE

**Variational Autoencoder with FST conditioning**:
- Faster than diffusion (8-12 hours training)
- Lower quality than diffusion (blurrier images)
- Useful for latent space interpolation

---

## 11. References

**Key Papers**:
1. Ju, L., et al. (2024). *FairSkin: Fair Diffusion for Skin Disease Image Generation*. arXiv:2410.22551.
2. *DermDiff: Generative Diffusion Model for Mitigating Racial Biases in Dermatology Diagnosis*. arXiv:2503.17536.
3. Wang, J., et al. (2024). *From Majority to Minority: A Diffusion-based Augmentation for Underrepresented Groups in Skin Lesion Analysis*. arXiv:2406.18375. [GitHub: janet-sw/skin-diff]
4. Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. [Stable Diffusion]

**Quality Metrics**:
5. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS 2017. [FID metric]
6. Zhang, R., et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018. [LPIPS metric]

**Tools & Libraries**:
- Hugging Face Diffusers: https://github.com/huggingface/diffusers
- PyTorch FID: https://github.com/mseitzer/pytorch-fid
- LPIPS: https://github.com/richzhang/PerceptualSimilarity

---

## 12. Timeline Summary

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 5 | Clone skin-diff repo, prepare FST V-VI dataset | 500-1000 FST V-VI images organized |
| 6 | Train Textual Inversion + LoRA | LoRA checkpoint + 10k synthetic images |
| 7 | Scale to 60k generation, quality validation | 60k synthetic images, FID/LPIPS report |
| 8 | Train baseline model (real only) + augmented model (real + synthetic) | Two trained models |
| 9 | Evaluate on DDI test set, compare AUROC per FST | Comparative performance report |
| 10 | Go/No-Go decision, documentation | `experiments/synthetic_augmentation/` results |

---

**Contact**:
- Research Lead: the_didact@mendicant-bias.ai
- GPU Resources: [Cloud provider contact]
- Dermatologist Validation: [Expert panel email]

**Version Control**:
- v1.0 (2025-10-13): Initial research and implementation plan

**Maintained by**: the_didact (MENDICANT_BIAS framework)
**Next Review**: 2025-10-20 (post-Phase 2A kickoff)
