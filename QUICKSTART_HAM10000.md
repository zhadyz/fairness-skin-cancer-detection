# HAM10000 Quick Start Guide

**Phase 1.5 Complete** - Real dataset integration ready for baseline training

---

## 5-Minute Setup

### 1. Download HAM10000 Dataset

Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Download:
- `HAM10000_images_part_1.zip` (~2.5 GB)
- `HAM10000_images_part_2.zip` (~2.5 GB)
- `HAM10000_metadata` (CSV)

Extract to: `data/raw/ham10000/`

### 2. Automated Setup

```bash
# One command to set up everything
python scripts/setup_ham10000.py

# This will:
#   - Check dataset integrity
#   - Generate FST annotations (ITA-based)
#   - Create train/val/test splits (stratified)
#   - Verify all is ready
```

### 3. Train Baseline Model

```bash
# Train ResNet50 on HAM10000 with FST evaluation
python experiments/baseline/train_resnet50.py --config configs/baseline_config.yaml
```

Done! Your model will train with fairness metrics tracked across FST groups.

---

## What's Included

### Core Implementation

| Component | File | Description |
|-----------|------|-------------|
| **Dataset Class** | `src/data/ham10000_dataset.py` | PyTorch Dataset with FST support |
| **FST Estimation** | `scripts/generate_ham10000_fst.py` | ITA-based skin tone annotation |
| **Stratified Splits** | `scripts/create_ham10000_splits.py` | Train/val/test with no leakage |
| **Verification** | `scripts/verify_ham10000.py` | Complete integrity check |
| **Setup Script** | `scripts/setup_ham10000.py` | Automated end-to-end setup |
| **Documentation** | `docs/ham10000_integration.md` | Comprehensive guide |

### Key Features

- **Automatic FST Annotation**: ITA-based estimation for all 10,015 images
- **Stratified Splitting**: Preserves diagnosis AND FST distribution
- **No Data Leakage**: Lesion-level splitting (same lesion stays in one split)
- **Comprehensive Metadata**: Age, sex, localization, lesion_id, FST
- **Graceful Fallbacks**: Uses dummy data if HAM10000 not available
- **Fairness Evaluation**: Built-in FST-stratified metrics

---

## Directory Structure After Setup

```
data/
├── raw/
│   └── ham10000/
│       ├── HAM10000_images_part_1/    # 5000 images
│       ├── HAM10000_images_part_2/    # 5015 images
│       └── HAM10000_metadata.csv
└── metadata/
    ├── ham10000_fst_estimated.csv     # Metadata + FST labels
    ├── ham10000_splits.json           # Train/val/test indices
    └── visualizations/                # FST distribution plots
        ├── ham10000_fst_distribution.png
        ├── ham10000_ita_distribution.png
        └── ham10000_fst_by_diagnosis.png
```

---

## Usage Example

```python
from src.data.ham10000_dataset import HAM10000Dataset, load_splits
from torch.utils.data import DataLoader

# Load splits
splits = load_splits("data/metadata/ham10000_splits.json")

# Create dataset
train_dataset = HAM10000Dataset(
    root_dir="data/raw/ham10000",
    metadata_path="data/metadata/ham10000_fst_estimated.csv",
    split="train",
    split_indices=splits['train'],
    use_fst_annotations=True,
)

print(f"Train samples: {len(train_dataset)}")  # ~7,010 images

# Access sample
sample = train_dataset[0]
# Returns:
#   - image: Tensor (3, H, W)
#   - label: int (0-6 diagnosis class)
#   - fst: int (1-6 Fitzpatrick type)
#   - lesion_id, image_id, age, sex, localization
```

---

## Expected Performance

### Diagnosis Classes (7 total)

| Class | Samples | % | Expected AUROC |
|-------|---------|---|----------------|
| nv (nevus) | 6705 | 67.0% | 0.90-0.95 |
| mel (melanoma) | 1113 | 11.1% | 0.85-0.90 |
| bkl (keratosis) | 1099 | 11.0% | 0.80-0.88 |
| bcc (carcinoma) | 514 | 5.1% | 0.85-0.92 |
| akiec (keratoses) | 327 | 3.3% | 0.78-0.85 |
| vasc (vascular) | 142 | 1.4% | 0.75-0.82 |
| df (dermatofibroma) | 115 | 1.1% | 0.72-0.80 |

### Fairness Gaps (Baseline ResNet50)

**Melanoma Detection AUROC by FST**:
- FST I-II (light): **0.88-0.92** (baseline)
- FST III-IV (medium): **0.85-0.89** (-3% gap)
- FST V-VI (dark): **0.80-0.86** (-6% gap)

**Phase 2 Goal**: Reduce gap to < 3% through synthetic data and debiasing.

---

## Manual Steps (if not using setup script)

### Generate FST Annotations

```bash
python scripts/generate_ham10000_fst.py \
    --data-dir data/raw/ham10000 \
    --output data/metadata/ham10000_fst_estimated.csv
```

### Create Splits

```bash
python scripts/create_ham10000_splits.py \
    --metadata data/metadata/ham10000_fst_estimated.csv \
    --output data/metadata/ham10000_splits.json \
    --visualize
```

### Verify Dataset

```bash
python scripts/verify_ham10000.py \
    --data-dir data/raw/ham10000 \
    --metadata data/metadata/ham10000_fst_estimated.csv \
    --splits data/metadata/ham10000_splits.json
```

---

## Troubleshooting

### "HAM10000 data directory not found"

**Solution**: Download dataset from Harvard Dataverse and extract to `data/raw/ham10000/`

### "Image not found for image_id: ISIC_XXXXXXX"

**Solution**: Verify both `HAM10000_images_part_1` and `part_2` directories exist with images

### "Severe class imbalance detected"

**Expected**: HAM10000 has 67% nevi, 1.1% dermatofibroma (natural imbalance)

**Solution**: Use weighted sampling or class-balanced loss during training

---

## Next Steps

### Phase 1.5 Complete ✓
- [x] HAM10000 dataset integration
- [x] FST annotation (ITA-based)
- [x] Stratified splitting
- [x] Baseline training ready

### Phase 2: Fairness Enhancement
- [ ] Synthetic data generation for FST balancing
- [ ] Group-balanced mini-batch sampling
- [ ] FST-aware loss functions
- [ ] Advanced architectures (ViT, Swin)

---

## Important Notes

### FST Estimation Accuracy

FST labels are **estimated using ITA**, not ground truth clinical assessments:
- **Accuracy**: ~70-80% agreement with expert annotations
- **Purpose**: Research on fairness gaps, NOT clinical FST assessment
- **Limitation**: Document this in publications/reports

To use external FST annotations (if available):

```python
dataset = HAM10000Dataset(
    fst_csv_path="data/annotations/expert_fst.csv",  # CSV with image_id, fst
    estimate_fst_if_missing=True,  # Fill gaps with ITA
)
```

### Class Imbalance

HAM10000 has severe imbalance (58:1 ratio). Consider:
- Weighted random sampling
- Focal loss or class-balanced loss
- Synthetic oversampling for minority classes (Phase 2)

---

## Documentation

- **Full Integration Guide**: `docs/ham10000_integration.md`
- **FST Annotation Details**: `src/data/fst_annotation.py`
- **Dataset API Reference**: `src/data/ham10000_dataset.py`
- **Training Configuration**: `configs/baseline_config.yaml`

---

## Quick Commands

```bash
# Setup everything
python scripts/setup_ham10000.py

# Train baseline
python experiments/baseline/train_resnet50.py --config configs/baseline_config.yaml

# Verify dataset
python scripts/verify_ham10000.py

# Test dataset loading
python -m src.data.ham10000_dataset
```

---

**Framework**: MENDICANT_BIAS
**Phase**: 1.5 Complete
**Agent**: HOLLOWED_EYES
**Status**: Ready for Baseline Training

Real experiments begin now.
