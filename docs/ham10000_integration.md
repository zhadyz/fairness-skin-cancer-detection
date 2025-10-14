# HAM10000 Dataset Integration Guide

**Framework**: MENDICANT_BIAS - Phase 1.5
**Agent**: HOLLOWED_EYES
**Version**: 1.0
**Date**: 2025-10-13

---

## Overview

Complete implementation of HAM10000 (Human Against Machine with 10,000 training images) dataset integration with Fitzpatrick Skin Type (FST) annotation support for baseline fairness experiments.

### What's Included

1. **HAM10000Dataset** - PyTorch Dataset class with FST support
2. **FST Estimation** - ITA-based automatic FST annotation
3. **Stratified Splitting** - Train/val/test splits preserving diagnosis and FST distribution
4. **Dataset Verification** - Comprehensive integrity checking
5. **Training Integration** - Updated baseline training pipeline

---

## Dataset Overview

### HAM10000 Statistics

- **Total Images**: 10,015 dermoscopic images
- **Diagnostic Categories**: 7 classes
- **Image Format**: JPEG
- **Resolution**: Variable (typically 600x450 to 6000x4000 pixels)

### Diagnosis Classes

| Code | Diagnosis | Full Name |
|------|-----------|-----------|
| `akiec` | Actinic Keratoses | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal Cell Carcinoma | Basal cell carcinoma |
| `bkl` | Benign Keratosis | Benign keratosis-like lesions |
| `df` | Dermatofibroma | Dermatofibroma |
| `mel` | Melanoma | Melanoma |
| `nv` | Melanocytic Nevi | Melanocytic nevi |
| `vasc` | Vascular Lesions | Vascular lesions |

### Class Distribution (Original)

```
nv (nevus):           6705 (67.0%)
mel (melanoma):       1113 (11.1%)
bkl (keratosis):       1099 (11.0%)
bcc (carcinoma):        514 (5.1%)
akiec (keratoses):      327 (3.3%)
vasc (vascular):        142 (1.4%)
df (dermatofibroma):    115 (1.1%)
```

**Note**: Severe class imbalance (58:1 ratio between largest and smallest classes).

---

## Installation and Setup

### 1. Download HAM10000 Dataset

**Source**: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

**Required Files**:
- `HAM10000_images_part_1.zip` (~2.5 GB)
- `HAM10000_images_part_2.zip` (~2.5 GB)
- `HAM10000_metadata` (CSV file)

**Installation Steps**:

```bash
# Create data directory
mkdir -p data/raw/ham10000

# Download files (manual or using wget/curl)
# Extract to correct location:
unzip HAM10000_images_part_1.zip -d data/raw/ham10000/
unzip HAM10000_images_part_2.zip -d data/raw/ham10000/
cp HAM10000_metadata.csv data/raw/ham10000/

# Verify directory structure
ls data/raw/ham10000/
# Expected:
#   HAM10000_images_part_1/
#   HAM10000_images_part_2/
#   HAM10000_metadata.csv
```

### 2. Generate FST Annotations

HAM10000 does **not** include native Fitzpatrick Skin Type (FST) labels. We estimate FST using Individual Typology Angle (ITA) calculation.

```bash
# Generate FST annotations using ITA
python scripts/generate_ham10000_fst.py \
    --data-dir data/raw/ham10000 \
    --output data/metadata/ham10000_fst_estimated.csv

# Options:
#   --no-exclude-lesion     Do NOT exclude lesion from ITA calculation
#   --no-visualizations     Skip distribution plots
```

**Output**:
- `data/metadata/ham10000_fst_estimated.csv` - Metadata with FST labels
- `data/metadata/visualizations/` - FST distribution plots

**ITA-FST Mapping** (Chardon et al. 1991):

| ITA Range | FST | Description |
|-----------|-----|-------------|
| > 55° | I | Very light |
| 41-55° | II | Light |
| 28-41° | III | Intermediate |
| 19-28° | IV | Tan/Olive |
| -30-19° | V | Brown |
| < -30° | VI | Dark brown/Black |

### 3. Create Train/Val/Test Splits

```bash
# Create stratified splits
python scripts/create_ham10000_splits.py \
    --metadata data/metadata/ham10000_fst_estimated.csv \
    --output data/metadata/ham10000_splits.json \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --random-seed 42 \
    --visualize

# Options:
#   --no-stratify-fst    Stratify by diagnosis only (not FST)
#   --visualize          Generate split distribution plots
```

**Features**:
- Stratified by diagnosis AND FST (proportional representation)
- Lesion-level splitting (no data leakage - same lesion stays in one split)
- Reproducible (fixed random seed)

**Output**:
- `data/metadata/ham10000_splits.json` - Train/val/test indices
- `data/metadata/split_visualizations/` - Distribution comparison plots

### 4. Verify Dataset Integrity

```bash
# Run comprehensive verification
python scripts/verify_ham10000.py \
    --data-dir data/raw/ham10000 \
    --metadata data/metadata/ham10000_fst_estimated.csv \
    --splits data/metadata/ham10000_splits.json \
    --sample-size 100

# Verifies:
#   [1/8] Directory structure
#   [2/8] Metadata completeness
#   [3/8] Image loading integrity
#   [4/8] Diagnosis distribution
#   [5/8] FST distribution
#   [6/8] Split integrity (no data leakage)
#   [7/8] Image statistics (mean, std)
```

**Expected Output**:
```
========================================================================
VERIFICATION SUMMARY
========================================================================

All checks PASSED
Dataset is ready for training
========================================================================
```

---

## Usage

### Basic Dataset Loading

```python
from src.data.ham10000_dataset import HAM10000Dataset, load_splits
from src.data.preprocessing import get_training_augmentation

# Load splits
splits = load_splits("data/metadata/ham10000_splits.json")

# Create dataset
train_dataset = HAM10000Dataset(
    root_dir="data/raw/ham10000",
    metadata_path="data/metadata/ham10000_fst_estimated.csv",
    split="train",
    split_indices=splits['train'],
    transform=get_training_augmentation(image_size=224),
    use_fst_annotations=True,
    estimate_fst_if_missing=True,
)

print(f"Dataset size: {len(train_dataset)}")
print(f"Classes: {train_dataset.get_class_distribution()}")
print(f"FST distribution: {train_dataset.get_fst_distribution()}")

# Access sample
sample = train_dataset[0]
print(f"Image: {sample['image'].shape}")
print(f"Label: {sample['label']} ({train_dataset.get_label_name(sample['label'])})")
print(f"FST: {sample['fst']}")
print(f"Age: {sample['age']}")
print(f"Sex: {sample['sex']}")
print(f"Localization: {sample['localization']}")
```

### Training with DataLoader

```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    """Custom collate function for HAM10000Dataset."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    fst_labels = torch.tensor([item['fst'] for item in batch])
    return images, labels, fst_labels

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)

# Training loop
for images, labels, fst_labels in train_loader:
    # images: (B, 3, H, W)
    # labels: (B,) - diagnosis class
    # fst_labels: (B,) - FST (1-6 or -1 if unknown)
    pass
```

### Running Baseline Training

```bash
# Train ResNet50 baseline
python experiments/baseline/train_resnet50.py \
    --config configs/baseline_config.yaml \
    --device cuda

# Options:
#   --test-only           Skip training, only evaluate
#   --checkpoint PATH     Load checkpoint for evaluation
```

**Training automatically**:
1. Loads HAM10000 dataset with FST annotations
2. Applies data augmentation
3. Trains ResNet50 with specified config
4. Evaluates fairness metrics (AUROC, TPR, FPR per FST)
5. Generates visualization plots
6. Saves checkpoints and results

---

## FST Annotation Details

### Method: ITA-based Estimation

**Individual Typology Angle (ITA)** calculation from CIELAB color space:

```
ITA = arctan((L* - 50) / b*) × (180 / π)
```

Where:
- **L*** (lightness): 0 (black) to 100 (white)
- **b*** (blue-yellow axis): negative (blue) to positive (yellow)

**Implementation** (`src/data/fst_annotation.py`):

1. Load dermoscopic image (RGB)
2. Exclude lesion area (remove darkest 20% pixels)
3. Convert to CIELAB color space
4. Calculate mean L* and b*
5. Compute ITA angle
6. Map ITA to FST (1-6)

### Limitations and Considerations

**ITA-based FST is an ESTIMATE, not ground truth**:

- **Accuracy**: ~70-80% agreement with expert annotations (varies by study)
- **Lesion Exclusion**: Simple percentile-based method (not perfect)
- **Lighting Variability**: Dermoscopic images have controlled lighting (better than clinical)
- **Use Case**: Suitable for research on fairness gaps, NOT for clinical FST assessment

**Recommended**:
- Document that FST is estimated (not clinically validated)
- Report inter-rater reliability if expert annotations available
- Consider using external FST annotations (e.g., Fitzpatrick17k overlaps) if possible

### Using External FST Annotations

If you have external FST annotations (CSV with `image_id` and `fst` columns):

```python
train_dataset = HAM10000Dataset(
    root_dir="data/raw/ham10000",
    metadata_path="data/metadata/ham10000_fst_estimated.csv",
    fst_csv_path="data/annotations/expert_fst_annotations.csv",  # External annotations
    estimate_fst_if_missing=True,  # Fill missing with ITA
)
```

---

## Directory Structure

After setup, your directory should look like:

```
data/
├── raw/
│   └── ham10000/
│       ├── HAM10000_images_part_1/
│       │   ├── ISIC_0024306.jpg
│       │   ├── ISIC_0024307.jpg
│       │   └── ... (5000 images)
│       ├── HAM10000_images_part_2/
│       │   ├── ISIC_0024308.jpg
│       │   └── ... (5015 images)
│       └── HAM10000_metadata.csv
├── metadata/
│   ├── ham10000_fst_estimated.csv
│   ├── ham10000_splits.json
│   ├── visualizations/
│   │   ├── ham10000_fst_distribution.png
│   │   ├── ham10000_ita_distribution.png
│   │   └── ham10000_fst_by_diagnosis.png
│   └── split_visualizations/
│       ├── diagnosis_distribution_by_split.png
│       ├── fst_distribution_by_split.png
│       └── diagnosis_comparison.png
└── annotations/
    └── (external annotations if available)
```

---

## API Reference

### HAM10000Dataset

```python
class HAM10000Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        metadata_path: Optional[str] = None,
        split: str = "train",
        split_indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        use_fst_annotations: bool = True,
        fst_csv_path: Optional[str] = None,
        estimate_fst_if_missing: bool = True,
        image_parts: List[str] = ["HAM10000_images_part_1", "HAM10000_images_part_2"],
    )
```

**Returns** (dict):
- `image`: Tensor (C, H, W) - transformed image
- `label`: int - diagnosis class (0-6)
- `fst`: int - FST (1-6) or -1 if unknown
- `lesion_id`: str - lesion identifier
- `image_id`: str - image identifier
- `age`: float - patient age (or NaN)
- `sex`: str - patient sex
- `localization`: str - anatomical location

**Methods**:
- `get_class_distribution()` → Dict[str, int]
- `get_fst_distribution()` → Dict[int, int]
- `get_label_name(label: int)` → str
- `get_metadata_df()` → pd.DataFrame

### create_fst_stratified_splits()

```python
def create_fst_stratified_splits(
    metadata_path: str,
    output_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_by_fst: bool = True,
) -> Dict[str, List[int]]
```

Creates train/val/test splits stratified by diagnosis AND FST.

**Ensures**:
- Balanced diagnosis distribution
- Proportional FST representation
- No lesion-level data leakage
- Reproducible splits (fixed seed)

---

## Expected Fairness Gaps (Baseline)

Based on literature and preliminary experiments, baseline ResNet50 is expected to show:

### Diagnosis Performance (AUROC)

| Class | Expected AUROC | Notes |
|-------|----------------|-------|
| Melanoma (mel) | 0.85-0.90 | High clinical importance |
| Nevus (nv) | 0.90-0.95 | Largest class |
| BCC (bcc) | 0.85-0.92 | Distinct features |
| Keratosis (bkl) | 0.80-0.88 | Moderate difficulty |
| Other classes | 0.75-0.85 | Small sample sizes |

### FST Fairness Gaps

**Melanoma Detection AUROC by FST**:

| FST | Expected AUROC | Gap vs FST I-II |
|-----|----------------|-----------------|
| I-II (light) | 0.88-0.92 | Baseline |
| III-IV (medium) | 0.85-0.89 | -2 to -4% |
| V-VI (dark) | 0.80-0.86 | -5 to -8% |

**Causes**:
1. Class imbalance (nevi >> melanoma)
2. FST distribution bias (more FST I-III in HAM10000)
3. Lesion contrast differences (darker lesions on darker skin)
4. Dataset collection bias (European populations over-represented)

**Phase 2 Goal**: Reduce fairness gaps to < 3% through synthetic data augmentation and debiasing.

---

## Troubleshooting

### Issue: "Image not found for image_id: ISIC_XXXXXXX"

**Cause**: Image file missing from HAM10000_images_part_X directories.

**Solution**:
1. Verify downloads completed successfully
2. Check directory structure matches expected format
3. Re-extract ZIP files if corrupted

### Issue: "Failed to estimate FST for image"

**Cause**: Image loading or ITA calculation error.

**Solution**:
1. Check image is valid JPEG
2. Verify image has 3 color channels (RGB)
3. Review error message for specific issue

### Issue: "Lesion leakage detected between splits"

**Cause**: Same lesion_id appears in multiple splits.

**Solution**:
- This should NOT happen with `create_fst_stratified_splits()`
- If detected, regenerate splits with fresh random seed
- Report as bug if persists

### Issue: "Severe class imbalance detected"

**Cause**: HAM10000 has natural class imbalance (67% nevi, 1.1% dermatofibroma).

**Solution**:
- Use weighted sampling in DataLoader
- Apply focal loss or class-balanced loss
- Generate synthetic samples for minority classes (Phase 2)

---

## Next Steps

### Phase 1.5 Complete ✓

- [x] HAM10000Dataset implementation
- [x] FST annotation (ITA-based)
- [x] Stratified splitting
- [x] Dataset verification
- [x] Baseline training integration

### Phase 2: Fairness-Enhanced Training

1. **Synthetic Data Generation**
   - Generate diverse skin tone synthetic lesions
   - Oversample underrepresented FST groups
   - Style transfer for domain adaptation

2. **Debiasing Techniques**
   - Group-balanced mini-batches
   - FST-aware loss functions
   - Adversarial debiasing

3. **Advanced Architectures**
   - Vision Transformers (ViT, Swin)
   - Multi-task learning (diagnosis + FST prediction)
   - Self-supervised pre-training

---

## References

### HAM10000 Dataset

- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161. [DOI: 10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

### Fitzpatrick Skin Type

- Fitzpatrick, T. B. (1988). The validity and practicality of sun-reactive skin types I through VI. *Archives of Dermatology*, 124(6), 869-871.

### ITA Calculation

- Chardon, A., Cretois, I., & Hourseau, C. (1991). Skin colour typology and suntanning pathways. *International Journal of Cosmetic Science*, 13(4), 191-208.

### Fairness in Medical AI

- Daneshjou, R., et al. (2022). Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances*, 8(32), eabq6147.
- Groh, M., et al. (2021). Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset. *CVPR Workshop*.

---

## Contact and Support

**Framework**: MENDICANT_BIAS
**Agent**: HOLLOWED_EYES (Primary Developer)
**Orchestrator**: MENDICANT_BIAS (Research Coordinator)

For issues, suggestions, or contributions:
- Check `.claude/memory/` for session history
- Review `docs/` for additional documentation
- Consult Phase 2 roadmap in project planning documents

---

**Last Updated**: 2025-10-13
**Status**: Phase 1.5 Complete - Ready for Baseline Training
