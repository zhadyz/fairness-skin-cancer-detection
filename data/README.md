# Data Organization and Management

## Directory Structure

```
data/
├── raw/                      # Original unmodified datasets
│   ├── ham10000/            # HAM10000 dataset
│   ├── isic2019/            # ISIC 2019 Challenge dataset
│   ├── fitzpatrick17k/      # Fitzpatrick17k dataset
│   ├── ddi/                 # Diverse Dermatology Images
│   ├── midas/               # MIDAS skin lesion dataset
│   └── scin/                # SCIN dataset
├── processed/               # Preprocessed and split datasets
│   ├── train/               # Training set
│   ├── val/                 # Validation set
│   └── test/                # Test set
├── synthetic/               # Synthetic data for augmentation
├── annotations/             # Fitzpatrick Skin Type (FST) annotations
└── metadata/                # Dataset metadata and statistics
```

## Dataset Sources and Access

### 1. HAM10000 (Human Against Machine with 10000 training images)

**Source**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

**Description**: 10,015 dermatoscopic images of pigmented skin lesions

**Classes** (7):
- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

**Download Instructions**:
```bash
# Manual download from Harvard Dataverse
# Extract to: data/raw/ham10000/
```

**Expected Structure**:
```
data/raw/ham10000/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── HAM10000_metadata.csv
└── README.txt
```

### 2. ISIC 2019 Challenge Dataset

**Source**: https://challenge.isic-archive.com/data/

**Description**: International Skin Imaging Collaboration challenge dataset

**Classes** (8):
- Melanoma
- Melanocytic nevus
- Basal cell carcinoma
- Actinic keratosis
- Benign keratosis
- Dermatofibroma
- Vascular lesion
- Squamous cell carcinoma

**Download Instructions**:
```bash
# Register at ISIC Archive
# Download via ISIC API or web interface
# Extract to: data/raw/isic2019/
```

**Expected Structure**:
```
data/raw/isic2019/
├── ISIC_2019_Training_Input/
├── ISIC_2019_Training_GroundTruth.csv
└── ISIC_2019_Training_Metadata.csv
```

### 3. Fitzpatrick17k

**Source**: https://github.com/mattgroh/fitzpatrick17k

**Description**: 16,577 clinical images labeled with Fitzpatrick skin types

**Skin Types** (6):
- Type I: Always burns, never tans
- Type II: Burns easily, tans minimally
- Type III: Burns moderately, tans gradually
- Type IV: Burns minimally, tans easily
- Type V: Rarely burns, tans profusely
- Type VI: Never burns, deeply pigmented

**Download Instructions**:
```bash
git clone https://github.com/mattgroh/fitzpatrick17k.git data/raw/fitzpatrick17k
```

**Expected Structure**:
```
data/raw/fitzpatrick17k/
├── images/
├── fitzpatrick17k.csv
└── README.md
```

### 4. Diverse Dermatology Images (DDI)

**Source**: https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965

**Description**: Diverse dermatological conditions across skin tones

**Download Instructions**:
```bash
# Request access from Stanford AIMI
# Follow institutional data sharing agreement
# Extract to: data/raw/ddi/
```

### 5. MIDAS (Melanoma Detection)

**Source**: Check academic collaborations or public repositories

**Download Instructions**:
```bash
# Access through research partnerships
# Extract to: data/raw/midas/
```

### 6. SCIN Dataset

**Source**: Research-specific access

**Download Instructions**:
```bash
# Contact dataset maintainers
# Extract to: data/raw/scin/
```

## Preprocessing Pipeline

### Step 1: Data Validation
```bash
python src/data/validate_datasets.py --data-dir data/raw
```

**Checks**:
- File integrity
- Image dimensions
- Metadata completeness
- Class distribution

### Step 2: FST Annotation

**For datasets without FST labels**:

```bash
python src/data/annotate_skin_type.py \
    --input data/raw/ham10000 \
    --output data/annotations/ham10000_fst.csv \
    --method [manual|model|crowdsource]
```

**Annotation Methods**:
1. **Manual**: Expert dermatologist review
2. **Model-based**: Automated FST classification using pre-trained models
3. **Crowdsourcing**: Multi-annotator consensus

**Guidelines**:
- Follow Fitzpatrick scale definitions
- Use reference images for calibration
- Record annotator confidence scores
- Resolve disagreements through consensus

### Step 3: Stratified Splitting

```bash
python src/data/create_splits.py \
    --data-dir data/raw \
    --output-dir data/processed \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --stratify-by class,skin_type \
    --seed 42
```

**Stratification ensures**:
- Balanced class distribution across splits
- Proportional FST representation
- No data leakage between splits

### Step 4: Preprocessing and Normalization

```bash
python src/data/preprocess.py \
    --input data/processed/train \
    --output data/processed/train_preprocessed \
    --resize 224 \
    --normalize imagenet \
    --remove-artifacts
```

**Preprocessing steps**:
- Resize to model input size (224x224 or 384x384)
- Normalize using ImageNet statistics
- Remove ruler marks, hair artifacts (optional)
- Color constancy correction

## Synthetic Data Generation

### Augmentation Techniques

```bash
python src/data/generate_synthetic.py \
    --input data/processed/train \
    --output data/synthetic \
    --method [cutmix|mixup|styleaug|gan] \
    --target-balance
```

**Methods**:
1. **CutMix**: Cut-and-paste augmentation
2. **MixUp**: Linear interpolation between images
3. **StyleAug**: Style transfer for domain adaptation
4. **GAN-based**: Synthetic image generation (StyleGAN3, Stable Diffusion)

**Use cases**:
- Class balancing (oversample minority classes)
- FST balancing (generate underrepresented skin types)
- Domain adaptation (bridge dataset distribution gaps)

## FST Annotation Guidelines

### Reference Standards

**Use calibrated reference images** from:
- Fitzpatrick TB (1988) original scale
- Willis I, Earles RM (1994) photographic standards
- Clinical dermatology textbooks

### Annotation Protocol

1. **Ambient Lighting**: Consistent, neutral lighting
2. **Multiple Regions**: Assess sun-exposed and protected areas
3. **Patient History**: Consider burning/tanning history when available
4. **Consensus**: Minimum 2 annotators for ambiguous cases

### Quality Control

```bash
python src/fairness/validate_annotations.py \
    --annotations data/annotations/ \
    --inter-rater-reliability
```

**Metrics**:
- Inter-rater agreement (Cohen's kappa > 0.7)
- Confidence score distribution
- Per-class annotation balance

## Dataset Statistics

After preprocessing, generate statistics:

```bash
python src/data/generate_statistics.py \
    --data-dir data/processed \
    --output data/metadata/dataset_stats.json
```

**Includes**:
- Sample counts per class
- FST distribution
- Image dimensions and formats
- Class imbalance ratios
- Train/val/test split details

## Data Loading Example

```python
from src.data.dataloader import SkinLesionDataset

dataset = SkinLesionDataset(
    data_dir='data/processed/train',
    metadata_path='data/metadata/train_metadata.csv',
    transform=augmentation_pipeline,
    skin_type_sensitive=True
)

# Access sample
image, label, skin_type = dataset[0]
```

## Privacy and Ethics

### Data Handling Requirements

1. **No PHI (Protected Health Information)**:
   - Remove patient identifiers
   - Anonymize metadata
   - Secure storage for sensitive datasets

2. **Institutional Review**:
   - IRB approval for human subjects research
   - Informed consent verification
   - Data use agreements

3. **Bias Mitigation**:
   - Document demographic distributions
   - Report performance stratified by FST
   - Address representation gaps

### Data Sharing Policy

- **Public datasets**: HAM10000, ISIC 2019, Fitzpatrick17k (follow original licenses)
- **Restricted datasets**: DDI, MIDAS, SCIN (institutional agreements only)
- **Synthetic data**: Can be shared openly

## Troubleshooting

### Issue: Missing FST labels

**Solution**:
- Use model-based annotation: `src/fairness/fst_classifier.py`
- Request crowdsourced annotations
- Manually annotate critical subsets

### Issue: Class imbalance

**Solution**:
- Apply weighted sampling: `WeightedRandomSampler`
- Use focal loss or class-balanced loss
- Generate synthetic samples for minority classes

### Issue: Dataset heterogeneity

**Solution**:
- Standardize preprocessing across datasets
- Apply domain adaptation techniques
- Use multi-dataset training strategies

### Issue: Large dataset size

**Solution**:
- Use data streaming with `IterableDataset`
- Enable efficient caching with `prefetch`
- Store preprocessed images in HDF5/WebDataset format

## Next Steps

1. Download required datasets (see sources above)
2. Run preprocessing pipeline
3. Generate FST annotations
4. Create stratified splits
5. Validate with `src/data/validate_datasets.py`
6. Proceed to model training: `experiments/baseline/`

## References

- Tschandl, P., et al. (2018). HAM10000 dataset. Nature Scientific Data.
- Combalia, M., et al. (2019). ISIC 2019 Challenge.
- Groh, M., et al. (2021). Fitzpatrick17k. CVPR Workshop.
- Daneshjou, R., et al. (2022). Diverse Dermatology Images (DDI).

---

**Last Updated**: 2025-10-13
**Maintained By**: ZHADYZ DevOps Agent
