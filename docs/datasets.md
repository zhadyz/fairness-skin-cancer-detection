# Datasets for Fairness-Aware Skin Cancer Detection

## Overview

This document describes the datasets used for training and evaluating the fairness-aware skin cancer detection system.

## Primary Datasets

### 1. HAM10000
- **Description**: Human Against Machine with 10,000 dermoscopic images
- **Size**: 10,015 images
- **Classes**: 7 diagnostic categories
- **Metadata**: Age, sex, localization
- **Limitation**: Limited diversity in skin tones (majority lighter skin)
- **Source**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

### 2. ISIC Archive
- **Description**: International Skin Imaging Collaboration dataset
- **Size**: 50,000+ dermoscopic images
- **Classes**: Multiple diagnostic categories
- **Metadata**: Patient demographics, lesion location
- **Source**: https://www.isic-archive.com/

### 3. DDI (Diverse Dermatology Images)
- **Description**: Focused dataset with representation across skin tones
- **Size**: Variable
- **Fitzpatrick Scale**: Diverse representation across I-VI
- **Importance**: Critical for fairness evaluation
- **Source**: Research publication

## Data Preprocessing

1. **Image Resizing**: 224x224 or 384x384 depending on model
2. **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Augmentation**:
   - Random rotation, flip, crop
   - Color jittering (carefully tuned to preserve diagnostic features)
   - Advanced: Mixup, CutMix

## Fairness Considerations

### Balanced Sampling Strategy
- Oversample underrepresented skin tones
- Class-balanced batches
- Stratified splits by Fitzpatrick scale

### Metadata Requirements
- Fitzpatrick skin type (I-VI)
- Patient demographics
- Image quality indicators

## Data Organization

```
data/
├── raw/
│   ├── ham10000/
│   ├── isic/
│   └── ddi/
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── train_meta.csv
    ├── val_meta.csv
    └── test_meta.csv
```

## Ethical Considerations

- Patient privacy: All datasets must be de-identified
- Informed consent: Only use datasets with proper consent
- Fairness: Ensure diverse representation in training data
- Clinical validation: Test on real-world diverse populations

---

*Documentation maintained by MENDICANT_BIAS framework*
