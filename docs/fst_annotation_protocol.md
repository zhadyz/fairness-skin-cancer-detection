# Fitzpatrick Skin Type (FST) Annotation Protocol

**Version**: 1.0
**Date**: 2025-10-13
**Framework**: MENDICANT_BIAS - the_didact research division
**Purpose**: Standardized protocol for annotating skin tone in dermatology images to ensure fairness-aware model development

---

## Executive Summary

This protocol establishes a rigorous, reproducible methodology for annotating Fitzpatrick Skin Type (FST) and Monk Skin Tone (MST) in dermatological images. Accurate skin tone annotation is CRITICAL for:
- Training fairness-aware AI models
- Stratified evaluation (performance metrics per FST group)
- Identifying and mitigating algorithmic bias

**Key Principle**: Skin tone annotation is inherently subjective. This protocol combines **automated algorithms (ITA), expert annotation, and crowd consensus** to maximize reliability.

---

## 1. Annotation Scales

### 1.1 Fitzpatrick Skin Type (FST) - 6-Point Scale

**Original Purpose**: Classify skin's response to UV exposure
**Clinical Definition**:

| FST | Description | UV Response | Prevalence (Global) |
|-----|-------------|-------------|---------------------|
| I | Very fair skin, pale white | Always burns, never tans | ~2% |
| II | Fair skin, white | Usually burns, tans minimally | ~12% |
| III | Medium skin, white to light brown | Sometimes burns, tans uniformly | ~28% |
| IV | Olive skin, moderate brown | Rarely burns, tans easily | ~45% |
| V | Brown skin, dark brown | Very rarely burns, tans very easily | ~11% |
| VI | Very dark skin, deeply pigmented | Never burns | ~2% |

**Limitations**:
- Originally designed for UV sensitivity, not visual appearance
- Coarse granularity (6 categories)
- Poor inter-rater reliability for intermediate types (FST III-IV)
- Racial bias potential (historically correlated with race)

### 1.2 Monk Skin Tone (MST) - 10-Point Scale

**Modern Alternative**: Perceptual skin tone classification (Google AI, 2022)
**Advantages over Fitzpatrick**:
- Finer granularity (10 categories vs 6)
- Based on visual appearance, not UV response
- Better representation of intermediate tones
- Higher inter-rater agreement in recent studies

**MST Scale**:
- Tones 1-2: Very light (corresponds to FST I-II)
- Tones 3-4: Light (FST II-III)
- Tones 5-6: Medium (FST III-IV)
- Tones 7-8: Dark (FST IV-V)
- Tones 9-10: Very dark (FST V-VI)

**Recommendation**: Use MST for NEW annotations, maintain FST for backward compatibility with existing datasets.

---

## 2. Automated Annotation: Individual Typology Angle (ITA)

### 2.1 ITA Algorithm Overview

**ITA Formula**:
```
ITA = [arctan((L* - 50) / b*)] × (180 / π)
```

Where:
- L* = Lightness (0-100 in CIELAB color space)
- b* = Blue-yellow axis (-128 to +127 in CIELAB color space)

**ITA to FST Mapping** (Established in Literature):

| ITA Range | Fitzpatrick Type | MST Equivalent |
|-----------|------------------|----------------|
| > 55° | FST I (Very light) | MST 1-2 |
| 41° - 55° | FST II (Light) | MST 2-3 |
| 28° - 41° | FST III (Intermediate) | MST 3-5 |
| 19° - 28° | FST IV (Tan) | MST 5-7 |
| -30° - 19° | FST V (Brown) | MST 7-9 |
| < -30° | FST VI (Dark brown/Black) | MST 9-10 |

### 2.2 ITA Implementation Steps

**Preprocessing Requirements**:
1. Image standardization: Remove hair, shadows, artifacts
2. Region of Interest (ROI): Select representative skin patch (avoid lesion, hair, shadows)
3. Color space conversion: RGB → CIELAB

**Automated ITA Calculation** (Python pseudocode):
```python
import cv2
import numpy as np
from skimage import color

def calculate_ita(image_path, roi=None):
    """
    Calculate Individual Typology Angle (ITA) for skin tone classification.

    Args:
        image_path: Path to dermatology image
        roi: Region of Interest coordinates (x, y, w, h). If None, use entire image.

    Returns:
        ita_value: ITA angle in degrees
        fst: Fitzpatrick Skin Type (I-VI)
        mst: Monk Skin Tone (1-10)
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract ROI (if specified)
    if roi:
        x, y, w, h = roi
        img_rgb = img_rgb[y:y+h, x:x+w]

    # Convert to CIELAB color space
    img_lab = color.rgb2lab(img_rgb)

    # Calculate mean L* and b* values
    L_mean = np.mean(img_lab[:, :, 0])
    b_mean = np.mean(img_lab[:, :, 2])

    # Calculate ITA
    ita_value = np.degrees(np.arctan((L_mean - 50) / b_mean))

    # Map ITA to FST
    if ita_value > 55:
        fst = 1
        mst = 1
    elif 41 <= ita_value <= 55:
        fst = 2
        mst = 2
    elif 28 <= ita_value < 41:
        fst = 3
        mst = 4
    elif 19 <= ita_value < 28:
        fst = 4
        mst = 6
    elif -30 <= ita_value < 19:
        fst = 5
        mst = 8
    else:  # ita_value < -30
        fst = 6
        mst = 10

    return ita_value, fst, mst
```

**Limitations of ITA**:
- Sensitive to illumination conditions
- Requires clean skin region (no lesions, hair, shadows)
- May misclassify erythema (redness) as lighter skin
- Less reliable for intermediate tones (FST III-IV)

**Mitigation**: Use ITA as **initial estimate**, validate with human annotation.

---

## 3. Human Annotation Protocol

### 3.1 Annotation Platform: LabelBox

**Why LabelBox**:
- Used in Fitzpatrick17k dataset creation (validated in literature)
- Supports custom classification taxonomies
- Facilitates multi-annotator consensus workflows
- Tracks inter-rater reliability metrics

**Alternative Platforms**:
- CVAT (Computer Vision Annotation Tool): Open-source, self-hosted
- Label Studio: Open-source, ML-assisted labeling
- Custom annotation tool: Python + Flask + React (if specific needs arise)

### 3.2 Annotator Selection

**Expertise Levels**:
1. **Expert Annotators** (Gold Standard):
   - Board-certified dermatologists
   - Minimum 2 years clinical experience
   - Trained in Fitzpatrick/Monk scales
   - Target: 2-3 experts for validation set

2. **Trained Layperson Annotators**:
   - Completed FST/MST training module (2-hour online course)
   - Passed calibration test (Kappa > 0.7 with expert consensus)
   - Target: 3-5 annotators per image for crowd consensus

**Training Module Content**:
- Fitzpatrick Scale history and clinical context
- Monk Skin Tone Scale visual guide
- Common annotation pitfalls (erythema, shadows, lighting)
- Practice annotation with expert-validated examples

### 3.3 Annotation Workflow

**Step 1: Automated Pre-Annotation**
- Run ITA algorithm on all images
- Generate initial FST/MST estimates
- Flag images with ambiguous ITA values (e.g., FST III-IV boundary)

**Step 2: Multi-Annotator Review**
- Each image annotated by 3 independent annotators
- Annotators see image + ITA estimate (as reference, not ground truth)
- Annotators classify using both FST (6-point) and MST (10-point)

**Step 3: Consensus Resolution**
- **High Agreement (3/3 or 2/3 same label)**: Accept majority label
- **Disagreement (all different)**: Route to expert dermatologist for adjudication
- **Borderline Cases**: Accept range (e.g., "FST III-IV") for stratified evaluation

**Step 4: Quality Control**
- Random 10% sample re-annotated by expert dermatologist
- Calculate Cohen's Kappa (inter-rater reliability)
- Target: Kappa > 0.7 (substantial agreement)

---

## 4. Inter-Rater Reliability Metrics

### 4.1 Cohen's Kappa

**Formula**:
```
κ = (P_observed - P_expected) / (1 - P_expected)
```

**Interpretation**:
- κ < 0.00: No agreement
- κ = 0.00-0.20: Slight agreement
- κ = 0.21-0.40: Fair agreement
- κ = 0.41-0.60: Moderate agreement
- κ = 0.61-0.80: Substantial agreement
- κ = 0.81-1.00: Almost perfect agreement

**Benchmark**: Fitzpatrick17k achieved κ = 0.70-0.75 between dermatologists.

### 4.2 Confusion Matrix Analysis

Track common misclassifications:
- FST III vs IV (most frequent disagreement)
- FST II vs III (fair vs intermediate skin)
- Erythema (redness) leading to lighter FST classification

---

## 5. Annotation Guidelines & Best Practices

### 5.1 Visual Assessment Rules

**What to Annotate**:
- Classify the **patient's skin tone**, NOT the lesion itself
- Use **perilesional skin** (adjacent healthy skin) as primary reference
- Consider **anatomical site**: Arms, legs may be tanner than torso

**What to Avoid**:
- Do NOT classify based on lesion color (melanomas can be dark on light skin)
- Do NOT use hair color, eye color as proxy for skin tone
- Do NOT let demographic assumptions (name, location) bias annotation

### 5.2 Handling Edge Cases

**Case 1: Erythema (Redness)**
- Problem: Inflammation can artificially lighten ITA estimate
- Solution: Focus on non-inflamed perilesional skin, adjust FST upward if erythema present

**Case 2: Tanning / Sun Damage**
- Problem: Anatomical site (e.g., face, arms) may be darker than constitutional skin type
- Solution: Annotate "effective skin tone" (observed), not constitutional. Note in metadata if sun-exposed site.

**Case 3: Vitiligo / Pigmentation Disorders**
- Problem: Skin tone is non-uniform
- Solution: Annotate predominant skin tone, flag as "pigmentation disorder" in metadata

**Case 4: Poor Lighting / Shadows**
- Problem: Shadows artificially darken skin, overexposure lightens
- Solution: Mark as "ambiguous" if lighting precludes reliable annotation. Request image recapture if possible.

### 5.3 Monk Skin Tone Reference Chart

**Visual Guide**: Use official MST reference card (10 color swatches)
- Print on calibrated color printer (sRGB, D65 illuminant)
- View under standardized lighting (D65 daylight simulator, 6500K)
- Compare perilesional skin to nearest MST swatch

**Digital Alternative**: Use MST digital palette (hex codes available on Google AI website)

---

## 6. FST Stratification for Model Evaluation

### 6.1 Binary Grouping (Simplified Evaluation)

**Light Skin** (FST I-III): Majority group in existing datasets
**Dark Skin** (FST IV-VI): Historically underrepresented group

**Use Case**: Initial fairness gap quantification (AUROC light vs dark)

### 6.2 Granular Grouping (Comprehensive Evaluation)

**Three Groups**:
- FST I-II: Very light to light
- FST III-IV: Intermediate to tan
- FST V-VI: Brown to very dark

**Use Case**: Detailed subgroup analysis, identify specific underperforming tones

### 6.3 Monk Skin Tone Grouping (Finest Resolution)

**Five Groups**:
- MST 1-2: Very light
- MST 3-4: Light
- MST 5-6: Medium
- MST 7-8: Dark
- MST 9-10: Very dark

**Use Case**: Research publications, maximum fairness transparency

---

## 7. Dataset-Specific Annotation Plans

### 7.1 Datasets with Existing FST Labels

**Fitzpatrick17k, DDI, MIDAS, SCIN**:
- **Action**: Use existing labels (validated by dermatologists)
- **Quality Check**: Spot-check 10% with ITA algorithm, flag major discrepancies
- **No re-annotation required** (trust expert curation)

### 7.2 Datasets WITHOUT FST Labels

**HAM10000, ISIC 2019**:
- **Phase 1 (Immediate)**: Run automated ITA on all images
- **Phase 2 (Week 2-3)**: Human annotation for validation set (1,000 images)
- **Phase 3 (Week 4)**: Train FST classifier (ResNet18) on validated set, pseudo-label remaining images
- **Validation**: Expert review of 100 random pseudo-labeled images

**Pseudo-Labeling Strategy**:
1. Manually annotate 5,000 diverse images (covering all FST types)
2. Train FST classifier: ResNet18, 6-class (FST I-VI)
3. Predict FST for remaining 60,000+ images
4. Use predicted labels for stratified splits, report as "estimated FST" in publications

---

## 8. Implementation Timeline

**Week 1**:
- Set up LabelBox project (or alternative platform)
- Create annotator training module
- Run ITA algorithm on HAM10000 + ISIC 2019

**Week 2**:
- Recruit and train annotators (3-5 laypersons)
- Begin multi-annotator review (target: 500 images/week)
- Establish expert adjudication workflow

**Week 3**:
- Complete 1,500+ annotations
- Calculate inter-rater reliability (Cohen's Kappa)
- Identify and document common disagreement patterns

**Week 4**:
- Train FST classifier for pseudo-labeling
- Validate pseudo-labels (expert review)
- Finalize stratified train/val/test splits

---

## 9. Software Tools & Libraries

### 9.1 ITA Calculation
```python
# Dependencies
pip install opencv-python scikit-image numpy

# Implementation: src/data/fst_annotation.py
```

### 9.2 Annotation Platforms
- **LabelBox**: https://labelbox.com (free tier available)
- **CVAT**: https://github.com/opencv/cvat (self-hosted)
- **Label Studio**: https://labelstud.io (open-source)

### 9.3 Inter-Rater Reliability
```python
# Cohen's Kappa calculation
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
```

---

## 10. Ethical Considerations

### 10.1 Privacy & Consent

- All images must be de-identified (no patient names, dates, metadata)
- Use only datasets with proper informed consent
- FST labels should NOT enable re-identification

### 10.2 Avoiding Racial Essentialism

**CRITICAL PRINCIPLE**: FST/MST are **visual appearance classifications**, NOT proxies for race/ethnicity.

- Do NOT conflate skin tone with race (e.g., "FST VI = Black race" is FALSE)
- Do NOT use FST to infer other demographic attributes
- Transparent disclosure: Models use FST for fairness-aware training, NOT for racial profiling

### 10.3 Annotator Bias Mitigation

- Train annotators to recognize implicit bias (e.g., assuming lighter skin for certain names)
- Blind annotators to patient demographics during labeling
- Diverse annotator pool (recruit annotators of varied skin tones)

---

## 11. Quality Metrics

**Target Benchmarks**:
- Inter-rater Kappa: > 0.70 (substantial agreement)
- ITA-Human agreement: > 75% (within 1 FST category)
- Expert validation accuracy: > 90% (for pseudo-labeled data)

**Monitoring**:
- Weekly inter-rater reliability reports
- Monthly expert validation audits
- Annotator performance tracking (flag outliers)

---

## 12. References

**Key Literature**:
1. Groh, M., et al. (2021). *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset*. arXiv:2104.09957.
   - **Relevance**: LabelBox annotation, inter-rater reliability analysis

2. Chardon, A., Cretois, I., & Hourseau, C. (1991). *Skin colour typology and suntanning pathways*. International Journal of Cosmetic Science, 13(4), 191-208.
   - **Relevance**: Original ITA formula and FST mapping

3. Monk, E. (2022). *The Monk Skin Tone Scale*. Google AI.
   - **Relevance**: MST 10-point scale, inter-rater agreement improvements

4. Jain, A., et al. (2024). *Crowdsourcing Dermatology Images with Google Search Ads: Creating a Real-World Skin Condition Dataset*. arXiv:2402.18545.
   - **Relevance**: SCIN dataset, eFST and eMST annotation methodology

5. Del Bino, S., et al. (2006). *Clinical and biological characterization of skin pigmentation diversity and its consequences on UV impact*. International Journal of Dermatology, 45(3), 314-319.
   - **Relevance**: Limitations of Fitzpatrick scale, need for objective measurements

**Automated Annotation Tools**:
- Scarletred Vision: https://www.scarletred.com/scarletredderm (ITA commercial platform)
- OpenCV + scikit-image: Open-source ITA implementation

---

## 13. Contact & Support

**Protocol Questions**: the_didact@mendicant-bias.ai
**Annotation Platform Issues**: [Platform-specific support]
**Expert Adjudication Requests**: [Designated dermatologist email]

---

**Version Control**:
- v1.0 (2025-10-13): Initial protocol established
- Next Review: 2025-10-20 (post-pilot annotation batch)

**Maintained by**: the_didact (MENDICANT_BIAS framework)
**Approved by**: mendicant_bias (Supreme Orchestrator)
