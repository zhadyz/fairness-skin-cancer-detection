"""
FairSkin Diffusion-Based Augmentation Module.

Implements diffusion-based data augmentation for FST-balanced synthetic
dermoscopy image generation using Stable Diffusion v1.5 + LoRA.

Components:
- fairskin_diffusion.py: Core diffusion model wrapper
- lora_trainer.py: LoRA fine-tuning on HAM10000
- synthetic_dataset.py: Mixed real+synthetic dataset
- quality_metrics.py: FID, LPIPS, quality validation

Framework: MENDICANT_BIAS - Phase 2
Agent: HOLLOWED_EYES
Version: 0.3.0 (FairSkin)
Date: 2025-10-13
"""

from .fairskin_diffusion import FairSkinDiffusionModel
from .lora_trainer import LoRATrainer
from .synthetic_dataset import SyntheticDermoscopyDataset, MixedDataset
from .quality_metrics import (
    compute_fid,
    compute_lpips,
    compute_diversity_score,
    QualityFilter
)

__all__ = [
    'FairSkinDiffusionModel',
    'LoRATrainer',
    'SyntheticDermoscopyDataset',
    'MixedDataset',
    'compute_fid',
    'compute_lpips',
    'compute_diversity_score',
    'QualityFilter',
]

__version__ = '0.3.0'
