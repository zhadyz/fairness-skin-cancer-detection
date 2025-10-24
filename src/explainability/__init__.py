"""
Explainability Module for Fairness-Aware Skin Cancer Detection

Provides SHAP-based explanations for model predictions with fairness analysis.

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

from .shap_explainer import SHAPExplainer, ExplanationResult
from .visualization import ExplanationVisualizer, create_saliency_overlay

__all__ = [
    "SHAPExplainer",
    "ExplanationResult",
    "ExplanationVisualizer",
    "create_saliency_overlay"
]
