"""
Model compression module for MENDICANT_BIAS framework.

Components:
- FairPrune: Fairness-aware pruning
- Quantization: INT8/FP16 conversion
- ONNX export: Production deployment

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
"""

from .fairprune import FairnessPruner, PruningConfig, FairnessEvaluator
from .pruning_trainer import PruningTrainer
from .quantization import ModelQuantizer, QuantizationConfig, quantize_model_pipeline
from .onnx_export import ONNXExporter, ONNXExportConfig

__all__ = [
    'FairnessPruner',
    'PruningConfig',
    'FairnessEvaluator',
    'PruningTrainer',
    'ModelQuantizer',
    'QuantizationConfig',
    'quantize_model_pipeline',
    'ONNXExporter',
    'ONNXExportConfig',
]
