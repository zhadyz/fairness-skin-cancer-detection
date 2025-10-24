"""
Pydantic Models for API Request/Response Validation

Defines data models for FastAPI endpoints with comprehensive validation.

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class DiagnosisClass(str, Enum):
    """Diagnosis classes for skin lesions."""
    MEL = "melanoma"  # Malignant
    NV = "melanocytic_nevus"  # Benign
    BCC = "basal_cell_carcinoma"  # Malignant
    AK = "actinic_keratosis"  # Pre-malignant
    BKL = "benign_keratosis"  # Benign
    DF = "dermatofibroma"  # Benign
    VASC = "vascular_lesion"  # Benign


class FitzpatrickSkinType(int, Enum):
    """Fitzpatrick skin type classification (I-VI)."""
    FST_I = 1
    FST_II = 2
    FST_III = 3
    FST_IV = 4
    FST_V = 5
    FST_VI = 6


class PredictionRequest(BaseModel):
    """Request model for single image prediction."""
    return_explanation: bool = Field(
        default=False,
        description="Whether to return SHAP explanation"
    )
    estimate_fst: bool = Field(
        default=True,
        description="Whether to estimate Fitzpatrick skin type"
    )
    explanation_method: str = Field(
        default="saliency",
        description="Explanation method (gradient_shap, integrated_gradients, saliency)"
    )

    @validator('explanation_method')
    def validate_explanation_method(cls, v):
        valid_methods = ['gradient_shap', 'integrated_gradients', 'saliency']
        if v not in valid_methods:
            raise ValueError(f"explanation_method must be one of {valid_methods}")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    return_explanations: bool = Field(
        default=False,
        description="Whether to return SHAP explanations for all images"
    )
    estimate_fst: bool = Field(
        default=True,
        description="Whether to estimate Fitzpatrick skin type"
    )


class PredictionResponse(BaseModel):
    """Response model for single image prediction."""
    diagnosis: str = Field(
        ...,
        description="Predicted diagnosis class"
    )
    confidence: float = Field(
        ...,
        description="Prediction confidence (0-1)",
        ge=0.0,
        le=1.0
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities for all classes"
    )
    fst_estimate: Optional[int] = Field(
        None,
        description="Estimated Fitzpatrick skin type (1-6)"
    )
    explanation: Optional[Dict[str, Any]] = Field(
        None,
        description="SHAP explanation data (if requested)"
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    model_version: str = Field(
        ...,
        description="Model version used for prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each image"
    )
    total_inference_time_ms: float = Field(
        ...,
        description="Total batch inference time in milliseconds"
    )
    batch_size: int = Field(
        ...,
        description="Number of images in batch"
    )


class ExplanationData(BaseModel):
    """Explanation data structure."""
    attribution_magnitude: float = Field(
        ...,
        description="Mean attribution magnitude"
    )
    concentration_score: float = Field(
        ...,
        description="Attribution concentration score (0-1)"
    )
    top_regions: List[Dict[str, float]] = Field(
        ...,
        description="Top important regions [(x, y, importance), ...]"
    )
    saliency_map_base64: Optional[str] = Field(
        None,
        description="Base64-encoded saliency overlay image"
    )
    explanation_method: str = Field(
        ...,
        description="Method used for explanation"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(
        ...,
        description="Service status (healthy, unhealthy)"
    )
    model_version: str = Field(
        ...,
        description="Loaded model version"
    )
    model_size_mb: float = Field(
        ...,
        description="Model size in megabytes"
    )
    compression_type: str = Field(
        ...,
        description="Model compression type (fp32, fp16, int8, onnx)"
    )
    avg_inference_time_ms: Optional[float] = Field(
        None,
        description="Average inference time (if available)"
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime in seconds"
    )
    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(
        ...,
        description="Error message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    error_code: str = Field(
        ...,
        description="Error code for client handling"
    )


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""
    total_requests: int = Field(
        ...,
        description="Total number of requests processed"
    )
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made"
    )
    avg_inference_time_ms: float = Field(
        ...,
        description="Average inference time"
    )
    p95_inference_time_ms: float = Field(
        ...,
        description="95th percentile inference time"
    )
    p99_inference_time_ms: float = Field(
        ...,
        description="99th percentile inference time"
    )
    error_rate: float = Field(
        ...,
        description="Error rate (0-1)"
    )
    uptime_seconds: float = Field(
        ...,
        description="Service uptime"
    )


# Error codes
class ErrorCode(str, Enum):
    """Standard error codes."""
    INVALID_IMAGE = "INVALID_IMAGE"
    MODEL_ERROR = "MODEL_ERROR"
    EXPLANATION_ERROR = "EXPLANATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
