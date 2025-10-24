"""
FastAPI Production API for Fairness-Aware Skin Cancer Detection

REST API with:
- Single and batch image prediction
- SHAP explainability
- FST estimation
- Health monitoring
- Rate limiting
- Prometheus metrics

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import logging
from typing import List, Optional
from pathlib import Path
import os

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
    ErrorCode,
    MetricsResponse
)
from .inference import InferenceEngine, ExplanationEngine
from .auth import require_admin, require_clinician, get_current_active_user
from .users import router as auth_router

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fairness-Aware Skin Cancer Detection API",
    description="Production API for dermoscopy image classification with SHAP explanations and JWT authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include authentication router
app.include_router(auth_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global state
class AppState:
    """Application state container."""
    def __init__(self):
        self.inference_engine: Optional[InferenceEngine] = None
        self.explanation_engine: Optional[ExplanationEngine] = None
        self.start_time = time.time()
        self.total_requests = 0
        self.total_predictions = 0
        self.error_count = 0
        self.model_version = "v0.5.0-dev"

app_state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Fairness-Aware Skin Cancer Detection API...")

    # Model configuration (should be environment variables in production)
    model_path = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
    device = os.getenv("DEVICE", "cpu")
    use_onnx = os.getenv("USE_ONNX", "false").lower() == "true"

    try:
        # Initialize inference engine
        app_state.inference_engine = InferenceEngine(
            model_path=model_path,
            device=device,
            use_onnx=use_onnx
        )

        # Initialize explanation engine (if not using ONNX)
        if not use_onnx:
            app_state.explanation_engine = ExplanationEngine(
                model=app_state.inference_engine.model,
                device=app_state.inference_engine.device,
                method="saliency"  # Fast method for API
            )

        logger.info("Models loaded successfully")
        logger.info(f"Model info: {app_state.inference_engine.get_model_info()}")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fairness-Aware Skin Cancer Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns system status and model information.
    """
    model_info = app_state.inference_engine.get_model_info()
    perf_metrics = app_state.inference_engine.get_performance_metrics()

    uptime = time.time() - app_state.start_time

    return HealthResponse(
        status="healthy",
        model_version=app_state.model_version,
        model_size_mb=model_info["model_size_mb"],
        compression_type="onnx" if model_info["use_onnx"] else "pytorch",
        avg_inference_time_ms=perf_metrics.get("avg_inference_time_ms"),
        uptime_seconds=uptime,
        gpu_available=model_info["device"] != "cpu"
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(current_user: dict = Depends(require_admin)):
    """
    Get performance metrics (admin only).

    Returns detailed performance statistics.
    """
    perf_metrics = app_state.inference_engine.get_performance_metrics()
    uptime = time.time() - app_state.start_time

    error_rate = (
        app_state.error_count / app_state.total_requests
        if app_state.total_requests > 0
        else 0.0
    )

    return MetricsResponse(
        total_requests=app_state.total_requests,
        total_predictions=app_state.total_predictions,
        avg_inference_time_ms=perf_metrics["avg_inference_time_ms"],
        p95_inference_time_ms=perf_metrics["p95_inference_time_ms"],
        p99_inference_time_ms=perf_metrics["p99_inference_time_ms"],
        error_rate=error_rate,
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    return_explanation: bool = False,
    estimate_fst: bool = True,
    explanation_method: str = "saliency",
    current_user: dict = Depends(require_clinician)
):
    """
    Predict diagnosis for single dermoscopy image (clinician/admin only).

    Args:
        file: Image file (JPEG, PNG)
        return_explanation: Whether to return SHAP explanation
        estimate_fst: Whether to estimate Fitzpatrick skin type
        explanation_method: Explanation method (saliency, gradient_shap, integrated_gradients)
        current_user: Authenticated user (clinician or admin role required)

    Returns:
        Prediction with confidence, probabilities, and optional explanation
    """
    app_state.total_requests += 1

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG)"
            )

        # Read image
        image_bytes = await file.read()

        # Preprocess
        image_tensor = app_state.inference_engine.preprocess_image(image_bytes)

        # Inference
        predicted_class, confidence, class_probs, inference_time = \
            app_state.inference_engine.predict(image_tensor)

        app_state.total_predictions += 1

        # FST estimation
        fst_estimate = None
        if estimate_fst:
            fst_estimate = app_state.inference_engine.estimate_fst(image_tensor)

        # Explanation
        explanation = None
        if return_explanation and app_state.explanation_engine is not None:
            try:
                explanation = app_state.explanation_engine.generate_explanation(
                    image_tensor,
                    predicted_class,
                    fst_label=fst_estimate,
                    return_visualization=True
                )
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")
                # Continue without explanation

        # Response
        diagnosis = app_state.inference_engine.class_names[predicted_class]

        return PredictionResponse(
            diagnosis=diagnosis,
            confidence=confidence,
            class_probabilities=class_probs,
            fst_estimate=fst_estimate,
            explanation=explanation,
            inference_time_ms=inference_time,
            model_version=app_state.model_version
        )

    except HTTPException:
        raise
    except Exception as e:
        app_state.error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
@limiter.limit("5/minute")
async def batch_predict(
    request: Request,
    files: List[UploadFile] = File(...),
    return_explanations: bool = False,
    estimate_fst: bool = True,
    current_user: dict = Depends(require_clinician)
):
    """
    Batch prediction for multiple images (clinician/admin only).

    Args:
        files: List of image files
        return_explanations: Whether to return explanations
        estimate_fst: Whether to estimate FST
        current_user: Authenticated user (clinician or admin role required)

    Returns:
        List of predictions
    """
    app_state.total_requests += 1

    try:
        # Validate batch size
        max_batch_size = 32
        if len(files) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum ({max_batch_size})"
            )

        # Process all images
        image_tensors = []
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not an image"
                )

            image_bytes = await file.read()
            image_tensor = app_state.inference_engine.preprocess_image(image_bytes)
            image_tensors.append(image_tensor)

        # Stack into batch
        import torch
        batch_tensor = torch.cat(image_tensors, dim=0)

        # Batch inference
        predicted_classes, confidences, class_probs_list, total_time = \
            app_state.inference_engine.predict_batch(batch_tensor)

        app_state.total_predictions += len(files)

        # Create responses
        predictions = []
        for i in range(len(files)):
            diagnosis = app_state.inference_engine.class_names[predicted_classes[i]]

            fst_estimate = None
            if estimate_fst:
                fst_estimate = app_state.inference_engine.estimate_fst(image_tensors[i])

            explanation = None
            if return_explanations and app_state.explanation_engine is not None:
                try:
                    explanation = app_state.explanation_engine.generate_explanation(
                        image_tensors[i],
                        predicted_classes[i],
                        fst_label=fst_estimate,
                        return_visualization=False  # Disable viz for batch
                    )
                except Exception as e:
                    logger.error(f"Explanation failed for image {i}: {e}")

            pred_response = PredictionResponse(
                diagnosis=diagnosis,
                confidence=confidences[i],
                class_probabilities=class_probs_list[i],
                fst_estimate=fst_estimate,
                explanation=explanation,
                inference_time_ms=total_time / len(files),  # Average
                model_version=app_state.model_version
            )
            predictions.append(pred_response)

        return BatchPredictionResponse(
            predictions=predictions,
            total_inference_time_ms=total_time,
            batch_size=len(files)
        )

    except HTTPException:
        raise
    except Exception as e:
        app_state.error_count += 1
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    app_state.error_count += 1

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            error_code=ErrorCode.INTERNAL_ERROR
        ).dict()
    )


if __name__ == "__main__":
    """Run API with uvicorn."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
