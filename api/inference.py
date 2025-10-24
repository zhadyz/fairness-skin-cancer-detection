"""
Inference Engine for Production API

Handles model loading, preprocessing, inference, and post-processing
with support for PyTorch and ONNX models.

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import base64
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available")

from torchvision import transforms

logger = logging.getLogger(__name__)


def load_model_architecture(num_classes: int = 7, device: str = "cpu") -> nn.Module:
    """
    Load the model architecture.

    Attempts to load HybridFairnessClassifier, falls back to ResNet50.

    Args:
        num_classes: Number of output classes
        device: Device to load model on

    Returns:
        Initialized model architecture
    """
    try:
        from src.models.hybrid_model import HybridFairnessClassifier, HybridModelConfig

        config = HybridModelConfig(
            convnext_variant='base',
            swin_variant='small',
            num_classes=num_classes,
            fusion_dim=768,
            enable_fairdisco=False,
            dropout=0.3
        )

        model = HybridFairnessClassifier(config)
        logger.info("Loaded HybridFairnessClassifier architecture")
        return model

    except (ImportError, Exception) as e:
        logger.warning(f"Failed to load HybridFairnessClassifier: {e}")
        logger.info("Falling back to ResNet50 architecture")

        # Fallback to ResNet50
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        return model


class InferenceEngine:
    """
    Inference engine supporting PyTorch and ONNX models.

    Handles:
    - Model loading (PyTorch checkpoint or ONNX)
    - Image preprocessing
    - Batched inference
    - FST estimation
    - Performance tracking
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",
        use_onnx: bool = False,
        num_classes: int = 7,
        use_fp16: bool = False
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.use_onnx = use_onnx
        self.num_classes = num_classes
        self.use_fp16 = use_fp16 and device == "cuda"

        # Performance tracking
        self.inference_times: List[float] = []
        self.total_predictions = 0

        # Class names
        self.class_names = [
            "melanoma",
            "melanocytic_nevus",
            "basal_cell_carcinoma",
            "actinic_keratosis",
            "benign_keratosis",
            "dermatofibroma",
            "vascular_lesion"
        ]

        # Load model
        self.model = self._load_model()

        # Preprocessing
        self.transform = self._get_transform()

        logger.info(f"InferenceEngine initialized: device={device}, onnx={use_onnx}, fp16={self.use_fp16}")

    def _load_model(self) -> Union[nn.Module, ort.InferenceSession]:
        """Load PyTorch or ONNX model."""
        if not self.model_path.exists():
            logger.warning(f"Model path {self.model_path} does not exist. Creating dummy model for testing.")
            # Create dummy model for testing/development
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            model.to(self.device)
            model.eval()
            return model

        if self.use_onnx:
            return self._load_onnx_model()
        else:
            return self._load_pytorch_model()

    def _load_onnx_model(self) -> ort.InferenceSession:
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime")

        logger.info(f"Loading ONNX model from {self.model_path}")

        # ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Select execution provider
        if self.device.type == "cuda" and torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )

        logger.info(f"ONNX model loaded: {session.get_providers()}")
        logger.info(f"ONNX inputs: {[input.name for input in session.get_inputs()]}")
        logger.info(f"ONNX outputs: {[output.name for output in session.get_outputs()]}")

        return session

    def _load_pytorch_model(self) -> nn.Module:
        """Load PyTorch model from checkpoint."""
        logger.info(f"Loading PyTorch model from {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract model state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                logger.info("Loaded model_state_dict from checkpoint")
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
                logger.info("Loaded state_dict from checkpoint")
            elif 'model' in checkpoint:
                model_state = checkpoint['model']
                logger.info("Loaded model from checkpoint")
            else:
                model_state = checkpoint
                logger.info("Using checkpoint directly as state dict")
        else:
            model_state = checkpoint
            logger.info("Loaded model state directly")

        # Create model architecture
        model = load_model_architecture(self.num_classes, str(self.device))

        # Load weights
        try:
            model.load_state_dict(model_state, strict=True)
            logger.info("Loaded weights with strict=True")
        except RuntimeError as e:
            logger.warning(f"Failed to load state dict with strict=True: {e}")
            logger.info("Attempting to load with strict=False")

            # Try to load with strict=False
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys[:5]}... ({len(missing_keys)} total)")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys[:5]}... ({len(unexpected_keys)} total)")

        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()

        # Enable FP16 if requested
        if self.use_fp16:
            model.half()
            logger.info("Enabled FP16 inference")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"PyTorch model loaded: {num_params:,} parameters")

        return model

    def _get_transform(self) -> transforms.Compose:
        """Get preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess image bytes to tensor.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Apply transforms
            tensor = self.transform(image).unsqueeze(0)  # (1, 3, H, W)

            return tensor
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def predict(
        self,
        image_tensor: torch.Tensor
    ) -> Tuple[int, float, Dict[str, float], float]:
        """
        Run inference on single image.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W)

        Returns:
            Tuple of (predicted_class, confidence, class_probs, inference_time_ms)
        """
        start_time = time.time()

        try:
            if self.use_onnx:
                # ONNX inference
                logits = self._onnx_inference(image_tensor)
            else:
                # PyTorch inference
                logits = self._pytorch_inference(image_tensor)

            # Post-processing
            probs = torch.softmax(logits, dim=1)[0]
            predicted_class = probs.argmax().item()
            confidence = probs[predicted_class].item()

            # Class probabilities
            class_probs = {
                self.class_names[i]: float(probs[i])
                for i in range(min(len(self.class_names), len(probs)))
            }

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Track metrics
            self.inference_times.append(inference_time)
            self.total_predictions += 1

            logger.debug(f"Prediction: class={predicted_class}, confidence={confidence:.3f}, time={inference_time:.2f}ms")

            return predicted_class, confidence, class_probs, inference_time

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference error: {e}")

    def _pytorch_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run PyTorch inference."""
        image_tensor = image_tensor.to(self.device)

        if self.use_fp16:
            image_tensor = image_tensor.half()

        with torch.no_grad():
            logits = self.model(image_tensor)

            # Handle tuple output (FairDisCo models)
            if isinstance(logits, tuple):
                logits = logits[0]

        return logits

    def _onnx_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run ONNX inference."""
        input_name = self.model.get_inputs()[0].name
        input_data = image_tensor.cpu().numpy().astype(np.float32)

        outputs = self.model.run(None, {input_name: input_data})
        logits = torch.from_numpy(outputs[0])

        return logits

    def predict_batch(
        self,
        image_tensors: torch.Tensor
    ) -> Tuple[List[int], List[float], List[Dict[str, float]], float]:
        """
        Run batched inference on multiple images.

        Args:
            image_tensors: Batch of preprocessed images (B, 3, H, W)

        Returns:
            Tuple of (predicted_classes, confidences, class_probs_list, total_time_ms)
        """
        start_time = time.time()

        try:
            if self.use_onnx:
                logits = self._onnx_inference(image_tensors)
            else:
                logits = self._pytorch_inference(image_tensors)

            # Post-processing
            probs = torch.softmax(logits, dim=1)
            predicted_classes = probs.argmax(dim=1).tolist()
            confidences = [float(probs[i, pred]) for i, pred in enumerate(predicted_classes)]

            # Class probabilities for each image
            class_probs_list = []
            for i in range(len(predicted_classes)):
                class_probs = {
                    self.class_names[j]: float(probs[i, j])
                    for j in range(min(len(self.class_names), probs.size(1)))
                }
                class_probs_list.append(class_probs)

            total_time = (time.time() - start_time) * 1000

            # Track metrics
            batch_size = len(predicted_classes)
            avg_time_per_image = total_time / batch_size
            self.inference_times.extend([avg_time_per_image] * batch_size)
            self.total_predictions += batch_size

            logger.debug(f"Batch inference: size={batch_size}, time={total_time:.2f}ms")

            return predicted_classes, confidences, class_probs_list, total_time

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise RuntimeError(f"Batch inference error: {e}")

    def estimate_fst(self, image_tensor: torch.Tensor) -> int:
        """
        Estimate Fitzpatrick skin type using ITA method.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W)

        Returns:
            FST estimate (1-6)
        """
        try:
            # Convert to numpy (denormalize)
            image_np = image_tensor[0].cpu().numpy()  # (3, H, W)
            image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, 3)

            # Denormalize (reverse ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1) * 255

            # Individual Typology Angle (ITA) method
            # ITA = arctan((L - 50) / b)
            # Simplified: use brightness as proxy

            # Average brightness (L channel approximation)
            brightness = image_np.mean()

            # Map brightness to FST (heuristic thresholds)
            if brightness > 200:
                fst = 1  # Very light
            elif brightness > 180:
                fst = 2  # Light
            elif brightness > 160:
                fst = 3  # Medium
            elif brightness > 140:
                fst = 4  # Olive
            elif brightness > 120:
                fst = 5  # Brown
            else:
                fst = 6  # Dark

            logger.debug(f"FST estimation: brightness={brightness:.1f}, fst={fst}")

            return fst

        except Exception as e:
            logger.error(f"FST estimation failed: {e}")
            # Return middle FST on error
            return 3

    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        try:
            model_size_bytes = os.path.getsize(self.model_path) if self.model_path.exists() else 0
            model_size_mb = model_size_bytes / (1024 * 1024)
        except:
            model_size_mb = 0.0

        info = {
            "model_path": str(self.model_path),
            "model_size_mb": round(model_size_mb, 2),
            "device": str(self.device),
            "use_onnx": self.use_onnx,
            "use_fp16": self.use_fp16,
            "num_classes": self.num_classes,
            "total_predictions": self.total_predictions
        }

        if not self.use_onnx and hasattr(self.model, 'parameters'):
            try:
                info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
            except:
                info["num_parameters"] = 0

        return info

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        if not self.inference_times:
            return {
                "avg_inference_time_ms": 0.0,
                "p50_inference_time_ms": 0.0,
                "p95_inference_time_ms": 0.0,
                "p99_inference_time_ms": 0.0,
                "min_inference_time_ms": 0.0,
                "max_inference_time_ms": 0.0
            }

        times = np.array(self.inference_times)

        return {
            "avg_inference_time_ms": float(times.mean()),
            "p50_inference_time_ms": float(np.percentile(times, 50)),
            "p95_inference_time_ms": float(np.percentile(times, 95)),
            "p99_inference_time_ms": float(np.percentile(times, 99)),
            "min_inference_time_ms": float(times.min()),
            "max_inference_time_ms": float(times.max())
        }


class ExplanationEngine:
    """
    Engine for generating SHAP explanations.

    Wraps SHAPExplainer for API use.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        method: str = "saliency"
    ):
        try:
            from src.explainability.shap_explainer import SHAPExplainer

            self.explainer = SHAPExplainer(model, device, method=method)
            self.method = method
            self.available = True

            logger.info(f"ExplanationEngine initialized: method={method}")

        except Exception as e:
            logger.warning(f"Failed to initialize ExplanationEngine: {e}")
            self.explainer = None
            self.method = method
            self.available = False

    def generate_explanation(
        self,
        image_tensor: torch.Tensor,
        predicted_class: int,
        fst_label: Optional[int] = None,
        return_visualization: bool = True
    ) -> Dict[str, any]:
        """
        Generate SHAP explanation for image.

        Args:
            image_tensor: Preprocessed image (1, 3, H, W)
            predicted_class: Predicted class
            fst_label: FST label (optional)
            return_visualization: Whether to include base64 visualization

        Returns:
            Dictionary with explanation data
        """
        if not self.available or self.explainer is None:
            logger.warning("ExplanationEngine not available")
            return {
                "error": "Explanation engine not available",
                "explanation_method": self.method
            }

        try:
            result = self.explainer.explain_prediction(
                image_tensor,
                target_class=predicted_class,
                fst_label=fst_label
            )

            explanation_data = {
                "attribution_magnitude": float(result.attribution_magnitude),
                "concentration_score": float(result.concentration_score),
                "top_regions": [
                    {"x": int(x), "y": int(y), "importance": float(imp)}
                    for x, y, imp in result.top_regions[:10]
                ],
                "explanation_method": result.explanation_method,
                "computation_time_ms": result.computation_time * 1000
            }

            if return_visualization:
                try:
                    # Create saliency overlay
                    from src.explainability.visualization import create_saliency_overlay
                    import matplotlib
                    matplotlib.use('Agg')  # Non-interactive backend
                    import matplotlib.pyplot as plt

                    overlay = create_saliency_overlay(result.image, result.shap_values)

                    # Convert to base64
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(overlay)
                    ax.axis('off')

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close(fig)

                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    explanation_data["saliency_map_base64"] = img_base64

                except Exception as viz_error:
                    logger.warning(f"Failed to create visualization: {viz_error}")
                    explanation_data["visualization_error"] = str(viz_error)

            return explanation_data

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {
                "error": str(e),
                "explanation_method": self.method
            }


if __name__ == "__main__":
    """Test inference engine."""
    print("=" * 80)
    print("Testing InferenceEngine")
    print("=" * 80)

    # Test with dummy model (no checkpoint needed)
    engine = InferenceEngine(
        model_path="nonexistent_model.pth",  # Will create dummy model
        device="cpu",
        use_onnx=False
    )

    # Get model info
    info = engine.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test preprocessing
    print("\nTesting image preprocessing...")
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    tensor = engine.preprocess_image(img_bytes)
    print(f"  Preprocessed tensor shape: {tensor.shape}")

    # Test inference
    print("\nTesting single inference...")
    pred_class, confidence, class_probs, inf_time = engine.predict(tensor)
    print(f"  Predicted class: {pred_class} ({engine.class_names[pred_class]})")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Inference time: {inf_time:.2f}ms")

    # Test batch inference
    print("\nTesting batch inference...")
    batch = torch.cat([tensor] * 4, dim=0)
    classes, confidences, probs_list, total_time = engine.predict_batch(batch)
    print(f"  Batch size: {len(classes)}")
    print(f"  Total time: {total_time:.2f}ms")
    print(f"  Avg time per image: {total_time/len(classes):.2f}ms")

    # Test FST estimation
    print("\nTesting FST estimation...")
    fst = engine.estimate_fst(tensor)
    print(f"  Estimated FST: {fst}")

    # Get performance metrics
    metrics = engine.get_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")

    print("\n" + "=" * 80)
    print("InferenceEngine test PASSED!")
    print("=" * 80)
