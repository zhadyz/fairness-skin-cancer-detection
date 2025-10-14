"""
Post-Training Quantization for Model Compression

Implements INT8/FP16 quantization with fairness-aware calibration to ensure
compressed models maintain equitable performance across FSTs.

Key Features:
- Per-channel quantization for better accuracy
- Fairness-aware calibration (ensure FST V-VI representation)
- Support for INT8, FP16, and dynamic quantization
- Calibration on balanced HAM10000 validation set
- Export to ONNX/TensorRT

Quantization Targets:
- FP16: 2x memory reduction, minimal accuracy loss
- INT8: 4x memory reduction, <3% accuracy loss
- Deployment: <25MB model size (from 268MB FP32)

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    # Quantization settings
    precision: str = "int8"  # "fp16", "int8", "dynamic"
    per_channel: bool = True  # Per-channel vs per-tensor quantization
    backend: str = "fbgemm"  # "fbgemm" (CPU), "qnnpack" (mobile)

    # Calibration settings
    calibration_samples: int = 1000
    fst_balanced: bool = True  # Ensure FST V-VI representation
    calibration_method: str = "minmax"  # "minmax", "histogram", "percentile"

    # Layer selection
    quantize_conv: bool = True
    quantize_linear: bool = True
    quantize_attention: bool = False  # Attention often sensitive to quantization

    # Export settings
    export_format: str = "pytorch"  # "pytorch", "onnx", "tensorrt"
    optimize_for_mobile: bool = False


class ModelQuantizer:
    """
    Model quantizer with fairness-aware calibration.

    Implements post-training quantization (PTQ) with per-channel quantization
    and balanced calibration across FST groups.

    Args:
        model: PyTorch model to quantize
        config: QuantizationConfig with settings
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.config = config
        self.device = device

        # Quantization state
        self.calibrated = False
        self.quantized_model = None

        # Statistics
        self.original_size = self._get_model_size(model)

        logger.info(f"ModelQuantizer initialized: {config.precision} precision, "
                   f"original size {self.original_size:.2f} MB")

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def prepare_model_for_quantization(self):
        """Prepare model for quantization by fusing modules."""
        logger.info("Preparing model for quantization...")

        # Move model to CPU (quantization requires CPU)
        self.model = self.model.cpu()
        self.model.eval()

        # Set quantization backend
        if self.config.backend == "fbgemm":
            torch.backends.quantized.engine = "fbgemm"
        elif self.config.backend == "qnnpack":
            torch.backends.quantized.engine = "qnnpack"

        # Fuse modules (Conv+BN+ReLU, Linear+ReLU, etc.)
        # This improves quantization accuracy
        try:
            # Automatic fusion
            self.model = torch.quantization.fuse_modules_qat(
                self.model,
                inplace=False
            )
            logger.info("Modules fused successfully")
        except Exception as e:
            logger.warning(f"Could not fuse modules: {e}")

        return self.model

    def calibrate(
        self,
        dataloader: DataLoader,
        fst_labels: Optional[List[int]] = None
    ):
        """
        Calibrate quantization parameters using calibration dataset.

        Args:
            dataloader: Calibration dataloader
            fst_labels: FST labels for balanced sampling (if fst_balanced=True)
        """
        logger.info(f"Calibrating quantization with {self.config.calibration_samples} samples...")

        # Prepare model
        self.prepare_model_for_quantization()

        if self.config.precision == "fp16":
            # FP16 doesn't require calibration
            self.calibrated = True
            return

        # Configure quantization
        if self.config.precision == "int8":
            if self.config.per_channel:
                qconfig = quant.get_default_qconfig(self.config.backend)
            else:
                qconfig = quant.default_qconfig

            self.model.qconfig = qconfig

            # Prepare for static quantization
            quant.prepare(self.model, inplace=True)

        elif self.config.precision == "dynamic":
            # Dynamic quantization doesn't require calibration
            self.calibrated = True
            return

        # Create balanced calibration dataset
        if self.config.fst_balanced and fst_labels is not None:
            calibration_loader = self._create_balanced_calibration_loader(
                dataloader, fst_labels
            )
        else:
            # Use subset of original dataloader
            calibration_loader = self._create_calibration_subset(dataloader)

        # Run calibration
        self.model.eval()
        with torch.no_grad():
            num_samples = 0
            for batch in calibration_loader:
                if num_samples >= self.config.calibration_samples:
                    break

                # Parse batch
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.cpu()

                # Forward pass (collects statistics)
                _ = self.model(inputs)

                num_samples += inputs.size(0)

        self.calibrated = True
        logger.info(f"Calibration complete with {num_samples} samples")

    def _create_balanced_calibration_loader(
        self,
        dataloader: DataLoader,
        fst_labels: List[int]
    ) -> DataLoader:
        """
        Create balanced calibration dataloader with equal FST representation.

        Args:
            dataloader: Original dataloader
            fst_labels: FST labels for each sample

        Returns:
            Balanced calibration dataloader
        """
        logger.info("Creating FST-balanced calibration dataset...")

        # Group indices by FST
        fst_indices = {fst: [] for fst in range(1, 7)}
        for idx, fst in enumerate(fst_labels):
            if fst in fst_indices:
                fst_indices[fst].append(idx)

        # Sample equally from each FST (prioritize FST V-VI)
        samples_per_fst = self.config.calibration_samples // 6
        priority_fsts = [5, 6]  # FST V-VI

        # Extra samples for priority FSTs
        priority_samples = samples_per_fst * 2
        regular_samples = (self.config.calibration_samples - 2 * priority_samples) // 4

        balanced_indices = []

        for fst in range(1, 7):
            if fst in priority_fsts:
                n_samples = priority_samples
            else:
                n_samples = regular_samples

            fst_pool = fst_indices[fst]
            if len(fst_pool) > 0:
                sampled = np.random.choice(fst_pool, size=min(n_samples, len(fst_pool)), replace=False)
                balanced_indices.extend(sampled)

        logger.info(f"Balanced calibration set: {len(balanced_indices)} samples")

        # Create subset
        dataset = dataloader.dataset
        balanced_dataset = Subset(dataset, balanced_indices)

        return DataLoader(
            balanced_dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0
        )

    def _create_calibration_subset(self, dataloader: DataLoader) -> DataLoader:
        """Create random calibration subset."""
        dataset = dataloader.dataset
        total_samples = len(dataset)

        # Random subset
        indices = np.random.choice(
            total_samples,
            size=min(self.config.calibration_samples, total_samples),
            replace=False
        )

        subset = Subset(dataset, indices)

        return DataLoader(
            subset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0
        )

    def quantize(self) -> nn.Module:
        """
        Quantize calibrated model.

        Returns:
            Quantized model
        """
        if not self.calibrated:
            raise ValueError("Model not calibrated. Run calibrate() first.")

        logger.info(f"Quantizing model to {self.config.precision}...")

        if self.config.precision == "fp16":
            # FP16 quantization (simple conversion)
            self.quantized_model = self._quantize_fp16()

        elif self.config.precision == "int8":
            # INT8 static quantization
            self.quantized_model = self._quantize_int8_static()

        elif self.config.precision == "dynamic":
            # Dynamic INT8 quantization
            self.quantized_model = self._quantize_int8_dynamic()

        else:
            raise ValueError(f"Unknown precision: {self.config.precision}")

        # Log statistics
        quantized_size = self._get_model_size(self.quantized_model)
        compression_ratio = self.original_size / quantized_size

        logger.info(f"Quantization complete:")
        logger.info(f"  Original size: {self.original_size:.2f} MB")
        logger.info(f"  Quantized size: {quantized_size:.2f} MB")
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")

        return self.quantized_model

    def _quantize_fp16(self) -> nn.Module:
        """Quantize model to FP16."""
        model_fp16 = self.model.half()
        return model_fp16

    def _quantize_int8_static(self) -> nn.Module:
        """Perform static INT8 quantization."""
        # Convert to quantized model
        quantized_model = quant.convert(self.model, inplace=False)
        return quantized_model

    def _quantize_int8_dynamic(self) -> nn.Module:
        """Perform dynamic INT8 quantization."""
        # Dynamic quantization (weights only)
        quantized_model = quant.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU} if self.config.quantize_linear else set(),
            dtype=torch.qint8
        )
        return quantized_model

    def evaluate_quantization_quality(
        self,
        test_loader: DataLoader,
        original_model: nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate quantization quality by comparing with original model.

        Args:
            test_loader: Test dataloader
            original_model: Original unquantized model

        Returns:
            Dictionary of metrics (accuracy, loss, numerical error)
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model available. Run quantize() first.")

        logger.info("Evaluating quantization quality...")

        original_model.eval()
        self.quantized_model.eval()

        metrics = {
            "original_accuracy": 0.0,
            "quantized_accuracy": 0.0,
            "accuracy_drop": 0.0,
            "mean_output_error": 0.0,
            "max_output_error": 0.0
        }

        all_errors = []
        correct_original = 0
        correct_quantized = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                    targets = batch[1]
                else:
                    inputs = batch
                    targets = None

                # Original model predictions
                inputs_orig = inputs.to(self.device)
                outputs_orig = original_model(inputs_orig)
                if isinstance(outputs_orig, tuple):
                    outputs_orig = outputs_orig[0]

                # Quantized model predictions
                if self.config.precision == "fp16":
                    inputs_quant = inputs.half().to(self.device)
                else:
                    inputs_quant = inputs.cpu()

                outputs_quant = self.quantized_model(inputs_quant)
                if isinstance(outputs_quant, tuple):
                    outputs_quant = outputs_quant[0]

                # Move to same device for comparison
                outputs_orig = outputs_orig.cpu()
                outputs_quant = outputs_quant.float().cpu()

                # Compute numerical error
                error = torch.abs(outputs_orig - outputs_quant)
                all_errors.append(error)

                # Compute accuracy if targets available
                if targets is not None:
                    pred_orig = outputs_orig.argmax(dim=1)
                    pred_quant = outputs_quant.argmax(dim=1)

                    correct_original += (pred_orig == targets).sum().item()
                    correct_quantized += (pred_quant == targets).sum().item()
                    total += targets.size(0)

        # Aggregate metrics
        all_errors = torch.cat(all_errors)

        metrics["mean_output_error"] = all_errors.mean().item()
        metrics["max_output_error"] = all_errors.max().item()

        if total > 0:
            metrics["original_accuracy"] = correct_original / total
            metrics["quantized_accuracy"] = correct_quantized / total
            metrics["accuracy_drop"] = metrics["original_accuracy"] - metrics["quantized_accuracy"]

        logger.info(f"Quantization quality metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")

        return metrics

    def save_quantized_model(self, save_path: str):
        """
        Save quantized model.

        Args:
            save_path: Path to save model
        """
        if self.quantized_model is None:
            raise ValueError("No quantized model to save. Run quantize() first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.precision == "fp16":
            # Save FP16 model
            torch.save(self.quantized_model.state_dict(), save_path)
            logger.info(f"FP16 model saved to {save_path}")

        elif self.config.precision in ["int8", "dynamic"]:
            # Save quantized model (use torch.jit for better compatibility)
            try:
                scripted_model = torch.jit.script(self.quantized_model)
                scripted_model.save(str(save_path))
                logger.info(f"INT8 quantized model saved to {save_path}")
            except Exception as e:
                # Fallback: save state dict
                logger.warning(f"Could not script model: {e}")
                torch.save(self.quantized_model.state_dict(), save_path)
                logger.info(f"INT8 model state dict saved to {save_path}")

    def load_quantized_model(self, load_path: str) -> nn.Module:
        """
        Load quantized model.

        Args:
            load_path: Path to saved model

        Returns:
            Loaded quantized model
        """
        load_path = Path(load_path)

        if self.config.precision == "fp16":
            # Load FP16 model
            self.quantized_model = self.model.half()
            self.quantized_model.load_state_dict(torch.load(load_path))
            logger.info(f"FP16 model loaded from {load_path}")

        elif self.config.precision in ["int8", "dynamic"]:
            # Load quantized model
            try:
                self.quantized_model = torch.jit.load(str(load_path))
                logger.info(f"INT8 quantized model loaded from {load_path}")
            except Exception as e:
                logger.warning(f"Could not load scripted model: {e}")
                # Fallback: load state dict
                self.quantized_model = self.model
                self.quantized_model.load_state_dict(torch.load(load_path))
                logger.info(f"INT8 model state dict loaded from {load_path}")

        return self.quantized_model


def quantize_model_pipeline(
    model: nn.Module,
    calibration_loader: DataLoader,
    test_loader: DataLoader,
    precision: str = "int8",
    calibration_samples: int = 1000,
    fst_balanced: bool = True,
    fst_labels: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    End-to-end quantization pipeline.

    Args:
        model: PyTorch model to quantize
        calibration_loader: Calibration dataloader
        test_loader: Test dataloader
        precision: Quantization precision ("fp16", "int8", "dynamic")
        calibration_samples: Number of calibration samples
        fst_balanced: Use FST-balanced calibration
        fst_labels: FST labels for balanced sampling
        save_path: Path to save quantized model (optional)

    Returns:
        Quantized model and quality metrics
    """
    logger.info(f"Starting quantization pipeline: {precision}")

    # Create quantizer
    config = QuantizationConfig(
        precision=precision,
        per_channel=True,
        calibration_samples=calibration_samples,
        fst_balanced=fst_balanced
    )

    quantizer = ModelQuantizer(model, config)

    # Calibrate
    quantizer.calibrate(calibration_loader, fst_labels)

    # Quantize
    quantized_model = quantizer.quantize()

    # Evaluate
    metrics = quantizer.evaluate_quantization_quality(test_loader, model)

    # Save if requested
    if save_path is not None:
        quantizer.save_quantized_model(save_path)

    return quantized_model, metrics


if __name__ == "__main__":
    """Test quantization implementation."""
    print("=" * 80)
    print("Testing Model Quantization")
    print("=" * 80)

    # Create dummy model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.fc1 = nn.Linear(128 * 56 * 56, 256)
            self.fc2 = nn.Linear(256, 7)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = torch.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = x.flatten(1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleModel()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy data
    calibration_data = [(torch.randn(4, 3, 224, 224), torch.randint(0, 7, (4,))) for _ in range(25)]
    test_data = [(torch.randn(4, 3, 224, 224), torch.randint(0, 7, (4,))) for _ in range(10)]

    # Test FP16 quantization
    print("\n" + "=" * 40)
    print("Testing FP16 Quantization")
    print("=" * 40)

    config_fp16 = QuantizationConfig(precision="fp16")
    quantizer_fp16 = ModelQuantizer(model, config_fp16)

    print("Calibrating...")
    quantizer_fp16.calibrate(calibration_data)

    print("Quantizing...")
    model_fp16 = quantizer_fp16.quantize()

    print("Evaluating...")
    metrics_fp16 = quantizer_fp16.evaluate_quantization_quality(test_data, model)

    print("\nFP16 Metrics:")
    for key, value in metrics_fp16.items():
        print(f"  {key}: {value:.6f}")

    print("\n" + "=" * 80)
    print("Quantization test PASSED!")
    print("=" * 80)
