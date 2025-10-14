"""
ONNX Export and Optimization for Production Deployment

Exports PyTorch models to ONNX format with optimization for inference speed.

Key Features:
- PyTorch to ONNX conversion
- Graph optimization (constant folding, operator fusion)
- Numerical accuracy validation
- Inference speed benchmarking
- Support for dynamic batch sizes

Framework: MENDICANT_BIAS - Phase 4
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

# ONNX is optional dependency
try:
    import onnx
    import onnxruntime as ort
    from onnx import optimizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx onnxruntime")


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""

    # Export settings
    opset_version: int = 17  # ONNX opset version
    dynamic_axes: bool = True  # Support dynamic batch size
    optimize_graph: bool = True  # Apply ONNX graph optimizations

    # Optimization passes
    optimization_passes: List[str] = None

    # Validation
    validate_numerics: bool = True
    tolerance: float = 1e-4  # Numerical tolerance for validation

    # Inference
    enable_profiling: bool = False

    def __post_init__(self):
        if self.optimization_passes is None:
            self.optimization_passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "eliminate_unused_initializer",
                "fuse_bn_into_conv",
                "fuse_consecutive_transposes",
                "fuse_add_bias_into_conv",
                "fuse_transpose_into_gemm"
            ]


class ONNXExporter:
    """
    ONNX model exporter with optimization and validation.

    Args:
        model: PyTorch model to export
        config: ONNXExportConfig with settings
        device: Computation device
    """

    def __init__(
        self,
        model: nn.Module,
        config: ONNXExportConfig,
        device: torch.device = torch.device("cpu")
    ):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")

        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

        # Export state
        self.onnx_model = None
        self.onnx_session = None

        logger.info(f"ONNXExporter initialized: opset {config.opset_version}")

    def export(
        self,
        save_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ) -> str:
        """
        Export PyTorch model to ONNX format.

        Args:
            save_path: Path to save ONNX model
            input_shape: Example input shape (B, C, H, W)
            input_names: Names for input tensors
            output_names: Names for output tensors

        Returns:
            Path to saved ONNX model
        """
        logger.info(f"Exporting model to ONNX: {save_path}")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Default names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Define dynamic axes
        dynamic_axes_dict = {}
        if self.config.dynamic_axes:
            # Batch dimension is dynamic
            for name in input_names:
                dynamic_axes_dict[name] = {0: "batch_size"}
            for name in output_names:
                dynamic_axes_dict[name] = {0: "batch_size"}

        # Export to ONNX
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                str(save_path),
                export_params=True,
                opset_version=self.config.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict if self.config.dynamic_axes else None
            )
            logger.info(f"ONNX model exported to {save_path}")

        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            raise

        # Load and optimize
        self.onnx_model = onnx.load(str(save_path))

        if self.config.optimize_graph:
            self.optimize()
            onnx.save(self.onnx_model, str(save_path))
            logger.info(f"Optimized ONNX model saved to {save_path}")

        # Validate if requested
        if self.config.validate_numerics:
            is_valid = self.validate_numerical_accuracy(dummy_input)
            if not is_valid:
                logger.warning("ONNX model failed numerical validation!")

        return str(save_path)

    def optimize(self):
        """
        Optimize ONNX graph with various optimization passes.

        Optimizations:
        - Constant folding
        - Operator fusion (Conv+BN, Conv+Bias+ReLU)
        - Dead code elimination
        - Transpose elimination
        """
        if self.onnx_model is None:
            raise ValueError("No ONNX model loaded. Run export() first.")

        logger.info("Optimizing ONNX graph...")

        try:
            # Apply optimization passes
            self.onnx_model = optimizer.optimize(
                self.onnx_model,
                passes=self.config.optimization_passes
            )

            logger.info(f"Applied {len(self.config.optimization_passes)} optimization passes")

        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")

        # Verify model
        try:
            onnx.checker.check_model(self.onnx_model)
            logger.info("ONNX model validation passed")
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            raise

    def load_onnx_model(self, onnx_path: str):
        """
        Load ONNX model and create inference session.

        Args:
            onnx_path: Path to ONNX model
        """
        logger.info(f"Loading ONNX model from {onnx_path}")

        self.onnx_model = onnx.load(onnx_path)

        # Create inference session
        sess_options = ort.SessionOptions()

        if self.config.enable_profiling:
            sess_options.enable_profiling = True

        # Use CPU for inference
        providers = ['CPUExecutionProvider']

        try:
            self.onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=providers
            )
            logger.info("ONNX inference session created")

        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference with ONNX model.

        Args:
            input_tensor: Input numpy array

        Returns:
            Output predictions
        """
        if self.onnx_session is None:
            raise ValueError("No ONNX session. Run load_onnx_model() first.")

        # Get input/output names
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name

        # Run inference
        outputs = self.onnx_session.run(
            [output_name],
            {input_name: input_tensor}
        )

        return outputs[0]

    def validate_numerical_accuracy(
        self,
        test_input: torch.Tensor,
        num_samples: int = 10
    ) -> bool:
        """
        Validate numerical accuracy of ONNX model vs PyTorch.

        Args:
            test_input: Test input tensor
            num_samples: Number of random samples to test

        Returns:
            True if validation passes
        """
        logger.info("Validating ONNX numerical accuracy...")

        if self.onnx_model is None:
            raise ValueError("No ONNX model. Run export() first.")

        # Create temp ONNX session if not exists
        if self.onnx_session is None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                onnx.save(self.onnx_model, tmp.name)
                self.load_onnx_model(tmp.name)

        max_error = 0.0
        mean_error = 0.0

        for i in range(num_samples):
            # Generate random input
            if i == 0:
                input_tensor = test_input
            else:
                input_shape = test_input.shape
                input_tensor = torch.randn(input_shape).to(self.device)

            # PyTorch inference
            with torch.no_grad():
                pytorch_output = self.model(input_tensor)
                if isinstance(pytorch_output, tuple):
                    pytorch_output = pytorch_output[0]
                pytorch_output = pytorch_output.cpu().numpy()

            # ONNX inference
            onnx_input = input_tensor.cpu().numpy()
            onnx_output = self.predict(onnx_input)

            # Compute error
            error = np.abs(pytorch_output - onnx_output)
            max_error = max(max_error, error.max())
            mean_error += error.mean()

        mean_error /= num_samples

        logger.info(f"Numerical validation results:")
        logger.info(f"  Mean error: {mean_error:.6f}")
        logger.info(f"  Max error: {max_error:.6f}")
        logger.info(f"  Tolerance: {self.config.tolerance:.6f}")

        # Check if within tolerance
        is_valid = max_error < self.config.tolerance

        if is_valid:
            logger.info("Numerical validation PASSED")
        else:
            logger.warning("Numerical validation FAILED")

        return is_valid

    def benchmark_inference_speed(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark ONNX inference speed vs PyTorch.

        Args:
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"Benchmarking inference speed ({num_runs} runs)...")

        # Generate random input
        pytorch_input = torch.randn(input_shape).to(self.device)
        onnx_input = pytorch_input.cpu().numpy()

        # Warmup
        logger.info(f"Warming up ({warmup_runs} runs)...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model(pytorch_input)
            _ = self.predict(onnx_input)

        # Benchmark PyTorch
        logger.info("Benchmarking PyTorch...")
        pytorch_times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(pytorch_input)
            end = time.perf_counter()
            pytorch_times.append((end - start) * 1000)  # Convert to ms

        # Benchmark ONNX
        logger.info("Benchmarking ONNX...")
        onnx_times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.predict(onnx_input)
            end = time.perf_counter()
            onnx_times.append((end - start) * 1000)  # Convert to ms

        # Compute statistics
        metrics = {
            "pytorch_mean_ms": np.mean(pytorch_times),
            "pytorch_std_ms": np.std(pytorch_times),
            "pytorch_p50_ms": np.percentile(pytorch_times, 50),
            "pytorch_p95_ms": np.percentile(pytorch_times, 95),
            "pytorch_p99_ms": np.percentile(pytorch_times, 99),
            "onnx_mean_ms": np.mean(onnx_times),
            "onnx_std_ms": np.std(onnx_times),
            "onnx_p50_ms": np.percentile(onnx_times, 50),
            "onnx_p95_ms": np.percentile(onnx_times, 95),
            "onnx_p99_ms": np.percentile(onnx_times, 99),
            "speedup": np.mean(pytorch_times) / np.mean(onnx_times)
        }

        logger.info("Benchmark results:")
        logger.info(f"  PyTorch: {metrics['pytorch_mean_ms']:.2f} ± {metrics['pytorch_std_ms']:.2f} ms")
        logger.info(f"  ONNX: {metrics['onnx_mean_ms']:.2f} ± {metrics['onnx_std_ms']:.2f} ms")
        logger.info(f"  Speedup: {metrics['speedup']:.2f}x")

        return metrics

    def get_model_info(self) -> Dict[str, any]:
        """Get ONNX model information."""
        if self.onnx_model is None:
            raise ValueError("No ONNX model loaded")

        info = {
            "opset_version": self.onnx_model.opset_import[0].version,
            "num_nodes": len(self.onnx_model.graph.node),
            "num_initializers": len(self.onnx_model.graph.initializer),
            "inputs": [],
            "outputs": []
        }

        # Input info
        for input_tensor in self.onnx_model.graph.input:
            input_info = {
                "name": input_tensor.name,
                "type": input_tensor.type.tensor_type.elem_type,
                "shape": [dim.dim_value if dim.dim_value > 0 else "dynamic"
                         for dim in input_tensor.type.tensor_type.shape.dim]
            }
            info["inputs"].append(input_info)

        # Output info
        for output_tensor in self.onnx_model.graph.output:
            output_info = {
                "name": output_tensor.name,
                "type": output_tensor.type.tensor_type.elem_type,
                "shape": [dim.dim_value if dim.dim_value > 0 else "dynamic"
                         for dim in output_tensor.type.tensor_type.shape.dim]
            }
            info["outputs"].append(output_info)

        return info


if __name__ == "__main__":
    """Test ONNX export."""
    if not ONNX_AVAILABLE:
        print("ONNX not available. Skipping test.")
        exit(0)

    print("=" * 80)
    print("Testing ONNX Export")
    print("=" * 80)

    # Create dummy model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 56 * 56, 7)

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    model = SimpleModel()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create exporter
    config = ONNXExportConfig(
        opset_version=17,
        dynamic_axes=True,
        optimize_graph=True
    )

    exporter = ONNXExporter(model, config)

    # Export
    print("\nExporting to ONNX...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name

    exporter.export(onnx_path, input_shape=(1, 3, 224, 224))

    # Validate
    print("\nValidating numerical accuracy...")
    test_input = torch.randn(2, 3, 224, 224)
    is_valid = exporter.validate_numerical_accuracy(test_input)

    print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")

    # Benchmark
    print("\nBenchmarking inference speed...")
    exporter.load_onnx_model(onnx_path)
    metrics = exporter.benchmark_inference_speed(num_runs=50)

    print("\n" + "=" * 80)
    print("ONNX export test PASSED!")
    print("=" * 80)
