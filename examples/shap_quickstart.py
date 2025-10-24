"""
Quick Start: SHAP Explanation for Single Image

Minimal example demonstrating SHAP explanation generation for a single image.

Usage:
    python examples/shap_quickstart.py --image_path path/to/image.jpg
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.visualization import visualize_explanation


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess image."""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)

    return image_tensor


def create_dummy_model(num_classes: int = 7) -> nn.Module:
    """
    Create a dummy model for demonstration.

    In practice, replace this with your trained model.
    """
    from torchvision import models

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def main():
    parser = argparse.ArgumentParser(description="Quick SHAP Explanation for Single Image")
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (optional, uses dummy model if not provided)')
    parser.add_argument('--output_path', type=str, default='explanation_output.png',
                       help='Output path for visualization')
    parser.add_argument('--method', type=str, default='saliency',
                       choices=['gradient_shap', 'integrated_gradients', 'saliency'],
                       help='Attribution method (saliency is fastest for demo)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Computation device')

    args = parser.parse_args()

    print("=" * 80)
    print("Quick SHAP Explanation Demo")
    print("=" * 80)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load or create model
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        model = torch.load(args.model_path, map_location=device)
    else:
        print("Using dummy ResNet18 model (for demonstration)")
        print("IMPORTANT: Replace with your trained model for real results!")
        model = create_dummy_model()

    model.to(device)
    model.eval()

    # Load image
    print(f"Loading image from {args.image_path}")
    image_tensor = load_image(args.image_path)

    # Create explainer
    print(f"Creating SHAP explainer with method: {args.method}")
    explainer = SHAPExplainer(
        model=model,
        device=device,
        method=args.method
    )

    # Set background data if using GradientSHAP
    if args.method == "gradient_shap":
        print("Note: GradientSHAP requires background data.")
        print("Using random background for demo (not recommended for production)")
        # Create random background
        background = torch.randn(10, 3, 224, 224).to(device)
        explainer.background_data = background

    # Generate explanation
    print("Generating explanation...")
    result = explainer.explain_prediction(
        image=image_tensor,
        target_class=None  # Use predicted class
    )

    print(f"\nResults:")
    print(f"  Predicted class: {result.predicted_class}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Attribution magnitude: {result.attribution_magnitude:.6f}")
    print(f"  Concentration score: {result.concentration_score:.3f}")
    print(f"  Computation time: {result.computation_time:.3f}s")

    # Create visualization
    print(f"Creating visualization...")
    fig = visualize_explanation(
        result=result,
        show_top_regions=True,
        save_path=args.output_path
    )

    print(f"\nVisualization saved to: {args.output_path}")
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    main()
