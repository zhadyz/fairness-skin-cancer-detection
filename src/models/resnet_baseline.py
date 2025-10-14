"""
ResNet50 Baseline Model for Skin Cancer Detection

Transfer learning from ImageNet with custom classification head for HAM10000/ISIC datasets.
Implements best practices for medical image classification.

Author: HOLLOWED_EYES
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any


class ResNet50Baseline(nn.Module):
    """
    ResNet50 baseline model with transfer learning from ImageNet.

    Architecture:
    - Pre-trained ResNet50 backbone (torchvision)
    - Global Average Pooling
    - Dropout (0.5)
    - Fully connected layer (2048 -> num_classes)

    Args:
        num_classes: Number of output classes (7 for HAM10000, 8 for ISIC)
        pretrained: Load ImageNet pre-trained weights
        freeze_backbone: Freeze backbone layers (fine-tune only classifier)
        dropout: Dropout rate before final classifier
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        super(ResNet50Baseline, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load pre-trained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)

        # Freeze backbone if requested (useful for initial fine-tuning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        # ResNet50 has 2048 features before FC layer
        in_features = self.backbone.fc.in_features

        # Custom classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        # Store model info for checkpointing
        self.model_info = {
            'architecture': 'resnet50',
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone,
            'dropout': dropout
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W) - typically (B, 3, 224, 224)

        Returns:
            Logits tensor (B, num_classes)
        """
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers: Optional[int] = None):
        """
        Unfreeze backbone layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end.
                       If None, unfreezes all layers.
        """
        if num_layers is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last num_layers
            # ResNet50 structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
            layers = [
                self.backbone.layer4,
                self.backbone.layer3,
                self.backbone.layer2,
                self.backbone.layer1
            ]

            for i in range(min(num_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = True

    def get_feature_extractor(self) -> nn.Module:
        """
        Returns the model without the classification head (for feature extraction).

        Returns:
            Feature extractor module (outputs 2048-dim features)
        """
        return nn.Sequential(*list(self.backbone.children())[:-1])

    def get_model_info(self) -> Dict[str, Any]:
        """Returns model configuration info."""
        return self.model_info


def create_resnet50_model(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.5
) -> ResNet50Baseline:
    """
    Factory function to create ResNet50 baseline model.

    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Freeze backbone during initial training
        dropout: Dropout rate

    Returns:
        ResNet50Baseline model instance

    Example:
        >>> model = create_resnet50_model(num_classes=7, pretrained=True)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([4, 7])
    """
    model = ResNet50Baseline(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing ResNet50 Baseline Model...")

    # Create model
    model = create_resnet50_model(num_classes=7, pretrained=True)

    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model info: {model.get_model_info()}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nResNet50 baseline model test PASSED!")
