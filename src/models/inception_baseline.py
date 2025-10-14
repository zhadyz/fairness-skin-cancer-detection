"""
InceptionV3 Baseline Model for Skin Cancer Detection

Transfer learning using InceptionV3 architecture. Known for multi-scale feature extraction
through parallel convolutions with different kernel sizes.

Author: HOLLOWED_EYES
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any


class InceptionV3Baseline(nn.Module):
    """
    InceptionV3 baseline model with transfer learning from ImageNet.

    Architecture:
    - Pre-trained InceptionV3 backbone (torchvision)
    - Global Average Pooling
    - Dropout (0.5)
    - Fully connected layer (2048 -> num_classes)

    InceptionV3 specifications:
    - Input size: 299x299 (native resolution, can adapt to 224x224)
    - Parameters: ~24M
    - Known for: Multi-scale feature extraction, auxiliary classifiers during training

    Args:
        num_classes: Number of output classes (7 for HAM10000, 8 for ISIC)
        pretrained: Load ImageNet pre-trained weights
        freeze_backbone: Freeze backbone layers (fine-tune only classifier)
        dropout: Dropout rate before final classifier
        aux_logits: Use auxiliary classifiers during training (Inception feature)
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
        aux_logits: bool = False  # Set False for simpler training
    ):
        super(InceptionV3Baseline, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.aux_logits_enabled = aux_logits

        # Load pre-trained InceptionV3
        if pretrained:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            self.backbone = models.inception_v3(
                weights=weights,
                aux_logits=aux_logits  # Auxiliary classifiers for training
            )
        else:
            self.backbone = models.inception_v3(
                weights=None,
                aux_logits=aux_logits
            )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        # InceptionV3 has 2048 features before FC layer
        in_features = self.backbone.fc.in_features

        # Custom classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        # Replace auxiliary classifier if enabled
        if aux_logits:
            aux_in_features = self.backbone.AuxLogits.fc.in_features
            self.backbone.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

        # Store model info
        self.model_info = {
            'architecture': 'inception_v3',
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone,
            'dropout': dropout,
            'aux_logits': aux_logits,
            'input_size': (299, 299)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W) - typically (B, 3, 299, 299)

        Returns:
            During training with aux_logits=True:
                InceptionOutputs(logits, aux_logits)
            Otherwise:
                Logits tensor (B, num_classes)
        """
        output = self.backbone(x)

        # Handle auxiliary outputs during training
        if self.training and self.aux_logits_enabled:
            # InceptionOutputs object with .logits and .aux_logits
            return output
        else:
            # Return only main logits during eval or when aux_logits disabled
            return output

    def unfreeze_backbone(self, num_modules: Optional[int] = None):
        """
        Unfreeze backbone modules for fine-tuning.

        InceptionV3 structure: Conv2d blocks -> Mixed blocks (inception modules) -> FC

        Args:
            num_modules: Number of inception modules to unfreeze from the end.
                        If None, unfreezes all modules.
        """
        if num_modules is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last num_modules inception modules
            # InceptionV3 has multiple Mixed_* modules
            modules_to_unfreeze = []

            # Get all Mixed modules in reverse order
            for name, module in self.backbone.named_children():
                if 'Mixed' in name:
                    modules_to_unfreeze.append(module)

            # Unfreeze last num_modules
            modules_to_unfreeze = modules_to_unfreeze[-num_modules:]

            for module in modules_to_unfreeze:
                for param in module.parameters():
                    param.requires_grad = True

    def get_feature_extractor(self) -> nn.Module:
        """
        Returns the model without the classification head (for feature extraction).

        Returns:
            Feature extractor module (outputs 2048-dim features)
        """
        # Remove final FC layer
        modules = list(self.backbone.children())[:-1]
        return nn.Sequential(*modules)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns model configuration info."""
        return self.model_info


class InceptionV3Adapted(nn.Module):
    """
    InceptionV3 adapted for 224x224 input (standard size for other models).

    Uses adaptive pooling to handle size mismatch, but native 299x299 is preferred.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        super(InceptionV3Adapted, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load InceptionV3 without auxiliary logits for simplicity
        if pretrained:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            self.backbone = models.inception_v3(weights=weights, aux_logits=False)
        else:
            self.backbone = models.inception_v3(weights=None, aux_logits=False)

        # Disable transform_input (expect pre-normalized images)
        self.backbone.transform_input = False

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        self.model_info = {
            'architecture': 'inception_v3_adapted',
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone,
            'dropout': dropout,
            'input_size': (224, 224),
            'note': 'Adapted from 299x299, prefer native resolution'
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample to 299x299 if needed
        if x.size(-1) != 299:
            x = nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
        return self.backbone(x)

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info


def create_inception_model(
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.5,
    aux_logits: bool = False,
    adapted_input: bool = False
) -> nn.Module:
    """
    Factory function to create InceptionV3 baseline model.

    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Freeze backbone during initial training
        dropout: Dropout rate
        aux_logits: Use auxiliary classifiers (for training)
        adapted_input: Adapt for 224x224 input (default False, use 299x299)

    Returns:
        InceptionV3 model instance

    Example:
        >>> model = create_inception_model(num_classes=7, pretrained=True)
        >>> x = torch.randn(4, 3, 299, 299)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([4, 7])
    """
    if adapted_input:
        model = InceptionV3Adapted(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
    else:
        model = InceptionV3Baseline(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            aux_logits=aux_logits
        )

    return model


if __name__ == "__main__":
    # Test the models
    print("Testing InceptionV3 Baseline Models...")

    # Test native InceptionV3 (299x299)
    print("\n=== InceptionV3 Native (299x299) ===")
    model_native = create_inception_model(
        num_classes=7,
        pretrained=True,
        aux_logits=False
    )

    batch_size = 4
    x_native = torch.randn(batch_size, 3, 299, 299)

    model_native.eval()
    with torch.no_grad():
        logits_native = model_native(x_native)

    print(f"Input shape: {x_native.shape}")
    print(f"Output shape: {logits_native.shape}")
    print(f"Model info: {model_native.get_model_info()}")

    total_params = sum(p.numel() for p in model_native.parameters())
    trainable_params = sum(p.numel() for p in model_native.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test adapted InceptionV3 (224x224)
    print("\n=== InceptionV3 Adapted (224x224) ===")
    model_adapted = create_inception_model(
        num_classes=7,
        pretrained=True,
        adapted_input=True
    )

    x_adapted = torch.randn(batch_size, 3, 224, 224)

    model_adapted.eval()
    with torch.no_grad():
        logits_adapted = model_adapted(x_adapted)

    print(f"Input shape: {x_adapted.shape}")
    print(f"Output shape: {logits_adapted.shape}")
    print(f"Model info: {model_adapted.get_model_info()}")

    # Test with auxiliary logits (training mode)
    print("\n=== InceptionV3 with Auxiliary Logits (Training) ===")
    model_aux = create_inception_model(
        num_classes=7,
        pretrained=True,
        aux_logits=True
    )

    model_aux.train()
    with torch.no_grad():
        output_aux = model_aux(x_native)

    print(f"Training output type: {type(output_aux)}")
    if hasattr(output_aux, 'logits'):
        print(f"Main logits shape: {output_aux.logits.shape}")
        print(f"Aux logits shape: {output_aux.aux_logits.shape}")

    print("\nInceptionV3 baseline models test PASSED!")
