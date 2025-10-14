"""
EfficientNet B4 Baseline Model for Skin Cancer Detection

Transfer learning using EfficientNet B4 architecture optimized for medical image classification.
EfficientNet provides better accuracy-efficiency tradeoffs compared to ResNet.

Author: HOLLOWED_EYES
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any


class EfficientNetB4Baseline(nn.Module):
    """
    EfficientNet B4 baseline model with transfer learning from ImageNet.

    Architecture:
    - Pre-trained EfficientNet B4 backbone (timm library)
    - Global Average Pooling
    - Dropout (0.4 - EfficientNet default)
    - Fully connected layer (1792 -> num_classes)

    EfficientNet B4 specifications:
    - Input size: 380x380 (can use 224x224 with adaptation)
    - Parameters: ~19M
    - Top-1 Accuracy on ImageNet: 83.4%

    Args:
        num_classes: Number of output classes (7 for HAM10000, 8 for ISIC)
        pretrained: Load ImageNet pre-trained weights
        freeze_backbone: Freeze backbone layers (fine-tune only classifier)
        dropout: Dropout rate before final classifier
        img_size: Input image size (default 224, native 380)
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
        img_size: int = 224
    ):
        super(EfficientNetB4Baseline, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.img_size = img_size

        # Load pre-trained EfficientNet B4 from timm
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'  # Global average pooling
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimension
        # EfficientNet B4 outputs 1792 features after global pooling
        in_features = self.backbone.num_features

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        # Store model info
        self.model_info = {
            'architecture': 'efficientnet_b4',
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone,
            'dropout': dropout,
            'img_size': img_size,
            'feature_dim': in_features
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W) - typically (B, 3, 224, 224) or (B, 3, 380, 380)

        Returns:
            Logits tensor (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def unfreeze_backbone(self, num_blocks: Optional[int] = None):
        """
        Unfreeze backbone blocks for fine-tuning.

        EfficientNet B4 has 7 blocks (blocks[0] to blocks[6]).

        Args:
            num_blocks: Number of blocks to unfreeze from the end.
                       If None, unfreezes all blocks.
        """
        if num_blocks is None:
            # Unfreeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last num_blocks
            blocks = self.backbone.blocks
            num_blocks_to_unfreeze = min(num_blocks, len(blocks))

            for i in range(len(blocks) - num_blocks_to_unfreeze, len(blocks)):
                for param in blocks[i].parameters():
                    param.requires_grad = True

    def get_feature_extractor(self) -> nn.Module:
        """
        Returns the model without the classification head (for feature extraction).

        Returns:
            Feature extractor module (outputs 1792-dim features for B4)
        """
        return self.backbone

    def get_model_info(self) -> Dict[str, Any]:
        """Returns model configuration info."""
        return self.model_info


class EfficientNetB3Baseline(nn.Module):
    """
    EfficientNet B3 baseline model (lighter alternative to B4).

    Specifications:
    - Input size: 300x300 (can use 224x224)
    - Parameters: ~12M
    - Top-1 Accuracy: 82.1%
    - Faster training and inference
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        img_size: int = 224
    ):
        super(EfficientNetB3Baseline, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.img_size = img_size

        # Load EfficientNet B3
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.num_features  # 1536 for B3

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

        self.model_info = {
            'architecture': 'efficientnet_b3',
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone,
            'dropout': dropout,
            'img_size': img_size,
            'feature_dim': in_features
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def unfreeze_backbone(self, num_blocks: Optional[int] = None):
        if num_blocks is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            blocks = self.backbone.blocks
            num_blocks_to_unfreeze = min(num_blocks, len(blocks))
            for i in range(len(blocks) - num_blocks_to_unfreeze, len(blocks)):
                for param in blocks[i].parameters():
                    param.requires_grad = True

    def get_feature_extractor(self) -> nn.Module:
        return self.backbone

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info


def create_efficientnet_model(
    variant: str = 'b4',
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: Optional[float] = None,
    img_size: int = 224
) -> nn.Module:
    """
    Factory function to create EfficientNet baseline model.

    Args:
        variant: EfficientNet variant ('b3' or 'b4')
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Freeze backbone during initial training
        dropout: Dropout rate (None = use default for variant)
        img_size: Input image size

    Returns:
        EfficientNet model instance

    Example:
        >>> model = create_efficientnet_model('b4', num_classes=7)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([4, 7])
    """
    variant = variant.lower()

    if variant == 'b4':
        dropout = dropout if dropout is not None else 0.4
        model = EfficientNetB4Baseline(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            img_size=img_size
        )
    elif variant == 'b3':
        dropout = dropout if dropout is not None else 0.3
        model = EfficientNetB3Baseline(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
            img_size=img_size
        )
    else:
        raise ValueError(f"Unsupported EfficientNet variant: {variant}. Use 'b3' or 'b4'.")

    return model


if __name__ == "__main__":
    # Test the models
    print("Testing EfficientNet Baseline Models...")

    # Test EfficientNet B4
    print("\n=== EfficientNet B4 ===")
    model_b4 = create_efficientnet_model('b4', num_classes=7, pretrained=True)

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        logits_b4 = model_b4(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits_b4.shape}")
    print(f"Model info: {model_b4.get_model_info()}")

    total_params_b4 = sum(p.numel() for p in model_b4.parameters())
    trainable_params_b4 = sum(p.numel() for p in model_b4.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params_b4:,}")
    print(f"Trainable parameters: {trainable_params_b4:,}")

    # Test EfficientNet B3
    print("\n=== EfficientNet B3 ===")
    model_b3 = create_efficientnet_model('b3', num_classes=7, pretrained=True)

    with torch.no_grad():
        logits_b3 = model_b3(x)

    print(f"Output shape: {logits_b3.shape}")
    print(f"Model info: {model_b3.get_model_info()}")

    total_params_b3 = sum(p.numel() for p in model_b3.parameters())
    trainable_params_b3 = sum(p.numel() for p in model_b3.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params_b3:,}")
    print(f"Trainable parameters: {trainable_params_b3:,}")

    print("\nEfficientNet baseline models test PASSED!")
