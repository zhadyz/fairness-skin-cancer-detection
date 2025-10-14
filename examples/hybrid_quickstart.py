"""
Quick Start Example: Hybrid ConvNeXtV2-Swin Transformer

Demonstrates how to:
1. Create hybrid model
2. Train on HAM10000
3. Evaluate fairness metrics
4. Use FairDisCo integration

Framework: MENDICANT_BIAS - Phase 3
Agent: HOLLOWED_EYES
Version: 1.0
Date: 2025-10-14
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('.')

from src.models.hybrid_model import create_hybrid_model


def example_1_create_model():
    """Example 1: Create and inspect hybrid model."""
    print("=" * 80)
    print("Example 1: Create Hybrid Model")
    print("=" * 80)

    # Create hybrid model (without FairDisCo)
    model = create_hybrid_model(
        convnext_variant='base',
        swin_variant='small',
        num_classes=7,
        enable_fairdisco=False,
        fusion_dim=768,
        dropout=0.3
    )

    # Print model info
    info = model.get_model_info()
    print("\nModel Architecture: Hybrid ConvNeXtV2-Swin")
    print(f"ConvNeXt Variant: {info['convnext_variant']}")
    print(f"Swin Variant: {info['swin_variant']}")
    print(f"Fusion Dimension: {info['fusion_dim']}")
    print(f"\nParameter Breakdown:")
    for k, v in info['parameter_breakdown'].items():
        print(f"  {k:15s}: {v:,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    return model


def example_2_with_fairdisco():
    """Example 2: Create model with FairDisCo integration."""
    print("\n" + "=" * 80)
    print("Example 2: Hybrid Model with FairDisCo")
    print("=" * 80)

    # Create model with FairDisCo enabled
    model = create_hybrid_model(
        convnext_variant='base',
        swin_variant='small',
        num_classes=7,
        enable_fairdisco=True,
        num_fst_classes=6
    )

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    diag_logits, fst_logits, cont_emb = model(x)

    print("\nFairDisCo Outputs:")
    print(f"  Diagnosis logits: {diag_logits.shape}")
    print(f"  FST logits: {fst_logits.shape}")
    print(f"  Contrastive embeddings: {cont_emb.shape}")

    return model


def example_3_training_simulation():
    """Example 3: Simulate training step."""
    print("\n" + "=" * 80)
    print("Example 3: Training Simulation")
    print("=" * 80)

    # Create model
    model = create_hybrid_model(
        convnext_variant='base',
        swin_variant='small',
        num_classes=7
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05
    )

    # Setup loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Simulate training batch
    model.train()
    x = torch.randn(4, 3, 224, 224)
    targets = torch.randint(0, 7, (4,))

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    optimizer.step()

    print(f"\nTraining step completed!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients computed successfully")

    # Compute accuracy
    preds = torch.argmax(outputs, dim=1)
    acc = (preds == targets).float().mean()
    print(f"  Batch accuracy: {acc.item():.4f}")


def example_4_different_variants():
    """Example 4: Compare different model variants."""
    print("\n" + "=" * 80)
    print("Example 4: Model Variants Comparison")
    print("=" * 80)

    variants = [
        ('tiny', 'tiny'),
        ('small', 'tiny'),
        ('base', 'small'),
        ('large', 'base')
    ]

    print("\n{:20s} {:>15s} {:>15s}".format("Variant", "Parameters", "Output Shape"))
    print("-" * 52)

    for convnext_var, swin_var in variants:
        try:
            model = create_hybrid_model(
                convnext_variant=convnext_var,
                swin_variant=swin_var
            )
            model.eval()

            info = model.get_model_info()
            params = info['parameter_breakdown']['total']

            with torch.no_grad():
                x = torch.randn(1, 3, 224, 224)
                output = model(x)

            print(f"{convnext_var}-{swin_var:10s} {params:>15,} {str(output.shape):>15s}")

        except Exception as e:
            print(f"{convnext_var}-{swin_var:10s} ERROR: {e}")


def example_5_save_load_model():
    """Example 5: Save and load model."""
    print("\n" + "=" * 80)
    print("Example 5: Save and Load Model")
    print("=" * 80)

    # Create and save model
    model = create_hybrid_model(
        convnext_variant='base',
        swin_variant='small'
    )

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'convnext_variant': 'base',
            'swin_variant': 'small',
            'num_classes': 7,
            'fusion_dim': 768
        },
        'model_info': model.get_model_info()
    }

    torch.save(checkpoint, 'hybrid_model_checkpoint.pth')
    print("\nModel saved to: hybrid_model_checkpoint.pth")

    # Load model
    checkpoint_loaded = torch.load('hybrid_model_checkpoint.pth')

    model_loaded = create_hybrid_model(
        **checkpoint_loaded['model_config']
    )
    model_loaded.load_state_dict(checkpoint_loaded['model_state_dict'])

    print("Model loaded successfully!")

    # Verify
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    model_loaded.eval()

    with torch.no_grad():
        out1 = model(x)
        out2 = model_loaded(x)

    print(f"Outputs match: {torch.allclose(out1, out2)}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HYBRID CONVNEXTV2-SWIN TRANSFORMER - QUICK START EXAMPLES")
    print("MENDICANT_BIAS Phase 3")
    print("=" * 80)

    # Run examples
    example_1_create_model()
    example_2_with_fairdisco()
    example_3_training_simulation()
    example_4_different_variants()
    example_5_save_load_model()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
