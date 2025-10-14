"""
Unit tests for model architecture and initialization.

Tests ResNet50, EfficientNet, and other architectures for proper initialization,
forward pass, gradient flow, and checkpoint loading.
"""

import pytest
import torch
import torch.nn as nn


# ============================================================================
# MODEL INITIALIZATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.models
class TestModelInitialization:
    """Test model initialization and architecture setup."""

    def test_resnet50_initialization(self, device):
        """Test ResNet50 model loads correctly."""
        try:
            from torchvision.models import resnet50
        except ImportError:
            pytest.skip("torchvision not available")

        model = resnet50(pretrained=False)
        model = model.to(device)

        # Check model is in proper state
        assert isinstance(model, nn.Module)
        assert next(model.parameters()).device.type == device.type

    def test_resnet50_num_classes(self, device):
        """Test ResNet50 with custom number of output classes."""
        try:
            from torchvision.models import resnet50
        except ImportError:
            pytest.skip("torchvision not available")

        num_classes = 7
        model = resnet50(pretrained=False)

        # Replace final layer for 7 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        # Check output layer
        assert model.fc.out_features == num_classes

    def test_resnet18_initialization(self, device):
        """Test ResNet18 (smaller model for testing)."""
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 7)
        model = model.to(device)

        assert model.fc.out_features == 7

    def test_model_parameters_trainable(self, mock_resnet_model):
        """Test that model parameters are trainable."""
        trainable_params = sum(p.numel() for p in mock_resnet_model.parameters() if p.requires_grad)
        assert trainable_params > 0

    def test_model_freeze_backbone(self, device):
        """Test freezing backbone layers."""
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 7)

        # Freeze all layers except final FC
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        # Check only FC is trainable
        trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
        assert all('fc' in name for name in trainable_params)


# ============================================================================
# FORWARD PASS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.models
class TestForwardPass:
    """Test model forward pass."""

    def test_forward_pass_single_image(self, mock_resnet_model, device):
        """Test forward pass with single image."""
        batch_size = 1
        x = torch.randn(batch_size, 3, 224, 224).to(device)

        with torch.no_grad():
            output = mock_resnet_model(x)

        assert output.shape == (batch_size, 7)

    def test_forward_pass_batch(self, mock_resnet_model, device):
        """Test forward pass with batch of images."""
        batch_size = 16
        x = torch.randn(batch_size, 3, 224, 224).to(device)

        with torch.no_grad():
            output = mock_resnet_model(x)

        assert output.shape == (batch_size, 7)

    def test_forward_pass_output_range(self, mock_resnet_model, device):
        """Test that forward pass produces logits (not probabilities)."""
        x = torch.randn(8, 3, 224, 224).to(device)

        with torch.no_grad():
            logits = mock_resnet_model(x)

        # Logits can be any real number (not constrained to [0, 1])
        assert not torch.all((logits >= 0) & (logits <= 1))

    def test_forward_pass_with_softmax(self, mock_resnet_model, device):
        """Test applying softmax to get probabilities."""
        x = torch.randn(8, 3, 224, 224).to(device)

        with torch.no_grad():
            logits = mock_resnet_model(x)
            probs = torch.softmax(logits, dim=1)

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(8).to(device), atol=1e-6)

        # All probabilities should be in [0, 1]
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_forward_pass_deterministic(self, mock_resnet_model, device, random_seed):
        """Test that forward pass is deterministic in eval mode."""
        torch.manual_seed(random_seed)
        x = torch.randn(4, 3, 224, 224).to(device)

        mock_resnet_model.eval()

        with torch.no_grad():
            output1 = mock_resnet_model(x)
            output2 = mock_resnet_model(x)

        assert torch.allclose(output1, output2)


# ============================================================================
# BACKWARD PASS / GRADIENT TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.models
class TestBackwardPass:
    """Test gradient computation and backpropagation."""

    def test_backward_pass_computes_gradients(self, mock_resnet_model, device):
        """Test that backward pass computes gradients."""
        x = torch.randn(4, 3, 224, 224).to(device)
        labels = torch.randint(0, 7, (4,)).to(device)

        mock_resnet_model.train()

        # Forward pass
        logits = mock_resnet_model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in mock_resnet_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_flow_to_all_layers(self, mock_resnet_model, device):
        """Test that gradients flow to all trainable layers."""
        x = torch.randn(2, 3, 224, 224).to(device)
        labels = torch.randint(0, 7, (2,)).to(device)

        mock_resnet_model.train()

        # Forward + backward
        logits = mock_resnet_model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Check all trainable params have non-zero gradients
        for name, param in mock_resnet_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                # Most gradients should be non-zero
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_gradient_accumulation(self, mock_resnet_model, device):
        """Test gradient accumulation over multiple batches."""
        mock_resnet_model.train()

        # Accumulate gradients over 3 mini-batches
        total_loss = 0
        for _ in range(3):
            x = torch.randn(2, 3, 224, 224).to(device)
            labels = torch.randint(0, 7, (2,)).to(device)

            logits = mock_resnet_model(x)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()  # Accumulate gradients
            total_loss += loss.item()

        # Gradients should be accumulated
        for param in mock_resnet_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gradient_clipping(self, mock_resnet_model, device):
        """Test gradient clipping."""
        x = torch.randn(4, 3, 224, 224).to(device)
        labels = torch.randint(0, 7, (4,)).to(device)

        mock_resnet_model.train()

        logits = mock_resnet_model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(mock_resnet_model.parameters(), max_norm)

        # Check gradient norms
        total_norm = 0
        for param in mock_resnet_model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm + 1e-6  # Small tolerance


# ============================================================================
# MODEL SAVE/LOAD TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.models
class TestModelCheckpoint:
    """Test model checkpoint saving and loading."""

    def test_save_model_state_dict(self, mock_resnet_model, temp_dir):
        """Test saving model state dict."""
        save_path = temp_dir / "model_state.pth"

        torch.save(mock_resnet_model.state_dict(), save_path)

        assert save_path.exists()

    def test_load_model_state_dict(self, mock_resnet_model, temp_dir, device):
        """Test loading model state dict."""
        save_path = temp_dir / "model_state.pth"

        # Save
        torch.save(mock_resnet_model.state_dict(), save_path)

        # Create new model and load
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        new_model = resnet18(pretrained=False)
        new_model.fc = nn.Linear(new_model.fc.in_features, 7)
        new_model = new_model.to(device)

        new_model.load_state_dict(torch.load(save_path))

        # Check weights match
        for (name1, param1), (name2, param2) in zip(
            mock_resnet_model.named_parameters(),
            new_model.named_parameters()
        ):
            assert torch.allclose(param1, param2)

    def test_save_full_checkpoint(self, mock_resnet_model, temp_dir):
        """Test saving full checkpoint with metadata."""
        save_path = temp_dir / "checkpoint.pth"

        checkpoint = {
            'epoch': 10,
            'model_state_dict': mock_resnet_model.state_dict(),
            'optimizer_state_dict': {},
            'loss': 0.234,
            'accuracy': 0.89
        }

        torch.save(checkpoint, save_path)

        # Load and verify
        loaded = torch.load(save_path)
        assert loaded['epoch'] == 10
        assert loaded['loss'] == 0.234
        assert loaded['accuracy'] == 0.89

    def test_load_pretrained_weights(self, mock_trained_model_weights, device):
        """Test loading pretrained weights."""
        checkpoint = torch.load(mock_trained_model_weights)

        assert 'model_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert checkpoint['epoch'] == 10


# ============================================================================
# MODEL ARCHITECTURE TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.models
class TestModelArchitecture:
    """Test model architecture properties."""

    def test_model_layer_count(self, mock_resnet_model):
        """Test model has expected number of layers."""
        num_layers = sum(1 for _ in mock_resnet_model.modules())
        assert num_layers > 10  # ResNet should have many layers

    def test_model_parameter_count(self, mock_resnet_model):
        """Test model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in mock_resnet_model.parameters())
        assert total_params > 1e6  # ResNet should have millions of parameters

    def test_model_input_shape(self, mock_resnet_model, device):
        """Test model accepts correct input shape."""
        # Valid input
        x_valid = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = mock_resnet_model(x_valid)
        assert output.shape[0] == 1

    def test_model_eval_mode(self, mock_resnet_model):
        """Test model can be set to eval mode."""
        mock_resnet_model.eval()
        assert not mock_resnet_model.training

    def test_model_train_mode(self, mock_resnet_model):
        """Test model can be set to train mode."""
        mock_resnet_model.train()
        assert mock_resnet_model.training


# ============================================================================
# LOSS FUNCTION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.models
class TestLossFunctions:
    """Test loss functions for training."""

    def test_cross_entropy_loss(self):
        """Test CrossEntropyLoss computation."""
        logits = torch.randn(8, 7)
        labels = torch.randint(0, 7, (8,))

        loss = nn.CrossEntropyLoss()(logits, labels)

        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_weighted_cross_entropy(self):
        """Test weighted CrossEntropyLoss for imbalanced classes."""
        logits = torch.randn(8, 7)
        labels = torch.randint(0, 7, (8,))

        # Class weights (inverse frequency)
        weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 4.0])
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)

        assert torch.isfinite(loss)

    def test_focal_loss_concept(self):
        """Test focal loss concept (focusing on hard examples)."""
        # Easy examples (high confidence correct predictions)
        easy_logits = torch.tensor([[10.0, -10.0, -10.0]])
        easy_labels = torch.tensor([0])

        # Hard examples (low confidence)
        hard_logits = torch.tensor([[0.5, 0.4, 0.3]])
        hard_labels = torch.tensor([0])

        ce_loss = nn.CrossEntropyLoss(reduction='none')

        easy_loss = ce_loss(easy_logits, easy_labels)
        hard_loss = ce_loss(hard_logits, hard_labels)

        # Hard examples should have higher loss
        assert hard_loss > easy_loss


# ============================================================================
# OPTIMIZER TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.models
class TestOptimizers:
    """Test optimizer configurations."""

    def test_adam_optimizer(self, mock_resnet_model):
        """Test Adam optimizer initialization."""
        optimizer = torch.optim.Adam(mock_resnet_model.parameters(), lr=0.001)

        assert len(optimizer.param_groups) > 0

    def test_sgd_optimizer(self, mock_resnet_model):
        """Test SGD optimizer with momentum."""
        optimizer = torch.optim.SGD(
            mock_resnet_model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4
        )

        assert optimizer.defaults['momentum'] == 0.9

    def test_optimizer_step(self, mock_resnet_model, device):
        """Test optimizer step updates weights."""
        x = torch.randn(2, 3, 224, 224).to(device)
        labels = torch.randint(0, 7, (2,)).to(device)

        optimizer = torch.optim.Adam(mock_resnet_model.parameters(), lr=0.001)

        # Store initial weights
        initial_weights = {name: param.clone() for name, param in mock_resnet_model.named_parameters()}

        # Training step
        mock_resnet_model.train()
        optimizer.zero_grad()
        logits = mock_resnet_model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        # Check weights changed
        for name, param in mock_resnet_model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(param, initial_weights[name])

    def test_learning_rate_scheduler(self, mock_resnet_model):
        """Test learning rate scheduler."""
        optimizer = torch.optim.Adam(mock_resnet_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']

        # Step scheduler 5 times
        for _ in range(5):
            scheduler.step()

        new_lr = optimizer.param_groups[0]['lr']

        # LR should have decreased
        assert new_lr < initial_lr
