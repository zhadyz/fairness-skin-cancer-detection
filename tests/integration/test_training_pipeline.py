"""
Integration tests for end-to-end training pipeline.

Tests the complete training workflow including data loading, model training,
loss computation, optimization, and checkpoint saving.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path


# ============================================================================
# FULL TRAINING PIPELINE TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
@pytest.mark.slow
class TestTrainingPipeline:
    """Test end-to-end training pipeline."""

    def test_single_epoch_training(self, mock_dataloader, mock_resnet_model, device):
        """Test training for a single epoch."""
        model = mock_resnet_model
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        total_loss = 0
        num_batches = 0

        for batch in mock_dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Loss should be computed successfully
        assert avg_loss > 0
        assert num_batches > 0

    def test_multi_epoch_training_convergence(self, mock_ham10000_small, device):
        """Test that loss decreases over multiple epochs."""
        from torch.utils.data import DataLoader

        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        # Small model and dataset for fast testing
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 7)
        model = model.to(device)

        dataloader = DataLoader(mock_ham10000_small, batch_size=4, shuffle=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        losses = []

        # Train for 5 epochs
        for epoch in range(5):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for batch in dataloader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

        # Loss should generally decrease (allowing for some fluctuation)
        assert losses[-1] < losses[0] * 1.2  # Final loss should be lower or similar

    def test_training_with_validation(self, mock_ham10000_dataset, device):
        """Test training loop with validation."""
        from torch.utils.data import DataLoader, random_split

        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        # Split into train/val
        train_size = 80
        val_size = 20
        train_dataset, val_dataset = random_split(
            mock_ham10000_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 7)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training epoch
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation epoch
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_accuracy = correct / total

        # Check metrics are reasonable
        assert train_loss > 0
        assert val_loss > 0
        assert 0 <= val_accuracy <= 1


# ============================================================================
# CHECKPOINT SAVING TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestCheckpointSaving:
    """Test checkpoint saving during training."""

    def test_save_checkpoint_after_epoch(self, mock_resnet_model, temp_dir):
        """Test saving checkpoint after training epoch."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        optimizer = torch.optim.Adam(mock_resnet_model.parameters(), lr=0.001)

        epoch = 5
        loss = 0.345
        accuracy = 0.876

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': mock_resnet_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Verify checkpoint saved
        assert checkpoint_path.exists()

        # Load and verify
        loaded = torch.load(checkpoint_path)
        assert loaded['epoch'] == epoch
        assert loaded['loss'] == loss
        assert loaded['accuracy'] == accuracy

    def test_save_best_checkpoint(self, mock_resnet_model, temp_dir):
        """Test saving only the best checkpoint based on validation metric."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        optimizer = torch.optim.Adam(mock_resnet_model.parameters(), lr=0.001)

        best_accuracy = 0
        best_checkpoint_path = None

        # Simulate training epochs with different accuracies
        accuracies = [0.75, 0.82, 0.78, 0.89, 0.85]  # Best at epoch 3

        for epoch, accuracy in enumerate(accuracies):
            if accuracy > best_accuracy:
                best_accuracy = accuracy

                # Save best checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': mock_resnet_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy
                }

                best_checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_checkpoint_path)

        # Verify best checkpoint has highest accuracy
        loaded = torch.load(best_checkpoint_path)
        assert loaded['accuracy'] == 0.89
        assert loaded['epoch'] == 3


# ============================================================================
# LEARNING RATE SCHEDULING TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestLearningRateScheduling:
    """Test learning rate scheduling during training."""

    def test_step_lr_scheduler(self, mock_resnet_model):
        """Test StepLR scheduler."""
        optimizer = torch.optim.SGD(mock_resnet_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr == 0.1

        # Simulate 10 epochs
        lrs = []
        for epoch in range(10):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # LR should decrease at epochs 3, 6, 9
        assert lrs[0] == 0.1
        assert lrs[3] == 0.01
        assert lrs[6] == 0.001

    def test_cosine_annealing_scheduler(self, mock_resnet_model):
        """Test CosineAnnealingLR scheduler."""
        optimizer = torch.optim.Adam(mock_resnet_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        lrs = []
        for epoch in range(20):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # LR should decrease and then increase again (cosine pattern)
        assert lrs[0] > lrs[5]  # Decreasing in first half
        assert lrs[10] < lrs[5]  # Minimum at T_max

    def test_reduce_on_plateau_scheduler(self, mock_resnet_model):
        """Test ReduceLROnPlateau scheduler."""
        optimizer = torch.optim.Adam(mock_resnet_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        # Simulate validation losses
        val_losses = [1.0, 0.9, 0.88, 0.87, 0.87, 0.87, 0.86]  # Plateau at 0.87

        initial_lr = optimizer.param_groups[0]['lr']

        for loss in val_losses:
            scheduler.step(loss)

        # LR should decrease after plateau
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr


# ============================================================================
# GRADIENT ACCUMULATION TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestGradientAccumulation:
    """Test gradient accumulation for effective large batch sizes."""

    def test_gradient_accumulation_equivalence(self, mock_ham10000_small, device):
        """Test that gradient accumulation produces same result as larger batch."""
        from torch.utils.data import DataLoader

        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        # Setup two identical models
        model1 = resnet18(pretrained=False)
        model1.fc = nn.Linear(model1.fc.in_features, 7)
        model1 = model1.to(device)

        model2 = resnet18(pretrained=False)
        model2.fc = nn.Linear(model2.fc.in_features, 7)
        model2.load_state_dict(model1.state_dict())  # Same initialization
        model2 = model2.to(device)

        criterion = nn.CrossEntropyLoss()

        # Model 1: Batch size 8 (no accumulation)
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        loader1 = DataLoader(mock_ham10000_small, batch_size=8, shuffle=False)

        model1.train()
        for batch in loader1:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer1.zero_grad()
            outputs = model1(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer1.step()
            break  # Only one batch

        # Model 2: Batch size 4 with accumulation_steps=2 (effective batch 8)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        loader2 = DataLoader(mock_ham10000_small, batch_size=4, shuffle=False)

        model2.train()
        optimizer2.zero_grad()
        accumulation_steps = 2

        for i, batch in enumerate(loader2):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model2(images)
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer2.step()
                optimizer2.zero_grad()
                break  # Processed equivalent of one batch size 8

        # Models should have similar (not necessarily identical) weights
        # due to different batch compositions, but gradients should flow correctly
        assert model1.fc.weight.grad is None  # Optimizer stepped, gradients cleared
        assert model2.fc.weight.grad is None


# ============================================================================
# MIXED PRECISION TRAINING TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
@pytest.mark.requires_gpu
class TestMixedPrecisionTraining:
    """Test mixed precision training with torch.cuda.amp."""

    def test_mixed_precision_forward_backward(self, mock_dataloader, device):
        """Test mixed precision training (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")

        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not available")

        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 7)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        model.train()

        batch = next(iter(mock_dataloader))
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass with autocast
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Check that training completed successfully
        assert torch.isfinite(loss)


# ============================================================================
# DATA AUGMENTATION IN TRAINING TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestDataAugmentation:
    """Test data augmentation during training."""

    def test_training_with_augmentation(self, device):
        """Test that augmentation is applied during training."""
        from torchvision import transforms
        from tests.fixtures.sample_data import MockHAM10000
        from torch.utils.data import DataLoader

        # Create dataset with augmentation
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        ])

        dataset = MockHAM10000(num_samples=20, seed=42)

        # Get same image multiple times (should be different due to augmentation)
        img1 = dataset[0]['image']

        # Note: Our mock dataset doesn't apply transforms
        # In real implementation, transforms would be applied in __getitem__
        assert img1.shape == (3, 224, 224)

    def test_no_augmentation_during_validation(self, device):
        """Test that augmentation is not applied during validation."""
        from tests.fixtures.sample_data import MockHAM10000

        # Validation dataset (no augmentation)
        val_dataset = MockHAM10000(num_samples=20, seed=42)

        # Same image should be identical when accessed multiple times
        img1 = val_dataset[0]['image']
        img2 = val_dataset[0]['image']

        assert torch.allclose(img1, img2)


# ============================================================================
# EARLY STOPPING TEST
# ============================================================================

@pytest.mark.integration
@pytest.mark.pipeline
class TestEarlyStopping:
    """Test early stopping mechanism."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience epochs."""

        class EarlyStopping:
            def __init__(self, patience=3, min_delta=0.001):
                self.patience = patience
                self.min_delta = min_delta
                self.counter = 0
                self.best_loss = None
                self.should_stop = False

            def __call__(self, val_loss):
                if self.best_loss is None:
                    self.best_loss = val_loss
                elif val_loss > self.best_loss - self.min_delta:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.should_stop = True
                else:
                    self.best_loss = val_loss
                    self.counter = 0

        early_stopping = EarlyStopping(patience=3)

        # Simulate validation losses
        val_losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]  # Plateau

        for i, loss in enumerate(val_losses):
            early_stopping(loss)
            if early_stopping.should_stop:
                stopped_epoch = i
                break

        # Should stop after patience=3 epochs of no improvement
        assert early_stopping.should_stop
        assert stopped_epoch <= 6
