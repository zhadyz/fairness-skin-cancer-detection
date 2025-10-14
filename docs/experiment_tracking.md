# Experiment Tracking and Management

## Overview

This project uses **TensorBoard** for experiment tracking and visualization, with optional **Weights & Biases (W&B)** integration for advanced features.

## TensorBoard Setup

### Directory Structure

```
experiments/
├── runs/                    # TensorBoard logs
│   ├── baseline/            # Baseline model experiments
│   ├── fairness/            # Fairness-enhanced models
│   └── ablation/            # Ablation studies
├── checkpoints/             # Model checkpoints
│   ├── resnet50_best.pth
│   ├── convnext_best.pth
│   └── swin_best.pth
└── configs/                 # Experiment configurations
    ├── baseline_config.yaml
    ├── fairness_config.yaml
    └── sweep_config.yaml
```

### Starting TensorBoard

**Launch TensorBoard server**:
```bash
tensorboard --logdir experiments/runs --port 6006
```

**Access in browser**: http://localhost:6006

**For remote servers** (SSH tunneling):
```bash
# On local machine
ssh -L 6006:localhost:6006 user@remote-server

# On remote server
tensorboard --logdir experiments/runs --port 6006
```

### Logging During Training

TensorBoard logging is integrated in `src/training/trainer.py`:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='experiments/runs/resnet50_baseline')

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (images, labels, skin_types) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Log scalar metrics
        writer.add_scalar('Loss/train', loss.item(), global_step)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    val_loss, val_acc = validate(model, val_loader)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    # Log learning rate
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

writer.close()
```

## Tracked Metrics

### Training Metrics

- **Loss**: Cross-entropy, focal loss, or custom fairness-aware loss
- **Accuracy**: Overall, per-class, per-skin-type
- **Learning Rate**: Current optimizer learning rate
- **Gradient Norms**: Monitor gradient flow
- **Batch Processing Time**: Data loading and training speed

### Validation Metrics

- **Overall Accuracy**: Total validation accuracy
- **Balanced Accuracy**: Macro-averaged accuracy across classes
- **Per-Class Metrics**: Precision, recall, F1-score for each disease class
- **Per-FST Metrics**: Performance stratified by Fitzpatrick skin type
- **Confusion Matrix**: Visual representation of predictions

### Fairness Metrics

- **Demographic Parity Difference**: Gap in positive prediction rates across FST groups
- **Equalized Odds Difference**: TPR and FPR disparities
- **Equal Opportunity Difference**: Sensitivity disparity across groups
- **Disparate Impact**: Ratio of selection rates
- **Calibration**: Prediction confidence vs accuracy across FST groups

### Model Complexity

- **Parameter Count**: Total trainable parameters
- **FLOPs**: Floating point operations
- **Inference Time**: Per-sample prediction latency
- **Model Size**: Checkpoint file size

## Experiment Configuration with Hydra

### Configuration Files

Create experiment configs in `experiments/configs/`:

**Example: `baseline_config.yaml`**
```yaml
model:
  architecture: resnet50
  pretrained: true
  num_classes: 7
  dropout: 0.3

data:
  dataset: ham10000
  batch_size: 32
  num_workers: 4
  augmentation:
    horizontal_flip: true
    rotation: 15
    color_jitter: 0.2

training:
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  epochs: 100
  scheduler: cosine
  early_stopping:
    patience: 10
    min_delta: 0.001

loss:
  type: cross_entropy
  label_smoothing: 0.1

fairness:
  enabled: false

logging:
  tensorboard_dir: experiments/runs/baseline
  checkpoint_dir: experiments/checkpoints
  save_frequency: 5
```

### Running Experiments with Hydra

```bash
python experiments/baseline/train_resnet50.py \
    --config-name baseline_config \
    model.architecture=resnet50 \
    training.epochs=50 \
    data.batch_size=64
```

**Override multiple parameters**:
```bash
python experiments/baseline/train_resnet50.py \
    model.architecture=convnext_base \
    training.learning_rate=0.0001 \
    fairness.enabled=true \
    fairness.method=reweighting
```

## Hyperparameter Sweeps

### TensorBoard HParams

Track hyperparameter experiments:

```python
from torch.utils.tensorboard import SummaryWriter

hparams = {
    'lr': 0.001,
    'batch_size': 32,
    'model': 'resnet50',
    'optimizer': 'adamw'
}

metrics = {
    'accuracy': val_acc,
    'loss': val_loss,
    'f1_score': f1
}

writer.add_hparams(hparams, metrics)
```

**View in TensorBoard**: Navigate to HPARAMS tab for parallel coordinates visualization.

### Weights & Biases Sweeps (Optional)

**Create sweep config** (`experiments/configs/sweep_config.yaml`):
```yaml
program: experiments/baseline/train_resnet50.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.01
  batch_size:
    values: [16, 32, 64]
  dropout:
    distribution: uniform
    min: 0.2
    max: 0.5
  weight_decay:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
```

**Initialize sweep**:
```bash
wandb sweep experiments/configs/sweep_config.yaml
```

**Run sweep agents**:
```bash
wandb agent <sweep-id>
```

## Visualization Examples

### Training Progress

```python
# Loss curves
writer.add_scalars('Loss', {
    'train': train_loss,
    'val': val_loss
}, epoch)

# Accuracy by skin type
for fst in range(1, 7):
    acc = compute_accuracy_for_fst(model, val_loader, fst)
    writer.add_scalar(f'Accuracy/FST_{fst}', acc, epoch)
```

### Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
writer.add_figure('ConfusionMatrix', fig, epoch)
```

### Feature Embeddings

```python
# t-SNE visualization of learned features
from sklearn.manifold import TSNE

features, labels, skin_types = extract_features(model, val_loader)
embeddings = TSNE(n_components=2).fit_transform(features)

writer.add_embedding(
    embeddings,
    metadata=labels,
    metadata_header=['class'],
    tag='feature_embeddings'
)
```

### Sample Predictions

```python
# Log prediction examples
images, labels, predictions = get_sample_predictions(model, val_loader)

# Add images with labels
writer.add_images('Predictions', images, global_step=epoch)
```

## Checkpoint Management

### Saving Checkpoints

```python
def save_checkpoint(model, optimizer, epoch, val_acc, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'config': config
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved: {path}')

# Save best model
if val_acc > best_acc:
    best_acc = val_acc
    save_checkpoint(
        model, optimizer, epoch, val_acc,
        'experiments/checkpoints/resnet50_best.pth'
    )

# Save periodic checkpoints
if epoch % 10 == 0:
    save_checkpoint(
        model, optimizer, epoch, val_acc,
        f'experiments/checkpoints/resnet50_epoch_{epoch}.pth'
    )
```

### Loading Checkpoints

```python
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_accuracy']
    return epoch, val_acc

# Resume training
if resume_from_checkpoint:
    epoch, best_acc = load_checkpoint(
        model, optimizer,
        'experiments/checkpoints/resnet50_best.pth'
    )
    print(f'Resumed from epoch {epoch} with val_acc={best_acc:.4f}')
```

## Experiment Comparison

### Compare Multiple Runs

```bash
# Launch TensorBoard with multiple runs
tensorboard --logdir_spec \
    baseline:experiments/runs/baseline, \
    fairness:experiments/runs/fairness, \
    swin:experiments/runs/swin_transformer
```

### Export Metrics

```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('experiments/runs/baseline')
ea.Reload()

# Extract scalars
val_accuracy = ea.Scalars('Accuracy/val')
df = pd.DataFrame(val_accuracy)
df.to_csv('experiments/results/baseline_metrics.csv', index=False)
```

## Weights & Biases (Optional Advanced Features)

### Setup

```bash
# Login
wandb login

# Initialize in training script
import wandb

wandb.init(
    project='skin-cancer-classification',
    config={
        'architecture': 'resnet50',
        'learning_rate': 0.001,
        'epochs': 100
    }
)
```

### Advanced Features

**Artifact tracking**:
```python
# Log dataset as artifact
artifact = wandb.Artifact('ham10000', type='dataset')
artifact.add_dir('data/processed')
wandb.log_artifact(artifact)

# Log model
artifact = wandb.Artifact('resnet50_model', type='model')
artifact.add_file('experiments/checkpoints/resnet50_best.pth')
wandb.log_artifact(artifact)
```

**Custom plots**:
```python
# Log custom fairness metrics
wandb.log({
    'demographic_parity': dp_diff,
    'equalized_odds': eo_diff,
    'calibration_error': cal_error
})

# Log confusion matrix
wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_true,
    preds=y_pred,
    class_names=class_names
)})
```

**Report generation**:
- Create rich reports with visualizations
- Share experiment results with collaborators
- Compare model performance across runs

## Best Practices

### Experiment Naming Convention

Use descriptive names:
```
<model>_<dataset>_<config>_<date>
```

Examples:
- `resnet50_ham10000_baseline_20251013`
- `convnext_isic2019_fairness_reweighting_20251013`
- `swin_multi_ablation_noaugment_20251013`

### Reproducibility

**Always log**:
- Random seed
- Environment details (Python version, PyTorch version, CUDA version)
- Git commit hash
- Full hyperparameter configuration

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Log environment
import sys
writer.add_text('Environment/Python', sys.version)
writer.add_text('Environment/PyTorch', torch.__version__)
writer.add_text('Environment/CUDA', torch.version.cuda)
```

### Monitoring During Training

**Watch for**:
- **Overfitting**: Train loss decreases while val loss increases
- **Underfitting**: Both train and val loss remain high
- **Gradient issues**: Exploding or vanishing gradients
- **Class imbalance**: Per-class performance disparities
- **Fairness degradation**: FST performance gaps widening

## Troubleshooting

### Issue: TensorBoard not updating

**Solution**:
```bash
# Force refresh
tensorboard --logdir experiments/runs --reload_interval 5
```

### Issue: Out of disk space from logs

**Solution**:
```bash
# Reduce logging frequency
writer.add_scalar('Loss/train', loss.item(), global_step)
# Instead of logging every batch, log every N batches
if batch_idx % 100 == 0:
    writer.add_scalar('Loss/train', loss.item(), global_step)
```

### Issue: Slow TensorBoard loading

**Solution**:
- Archive old experiments
- Use `--samples_per_plugin` flag to limit data points

```bash
tensorboard --logdir experiments/runs --samples_per_plugin scalars=500
```

## Next Steps

1. Run baseline experiments: `experiments/baseline/`
2. Monitor training progress in TensorBoard
3. Compare model architectures
4. Conduct hyperparameter sweeps
5. Analyze fairness metrics
6. Document findings in experiment reports

---

**Last Updated**: 2025-10-13
**Tools**: TensorBoard 2.20.0, Weights & Biases 0.22.2 (optional)
