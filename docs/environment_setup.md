# Environment Setup Guide

## Prerequisites

- Python 3.10+ (tested with Python 3.13.7)
- Git
- 16GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA support for training

## Installation Instructions

### Windows

1. **Clone the repository**:
```bash
git clone <repository-url>
cd "skin cancer"
```

2. **Create virtual environment**:
```bash
python -m venv venv
```

3. **Activate virtual environment**:
```bash
# Command Prompt
venv\Scripts\activate.bat

# PowerShell
venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

4. **Upgrade pip**:
```bash
pip install --upgrade pip setuptools wheel
```

5. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Linux / macOS

1. **Clone the repository**:
```bash
git clone <repository-url>
cd skin-cancer-classification
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
```

3. **Activate virtual environment**:
```bash
source venv/bin/activate
```

4. **Upgrade pip**:
```bash
pip install --upgrade pip setuptools wheel
```

5. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## GPU Setup (CUDA)

### NVIDIA GPU Requirements

- CUDA 12.1+ compatible GPU
- NVIDIA Driver 530.30.02+
- CUDA Toolkit 12.1+
- cuDNN 8.x

### Installing PyTorch with CUDA Support

**Replace the CPU version** with GPU-enabled PyTorch:

```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install CUDA 12.1 version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output (with GPU):
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

## Verification Commands

### Check Python Version
```bash
python --version
# Should show: Python 3.10.x or higher
```

### Verify Core Libraries
```bash
# PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# timm (model architectures)
python -c "import timm; print(f'timm {timm.__version__}')"

# Fairness libraries
python -c "import fairlearn; print(f'Fairlearn {fairlearn.__version__}')"
python -c "import aif360; print('AIF360 installed')"

# Data science stack
python -c "import numpy, pandas, sklearn; print('Data science stack OK')"

# Computer vision
python -c "import cv2, albumentations; print('CV libraries OK')"

# Experiment tracking
python -c "import tensorboard, wandb; print('Tracking tools OK')"
```

### Run All Verifications
```bash
python -c "
import sys
import torch
import timm
import fairlearn
import aif360
import numpy
import pandas
import sklearn
import cv2
import albumentations
import tensorboard
import wandb

print('=== Environment Verification ===')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'timm: {timm.__version__}')
print(f'Fairlearn: {fairlearn.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print('All critical dependencies installed successfully!')
"
```

## Troubleshooting

### Issue: `ModuleNotFoundError` after installation

**Solution**: Ensure virtual environment is activated:
```bash
# Check which Python is being used
which python   # Linux/macOS
where python   # Windows

# Should point to venv directory
```

### Issue: CUDA not available despite GPU present

**Solution**:
1. Verify NVIDIA driver: `nvidia-smi`
2. Check CUDA toolkit: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version (see GPU Setup above)
4. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Out of memory during training

**Solutions**:
- Reduce batch size in configuration files
- Enable gradient checkpointing (already configured in model configs)
- Use mixed precision training (FP16)
- Consider using a smaller model architecture

### Issue: Slow data loading

**Solutions**:
- Increase `num_workers` in DataLoader (default: 4)
- Use SSD for dataset storage
- Pre-process and cache augmentations

### Issue: `ImportError: DLL load failed` (Windows)

**Solution**:
1. Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Reinstall PyTorch
3. Restart terminal

### Issue: Permission denied when creating directories

**Solution**:
```bash
# Linux/macOS: Use sudo or change ownership
sudo chown -R $USER:$USER .

# Windows: Run terminal as Administrator
```

### Issue: Pre-commit hooks failing

**Solution**:
```bash
# Reinstall pre-commit hooks
pre-commit clean
pre-commit install

# Run manually to test
pre-commit run --all-files
```

## Environment Variables (Optional)

Create `.env` file in project root:

```bash
# Weights & Biases (optional)
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=skin-cancer-classification

# Data directories
DATA_ROOT=./data
EXPERIMENTS_ROOT=./experiments

# Training configuration
CUDA_VISIBLE_DEVICES=0  # GPU device ID
OMP_NUM_THREADS=4       # CPU threads for data loading
```

## Development Tools

### Jupyter Notebook (Optional)
```bash
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=skin-cancer-env
```

### VS Code Extensions (Recommended)
- Python (Microsoft)
- Pylance
- Black Formatter
- Jupyter
- GitLens

### PyCharm Configuration
1. File > Settings > Project > Python Interpreter
2. Add Interpreter > Existing Environment
3. Select `venv/bin/python` (Linux/macOS) or `venv\Scripts\python.exe` (Windows)

## Next Steps

After successful environment setup:

1. Review project structure: `README.md`
2. Set up data directories: `docs/data_setup.md`
3. Configure experiments: `docs/experiment_tracking.md`
4. Run baseline experiments: `experiments/baseline/README.md`

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 16GB
- Storage: 50GB
- Python: 3.10+

### Recommended (for training)
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA RTX 3090 or better (24GB VRAM)
- Storage: 100GB+ SSD
- Python: 3.10-3.12

### Cloud Options
- Google Colab (Free GPU tier available)
- Kaggle Notebooks (Free GPU: 30hrs/week)
- AWS SageMaker
- Azure ML
- Lambda Labs

## Support

For issues not covered here:
1. Check GitHub Issues
2. Review PyTorch documentation: https://pytorch.org/docs/
3. Consult timm documentation: https://huggingface.co/docs/timm
4. Fairlearn docs: https://fairlearn.org/

---

**Last Updated**: 2025-10-13
**Python Version Tested**: 3.13.7
**PyTorch Version**: 2.8.0
