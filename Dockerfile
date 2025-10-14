# Multi-stage Dockerfile for skin cancer classification

# Base stage - PyTorch with CUDA support
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/data /app/experiments /app/logs

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Development stage - includes dev tools
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    notebook \
    ipykernel

# Copy source code
COPY . .

# Install pre-commit hooks
RUN pip install --no-cache-dir pre-commit && \
    git init && \
    pre-commit install || true

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command for development
CMD ["/bin/bash"]

# Production/Training stage
FROM base AS production

# Copy only necessary files
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY README.md LICENSE ./

# Set Python path
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1

# Default training command
CMD ["python", "experiments/baseline/train_resnet50.py"]

# Inference stage - minimal runtime
FROM base AS inference

# Copy only model code and checkpoints
COPY src/models/ ./src/models/
COPY src/utils/ ./src/utils/
COPY experiments/checkpoints/ ./experiments/checkpoints/

# Create inference user
RUN useradd -m -u 1000 inferuser && \
    chown -R inferuser:inferuser /app

USER inferuser

# Expose inference API port
EXPOSE 8000

# Run inference server (to be implemented)
CMD ["python", "src/api/inference_server.py"]
