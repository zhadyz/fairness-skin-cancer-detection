# Docker Deployment Guide

## Overview

This project uses Docker and Docker Compose for reproducible environments across development, training, and inference.

## Prerequisites

### Required Software

- **Docker**: 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: 2.0+ (included with Docker Desktop)
- **NVIDIA Docker** (for GPU): nvidia-docker2

### GPU Support (Linux)

**Install NVIDIA Container Toolkit**:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Docker Images

### Available Build Targets

1. **base**: PyTorch with CUDA runtime
2. **development**: Base + dev tools (Jupyter, IPython)
3. **production**: Optimized for training
4. **inference**: Minimal runtime for serving models

## Quick Start

### 1. Build Images

**Build all images**:
```bash
docker-compose build
```

**Build specific target**:
```bash
# Development
docker build --target development -t skin-cancer-classifier:dev .

# Production
docker build --target production -t skin-cancer-classifier:prod .

# Inference
docker build --target inference -t skin-cancer-classifier:inference .
```

### 2. Run Services

#### Development Environment (Jupyter + TensorBoard)

```bash
docker-compose up dev
```

**Access**:
- Jupyter Notebook: http://localhost:8888
- TensorBoard: http://localhost:6006

**Get Jupyter token**:
```bash
docker logs skin-cancer-dev 2>&1 | grep token
```

#### Training with GPU

```bash
docker-compose up training
```

**Monitor logs**:
```bash
docker logs -f skin-cancer-training
```

#### Training on CPU (for testing)

```bash
docker-compose up training-cpu
```

#### TensorBoard Only

```bash
docker-compose up tensorboard
```

Access at: http://localhost:6006

#### Inference Server

```bash
docker-compose up inference
```

API available at: http://localhost:8000

### 3. Interactive Shell

**Enter development container**:
```bash
docker-compose run --rm dev bash
```

**Enter training container**:
```bash
docker-compose run --rm training bash
```

## Advanced Usage

### Custom Training Configuration

**Override command**:
```bash
docker-compose run --rm training \
    python experiments/baseline/train_resnet50.py \
    --config baseline_config.yaml \
    --epochs 50 \
    --batch-size 64
```

### Multi-GPU Training

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # Use all GPUs
          capabilities: [gpu]
```

Or specify GPU IDs:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3
```

### Volume Mounting

**Mount custom data directory**:
```bash
docker-compose run --rm \
    -v /path/to/custom/data:/app/data \
    training python experiments/baseline/train_resnet50.py
```

### Environment Variables

**Pass custom environment variables**:
```bash
docker-compose run --rm \
    -e WANDB_API_KEY=your_key \
    -e BATCH_SIZE=64 \
    training python experiments/baseline/train_resnet50.py
```

### Resource Limits

**Limit CPU and memory**:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
    reservations:
      cpus: '2'
      memory: 8G
```

## Production Deployment

### Build Optimized Images

```bash
# Optimize for size
docker build --target production \
    --build-arg PYTHON_VERSION=3.10 \
    -t skin-cancer-classifier:v1.0 .

# Tag for registry
docker tag skin-cancer-classifier:v1.0 \
    your-registry/skin-cancer-classifier:v1.0

# Push to registry
docker push your-registry/skin-cancer-classifier:v1.0
```

### Kubernetes Deployment

**Example Job for training**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: skin-cancer-training
spec:
  template:
    spec:
      containers:
      - name: training
        image: your-registry/skin-cancer-classifier:v1.0
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: experiments
          mountPath: /app/experiments
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: data-pvc
      - name: experiments
        persistentVolumeClaim:
          claimName: experiments-pvc
      restartPolicy: Never
```

### Docker Swarm

```bash
docker stack deploy -c docker-compose.yml skin-cancer-stack
```

## Inference API

### Start Inference Server

```bash
docker-compose up -d inference
```

### API Endpoints

**Health check**:
```bash
curl http://localhost:8000/health
```

**Predict** (example - API to be implemented):
```bash
curl -X POST http://localhost:8000/predict \
    -F "image=@sample_lesion.jpg" \
    -F "metadata={\"age\":45,\"location\":\"arm\"}"
```

**Batch inference**:
```bash
curl -X POST http://localhost:8000/predict_batch \
    -F "images[]=@image1.jpg" \
    -F "images[]=@image2.jpg"
```

## Data Management

### Persistent Volumes

**Create named volumes**:
```bash
docker volume create skin-cancer-data
docker volume create skin-cancer-experiments
```

**Use in docker-compose.yml**:
```yaml
volumes:
  - skin-cancer-data:/app/data
  - skin-cancer-experiments:/app/experiments
```

### Backup Data

```bash
# Backup data volume
docker run --rm \
    -v skin-cancer-data:/data \
    -v $(pwd):/backup \
    busybox tar czf /backup/data-backup.tar.gz /data

# Restore
docker run --rm \
    -v skin-cancer-data:/data \
    -v $(pwd):/backup \
    busybox tar xzf /backup/data-backup.tar.gz -C /
```

## Troubleshooting

### Issue: GPU not accessible in container

**Check NVIDIA Docker runtime**:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**If error, restart Docker**:
```bash
sudo systemctl restart docker
```

**Verify runtime config** (`/etc/docker/daemon.json`):
```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

### Issue: Permission denied on volumes

**Fix permissions**:
```bash
# Linux
sudo chown -R $USER:$USER data/ experiments/

# Or run container as current user
docker-compose run --rm --user $(id -u):$(id -g) training bash
```

### Issue: Out of memory

**Reduce batch size** or **increase Docker memory limit**:

```bash
# Docker Desktop: Settings > Resources > Memory
# Linux: Edit /etc/docker/daemon.json
{
  "default-shm-size": "2G"
}
```

### Issue: Slow build times

**Use BuildKit**:
```bash
export DOCKER_BUILDKIT=1
docker build --target production -t skin-cancer-classifier:prod .
```

**Cache optimization**:
```bash
# Use cache from registry
docker build --cache-from your-registry/skin-cancer-classifier:latest \
    --target production -t skin-cancer-classifier:prod .
```

### Issue: Container exits immediately

**Check logs**:
```bash
docker logs skin-cancer-training
```

**Run interactively**:
```bash
docker-compose run --rm training bash
```

## Best Practices

### Security

1. **Don't run as root in production**:
   - Use non-root user (already configured in Dockerfile)
   - Scan images: `docker scan skin-cancer-classifier:prod`

2. **Secrets management**:
   - Use Docker secrets or environment files
   - Never hardcode API keys in images

3. **Network security**:
   - Use custom networks
   - Restrict port exposure

### Performance

1. **Multi-stage builds**: Reduce image size
2. **Layer caching**: Order Dockerfile commands by change frequency
3. **Shared memory**: Increase for data loading

```yaml
shm_size: '2gb'  # In docker-compose.yml
```

### Monitoring

**Container stats**:
```bash
docker stats skin-cancer-training
```

**GPU usage**:
```bash
docker exec skin-cancer-training nvidia-smi
```

**Resource usage over time**:
```bash
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
    --no-stream
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build Docker image
  run: |
    docker build --target production -t skin-cancer-classifier:latest .

- name: Run tests in container
  run: |
    docker run --rm skin-cancer-classifier:latest pytest tests/

- name: Push to registry
  run: |
    echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
    docker push your-registry/skin-cancer-classifier:latest
```

## Cleanup

**Stop all services**:
```bash
docker-compose down
```

**Remove volumes**:
```bash
docker-compose down -v
```

**Clean up images**:
```bash
# Remove unused images
docker image prune -a

# Remove specific image
docker rmi skin-cancer-classifier:dev
```

**Complete cleanup**:
```bash
docker system prune -a --volumes
```

## Next Steps

1. Build development image: `docker-compose build dev`
2. Start Jupyter environment: `docker-compose up dev`
3. Run baseline training: `docker-compose up training-cpu`
4. Monitor with TensorBoard: `docker-compose up tensorboard`
5. Deploy inference server: `docker-compose up inference`

---

**Last Updated**: 2025-10-13
**Docker Version**: 20.10+
**Docker Compose Version**: 2.0+
