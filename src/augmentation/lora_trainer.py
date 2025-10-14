"""
LoRA Trainer for FairSkin Diffusion Model.

Implements Low-Rank Adaptation (LoRA) fine-tuning of Stable Diffusion
on HAM10000 dermoscopy dataset with FST-balanced prompting.

LoRA Approach:
- Freeze base Stable Diffusion weights (860M params)
- Train low-rank decomposition matrices: ΔW = BA
- Rank r=16 (3-10M trainable params vs 860M)
- Target: U-Net cross-attention layers (most semantic)

Training Protocol:
- 5000-10000 steps (~10-20 GPU hours on RTX 3090)
- Batch size: 4 (gradient accumulation if needed)
- Learning rate: 1e-4
- Mixed precision (FP16) training
- Balanced FST sampling during training

Framework: MENDICANT_BIAS - Phase 2
Agent: HOLLOWED_EYES
Version: 0.3.0
Date: 2025-10-13
"""

import os
import math
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn("peft library not available. LoRA training will not work.")

from tqdm.auto import tqdm
import numpy as np


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training."""

    # Model
    model_id: str = "runwayml/stable-diffusion-v1-5"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # Default set in __post_init__

    # Training
    num_train_steps: int = 10000
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 500
    num_cycles: int = 3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Diffusion
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    prediction_type: str = "epsilon"
    snr_gamma: Optional[float] = 5.0

    # Data
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True

    # Optimization
    use_8bit_adam: bool = False
    mixed_precision: bool = True
    gradient_checkpointing: bool = True

    # Logging & Checkpointing
    logging_steps: int = 100
    checkpoint_steps: int = 1000
    validation_steps: int = 500
    output_dir: str = "checkpoints/fairskin_lora"

    # Hardware
    device: str = "cuda"
    dataloader_num_workers: int = 4

    def __post_init__(self):
        """Set default target modules if not provided."""
        if self.target_modules is None:
            # Target cross-attention layers in U-Net
            self.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]


class DermoscopyPromptDataset(Dataset):
    """
    Dataset that wraps HAM10000 and generates text prompts.

    For each image, generates:
    - Text prompt: "A dermoscopic image of {diagnosis} on FST {fst}"
    - Preprocessed image tensor
    """

    def __init__(
        self,
        ham10000_dataset: Dataset,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        """
        Initialize prompt dataset.

        Args:
            ham10000_dataset: HAM10000Dataset instance
            tokenizer: CLIP tokenizer for text encoding
            resolution: Image resolution (512)
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images
        """
        self.dataset = ham10000_dataset
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip

        # Diagnosis labels
        self.diagnosis_labels = {
            0: 'actinic keratosis',
            1: 'basal cell carcinoma',
            2: 'benign keratosis',
            3: 'dermatofibroma',
            4: 'melanoma',
            5: 'nevus',
            6: 'vascular lesion',
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.

        Returns:
            Dictionary with:
                - pixel_values: Image tensor (3, 512, 512), normalized [-1, 1]
                - input_ids: Tokenized prompt (77 tokens)
                - label: Diagnosis class
                - fst: FST class
        """
        sample = self.dataset[idx]

        # Get image (already tensor from HAM10000Dataset)
        image = sample['image']  # (C, H, W)

        # Preprocess image
        image = self._preprocess_image(image)

        # Create prompt
        diagnosis = sample['label']
        fst = sample.get('fst', -1)

        if fst == -1:
            # Unknown FST, use generic prompt
            prompt = f"A dermoscopic image of {self.diagnosis_labels[diagnosis]}, medical imaging"
        else:
            prompt = f"A dermoscopic image of {self.diagnosis_labels[diagnosis]} on Fitzpatrick type {fst} skin"

        # Tokenize prompt
        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'label': diagnosis,
            'fst': fst,
        }

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess image for Stable Diffusion.

        Args:
            image: Input tensor (C, H, W), range [0, 1] or [0, 255]

        Returns:
            Preprocessed tensor (C, 512, 512), range [-1, 1]
        """
        # Ensure float and [0, 1] range
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        # Resize to resolution
        if image.shape[1] != self.resolution or image.shape[2] != self.resolution:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Random flip
        if self.random_flip and torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[2])

        # Normalize to [-1, 1] (Stable Diffusion expects this range)
        image = image * 2.0 - 1.0

        return image


class LoRATrainer:
    """
    LoRA Trainer for FairSkin Diffusion Model.

    Implements efficient fine-tuning of Stable Diffusion using LoRA
    on HAM10000 dermoscopy dataset with FST-balanced sampling.
    """

    def __init__(
        self,
        config: LoRATrainingConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Initialize LoRA trainer.

        Args:
            config: Training configuration
            train_dataset: Training dataset (HAM10000Dataset)
            val_dataset: Optional validation dataset
        """
        if not PEFT_AVAILABLE:
            raise ImportError("peft library required for LoRA training. Install: pip install peft")

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Device setup
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        if config.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initializing LoRA Trainer...")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")

        # Load models
        self._load_models()

        # Setup LoRA
        self._setup_lora()

        # Setup training
        self._setup_training()

        print(f"LoRA Trainer initialized!")

    def _load_models(self):
        """Load Stable Diffusion models."""
        print(f"Loading Stable Diffusion models...")

        # Load full pipeline first
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.config.mixed_precision else torch.float32,
        )

        # Extract components
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # Move to device
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.unet.to(self.device)

        # Freeze VAE and text encoder (only train U-Net with LoRA)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        print(f"  Models loaded successfully")

    def _setup_lora(self):
        """Setup LoRA adaptation for U-Net."""
        print(f"Setting up LoRA...")

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            init_lora_weights=True,
        )

        # Apply LoRA to U-Net
        self.unet = get_peft_model(self.unet, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"  LoRA rank: {self.config.lora_rank}")
        print(f"  LoRA alpha: {self.config.lora_alpha}")
        print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Total params: {total_params:,}")

    def _setup_training(self):
        """Setup optimizer, scheduler, and dataloaders."""
        print(f"Setting up training...")

        # Create prompt dataset
        self.prompt_train_dataset = DermoscopyPromptDataset(
            ham10000_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            resolution=self.config.resolution,
            center_crop=self.config.center_crop,
            random_flip=self.config.random_flip,
        )

        # Create dataloader
        self.train_dataloader = DataLoader(
            self.prompt_train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
        )

        # Optimizer
        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                warnings.warn("bitsandbytes not available, using standard AdamW")
                optimizer_cls = AdamW
        else:
            optimizer_cls = AdamW

        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.num_train_steps,
            num_cycles=self.config.num_cycles if 'cosine' in self.config.lr_scheduler else 1,
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.mixed_precision else None

        # Training state
        self.global_step = 0
        self.epoch = 0

        print(f"  Optimizer: {optimizer_cls.__name__}")
        print(f"  LR scheduler: {self.config.lr_scheduler}")
        print(f"  Mixed precision: {self.config.mixed_precision}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space using VAE.

        Args:
            images: Batch of images (B, 3, 512, 512), range [-1, 1]

        Returns:
            Latent tensors (B, 4, 64, 64)
        """
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def encode_prompts(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text prompts using CLIP text encoder.

        Args:
            input_ids: Tokenized prompts (B, 77)

        Returns:
            Text embeddings (B, 77, 768)
        """
        return self.text_encoder(input_ids)[0]

    def compute_loss(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.

        Args:
            latents: Latent representations (B, 4, 64, 64)
            encoder_hidden_states: Text embeddings (B, 77, 768)
            timesteps: Optional timesteps (B,)

        Returns:
            Loss scalar
        """
        batch_size = latents.shape[0]

        # Sample timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        # Compute loss
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")

        loss = F.mse_loss(model_pred, target, reduction="mean")

        # SNR weighting (improves quality)
        if self.config.snr_gamma is not None:
            snr = self.scheduler.alphas_cumprod[timesteps] / (1 - self.scheduler.alphas_cumprod[timesteps])
            snr_weight = torch.clamp(snr, max=self.config.snr_gamma)
            loss = loss * snr_weight.mean()

        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.

        Args:
            batch: Batch dictionary with pixel_values and input_ids

        Returns:
            Loss value
        """
        # Move to device
        pixel_values = batch['pixel_values'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)

        # Encode
        with autocast(enabled=self.config.mixed_precision):
            latents = self.encode_images(pixel_values)
            encoder_hidden_states = self.encode_prompts(input_ids)

            # Compute loss
            loss = self.compute_loss(latents, encoder_hidden_states)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train(self):
        """Main training loop."""
        print(f"\nStarting LoRA training...")
        print(f"  Total steps: {self.config.num_train_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        self.unet.train()
        progress_bar = tqdm(total=self.config.num_train_steps, desc="Training")

        accumulation_loss = 0.0
        steps_in_epoch = len(self.train_dataloader)

        while self.global_step < self.config.num_train_steps:
            self.epoch += 1

            for step, batch in enumerate(self.train_dataloader):
                # Training step
                loss = self.train_step(batch)
                accumulation_loss += loss

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.unet.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Update progress
                    self.global_step += 1
                    progress_bar.update(1)

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulation_loss / self.config.logging_steps
                        lr = self.optimizer.param_groups[0]['lr']
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{lr:.2e}',
                            'epoch': self.epoch
                        })
                        accumulation_loss = 0.0

                    # Checkpointing
                    if self.global_step % self.config.checkpoint_steps == 0:
                        self.save_checkpoint()

                    # Check if done
                    if self.global_step >= self.config.num_train_steps:
                        break

            if self.global_step >= self.config.num_train_steps:
                break

        progress_bar.close()

        # Save final checkpoint
        self.save_checkpoint(is_final=True)

        print(f"\nTraining complete!")
        print(f"  Total steps: {self.global_step}")
        print(f"  Total epochs: {self.epoch}")

    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        if is_final:
            checkpoint_path = self.output_dir / "lora_weights_final.pt"
        else:
            checkpoint_path = self.output_dir / f"lora_weights_step_{self.global_step}.pt"

        # Save LoRA weights only
        lora_state_dict = self.unet.get_peft_model_state_dict()

        torch.save({
            'lora_state_dict': lora_state_dict,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }, checkpoint_path)

        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.unet.load_state_dict(checkpoint['lora_state_dict'], strict=False)
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)

        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"  Global step: {self.global_step}")
        print(f"  Epoch: {self.epoch}")


if __name__ == "__main__":
    """Demo of LoRA trainer."""
    print("=" * 70)
    print("LoRA Trainer Demo")
    print("=" * 70)

    # Create config
    config = LoRATrainingConfig(
        num_train_steps=100,  # Short demo
        batch_size=2,
        output_dir="checkpoints/fairskin_lora_demo"
    )

    print("\nTraining configuration:")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Training steps: {config.num_train_steps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")

    print("\n" + "=" * 70)
    print("Note: Actual training requires HAM10000 dataset and GPU")
    print("=" * 70)
