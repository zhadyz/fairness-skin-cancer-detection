"""
FairSkin Diffusion Model Implementation.

Wraps Stable Diffusion v1.5 with LoRA adaptation for FST-balanced
dermoscopy image generation. Supports text-conditioned generation
with diagnosis and FST-specific prompting.

Key Features:
- Stable Diffusion v1.5 base model
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- FST-targeted prompt engineering
- Batch generation with quality filtering
- Half-precision (FP16) for memory efficiency

Framework: MENDICANT_BIAS - Phase 2
Agent: HOLLOWED_EYES
Version: 0.3.0
Date: 2025-10-13
"""

import os
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.models import UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel


# HAM10000 diagnosis labels for prompt generation
DIAGNOSIS_LABELS = {
    0: 'actinic keratosis',
    1: 'basal cell carcinoma',
    2: 'benign keratosis',
    3: 'dermatofibroma',
    4: 'melanoma',
    5: 'nevus',
    6: 'vascular lesion',
}

# FST descriptions for prompting
FST_DESCRIPTIONS = {
    1: 'very light skin, pale white, type I',
    2: 'light skin, white, type II',
    3: 'light brown skin, type III',
    4: 'moderate brown skin, type IV',
    5: 'dark brown skin, type V',
    6: 'very dark brown to black skin, type VI',
}


class FairSkinDiffusionModel:
    """
    FairSkin Diffusion Model for FST-balanced dermoscopy image generation.

    Architecture:
    - Base: Stable Diffusion v1.5 (860M params)
    - Adaptation: LoRA rank-16 on U-Net attention layers
    - Text encoder: CLIP (frozen)
    - VAE: Variational autoencoder for latent diffusion

    Training:
    - Fine-tuned on HAM10000 with FST-balanced prompts
    - LoRA rank 16, alpha 16
    - Training steps: 5000-10000
    - Resolution: 512x512
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        lora_weights_path: Optional[str] = None,
        safety_checker: Optional[bool] = False,
        scheduler_type: str = "pndm",
    ):
        """
        Initialize FairSkin diffusion model.

        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Model precision (torch.float16 or torch.float32)
            lora_weights_path: Path to trained LoRA weights (optional)
            safety_checker: Whether to use NSFW safety checker (disable for medical)
            scheduler_type: Noise scheduler ('pndm', 'ddim', 'ddpm', 'dpm')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_id = model_id
        self.lora_weights_path = lora_weights_path

        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        print(f"Initializing FairSkin Diffusion Model...")
        print(f"  Model: {model_id}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {dtype}")

        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None if not safety_checker else "default",
        )

        # Set scheduler
        self._set_scheduler(scheduler_type)

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Load LoRA weights if provided
        if lora_weights_path and Path(lora_weights_path).exists():
            self.load_lora_weights(lora_weights_path)
            print(f"  LoRA weights loaded from: {lora_weights_path}")
        else:
            if lora_weights_path:
                warnings.warn(f"LoRA weights not found at {lora_weights_path}")
            print(f"  No LoRA weights loaded (using base SD)")

        # Enable memory optimizations
        if self.device.type == "cuda":
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print(f"  xFormers memory efficient attention enabled")
            except Exception as e:
                print(f"  xFormers not available: {e}")

        print(f"FairSkin model initialized successfully!")

    def _set_scheduler(self, scheduler_type: str):
        """Set noise scheduler for diffusion sampling."""
        schedulers = {
            'ddpm': DDPMScheduler,
            'ddim': DDIMScheduler,
            'pndm': PNDMScheduler,
            'dpm': DPMSolverMultistepScheduler,
        }

        if scheduler_type not in schedulers:
            raise ValueError(f"Unknown scheduler: {scheduler_type}. Options: {list(schedulers.keys())}")

        scheduler_class = schedulers[scheduler_type]
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)

    def load_lora_weights(self, lora_weights_path: str):
        """
        Load LoRA weights into U-Net.

        Args:
            lora_weights_path: Path to LoRA checkpoint (.pt or .safetensors)
        """
        if not Path(lora_weights_path).exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")

        # Load state dict
        lora_state_dict = torch.load(lora_weights_path, map_location=self.device)

        # Load into U-Net
        # Note: This assumes LoRA weights are saved in a compatible format
        # In practice, you'd use peft's load_lora_weights or similar
        try:
            self.pipe.unet.load_attn_procs(lora_state_dict)
        except Exception as e:
            warnings.warn(f"Failed to load LoRA weights with load_attn_procs: {e}")
            # Fallback: try direct state dict loading
            try:
                self.pipe.unet.load_state_dict(lora_state_dict, strict=False)
            except Exception as e2:
                raise RuntimeError(f"Failed to load LoRA weights: {e2}")

    def save_lora_weights(self, output_path: str):
        """
        Save current LoRA weights.

        Args:
            output_path: Path to save LoRA checkpoint
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract LoRA state dict from U-Net
        try:
            lora_state_dict = self.pipe.unet.get_attn_procs()
            torch.save(lora_state_dict, output_path)
            print(f"LoRA weights saved to: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save LoRA weights: {e}")

    def create_prompt(
        self,
        diagnosis: Union[int, str],
        fst: int,
        style: str = "clinical",
        add_quality_tokens: bool = True,
    ) -> str:
        """
        Create text prompt for dermoscopy image generation.

        Args:
            diagnosis: Diagnosis class (int 0-6 or string name)
            fst: Fitzpatrick Skin Type (1-6)
            style: Prompt style ('clinical', 'dermoscopic', 'medical')
            add_quality_tokens: Whether to add quality-enhancing tokens

        Returns:
            Text prompt string
        """
        # Map diagnosis to name
        if isinstance(diagnosis, int):
            diagnosis_name = DIAGNOSIS_LABELS.get(diagnosis, 'skin lesion')
        else:
            diagnosis_name = diagnosis.lower()

        # Get FST description
        fst_desc = FST_DESCRIPTIONS.get(fst, f'Fitzpatrick type {fst} skin')

        # Build prompt
        if style == "clinical":
            prompt = (
                f"A high-quality clinical dermoscopic photograph of {diagnosis_name} "
                f"on {fst_desc}, professional medical imaging"
            )
        elif style == "dermoscopic":
            prompt = (
                f"Dermoscopic image of {diagnosis_name} on {fst_desc}, "
                f"polarized light, detailed lesion structure"
            )
        elif style == "medical":
            prompt = (
                f"Medical photograph showing {diagnosis_name} on {fst_desc}, "
                f"clinical documentation quality"
            )
        else:
            prompt = f"{diagnosis_name} on {fst_desc}"

        # Add quality tokens
        if add_quality_tokens:
            prompt += ", sharp focus, high resolution, detailed, professional lighting"

        return prompt

    def create_negative_prompt(self) -> str:
        """
        Create negative prompt to avoid unwanted artifacts.

        Returns:
            Negative prompt string
        """
        return (
            "blurry, low quality, distorted, amateur, watermark, text, "
            "cartoon, illustration, drawing, painting, artistic, "
            "duplicated, multiple lesions, ruler, annotation"
        )

    @torch.no_grad()
    def generate_image(
        self,
        diagnosis: Union[int, str],
        fst: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
        prompt_style: str = "clinical",
        return_latents: bool = False,
    ) -> Union[Image.Image, Tuple[Image.Image, torch.Tensor]]:
        """
        Generate single synthetic dermoscopy image.

        Args:
            diagnosis: Diagnosis class (int 0-6 or string)
            fst: Fitzpatrick Skin Type (1-6)
            num_inference_steps: Number of diffusion steps (20-100)
            guidance_scale: Classifier-free guidance scale (7-10 recommended)
            seed: Random seed for reproducibility
            height: Image height (must be multiple of 8)
            width: Image width (must be multiple of 8)
            prompt_style: Prompt style ('clinical', 'dermoscopic', 'medical')
            return_latents: Whether to return latent representations

        Returns:
            Generated PIL Image (and latents if return_latents=True)
        """
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Create prompts
        prompt = self.create_prompt(diagnosis, fst, style=prompt_style)
        negative_prompt = self.create_negative_prompt()

        # Generate image
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="pil" if not return_latents else "latent",
        )

        if return_latents:
            return output.images[0], output.latents
        else:
            return output.images[0]

    @torch.no_grad()
    def generate_batch(
        self,
        diagnoses: List[Union[int, str]],
        fsts: List[int],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seeds: Optional[List[int]] = None,
        batch_size: int = 4,
        height: int = 512,
        width: int = 512,
        prompt_style: str = "clinical",
    ) -> List[Image.Image]:
        """
        Generate batch of synthetic images.

        Args:
            diagnoses: List of diagnosis classes
            fsts: List of FST values (must match diagnoses length)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seeds: List of random seeds (optional)
            batch_size: Batch size for generation
            height: Image height
            width: Image width
            prompt_style: Prompt style

        Returns:
            List of generated PIL Images
        """
        if len(diagnoses) != len(fsts):
            raise ValueError("diagnoses and fsts must have same length")

        if seeds is not None and len(seeds) != len(diagnoses):
            raise ValueError("seeds must match diagnoses length")

        images = []
        num_batches = (len(diagnoses) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(diagnoses))

            batch_diagnoses = diagnoses[start_idx:end_idx]
            batch_fsts = fsts[start_idx:end_idx]
            batch_seeds = seeds[start_idx:end_idx] if seeds else [None] * len(batch_diagnoses)

            # Generate prompts
            prompts = [
                self.create_prompt(dx, fst, style=prompt_style)
                for dx, fst in zip(batch_diagnoses, batch_fsts)
            ]
            negative_prompt = self.create_negative_prompt()

            # Generate batch
            generators = [
                torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
                for seed in batch_seeds
            ]

            output = self.pipe(
                prompt=prompts,
                negative_prompt=[negative_prompt] * len(prompts),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generators[0] if len(generators) == 1 else None,
            )

            images.extend(output.images)

        return images

    def generate_fst_balanced_dataset(
        self,
        num_images_per_fst: int,
        target_fsts: List[int] = [5, 6],
        diagnoses: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        quality_filter: Optional['QualityFilter'] = None,
    ) -> Dict[str, List[Image.Image]]:
        """
        Generate FST-balanced synthetic dataset.

        This method generates synthetic images with balanced FST distribution,
        focusing on minority FST groups (V-VI).

        Args:
            num_images_per_fst: Number of images to generate per FST
            target_fsts: FST classes to generate (default [5, 6])
            diagnoses: Diagnosis classes to generate (default all 7)
            output_dir: Optional directory to save images
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            quality_filter: Optional quality filter to apply

        Returns:
            Dictionary mapping FST to list of generated images
        """
        if diagnoses is None:
            diagnoses = list(range(7))  # All 7 diagnosis classes

        # Calculate total images
        total_images = num_images_per_fst * len(target_fsts) * len(diagnoses)
        print(f"\nGenerating FST-balanced dataset:")
        print(f"  Target FSTs: {target_fsts}")
        print(f"  Diagnoses: {len(diagnoses)} classes")
        print(f"  Images per (FST, diagnosis): {num_images_per_fst}")
        print(f"  Total images: {total_images}")

        # Prepare generation lists
        all_diagnoses = []
        all_fsts = []

        for fst in target_fsts:
            for dx in diagnoses:
                all_diagnoses.extend([dx] * num_images_per_fst)
                all_fsts.extend([fst] * num_images_per_fst)

        # Generate images
        print(f"\nGenerating {len(all_diagnoses)} images...")
        generated_images = self.generate_batch(
            diagnoses=all_diagnoses,
            fsts=all_fsts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Apply quality filter if provided
        if quality_filter:
            print(f"Applying quality filter...")
            filtered_images = []
            for img, dx, fst in zip(generated_images, all_diagnoses, all_fsts):
                if quality_filter.passes_filter(img, dx, fst):
                    filtered_images.append((img, dx, fst))
            print(f"  Retained {len(filtered_images)}/{len(generated_images)} images")
            generated_images = [x[0] for x in filtered_images]
            all_diagnoses = [x[1] for x in filtered_images]
            all_fsts = [x[2] for x in filtered_images]

        # Organize by FST
        fst_images = {fst: [] for fst in target_fsts}
        for img, fst in zip(generated_images, all_fsts):
            fst_images[fst].append(img)

        # Save if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for idx, (img, dx, fst) in enumerate(zip(generated_images, all_diagnoses, all_fsts)):
                dx_name = DIAGNOSIS_LABELS[dx].replace(' ', '_')
                filename = f"synthetic_fst{fst}_{dx_name}_{idx:05d}.png"
                img.save(output_path / filename)

            print(f"Saved {len(generated_images)} images to {output_dir}")

        return fst_images

    def __repr__(self) -> str:
        return (
            f"FairSkinDiffusionModel(\n"
            f"  model_id={self.model_id},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f"  lora_loaded={self.lora_weights_path is not None}\n"
            f")"
        )


if __name__ == "__main__":
    """Demo and testing of FairSkinDiffusionModel."""
    print("=" * 70)
    print("FairSkin Diffusion Model Demo")
    print("=" * 70)

    # Initialize model (requires GPU for practical use)
    try:
        model = FairSkinDiffusionModel(
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        print("\nModel initialized successfully!")
        print(model)

        # Test prompt generation
        print("\nTesting prompt generation:")
        for dx in [0, 4, 5]:
            for fst in [1, 6]:
                prompt = model.create_prompt(dx, fst)
                print(f"  Diagnosis {dx}, FST {fst}: {prompt}")

        # Test image generation (commented out to avoid long runtime)
        # print("\nGenerating test image...")
        # image = model.generate_image(diagnosis=4, fst=6, seed=42)
        # image.save("test_fairskin_output.png")
        # print("  Saved to: test_fairskin_output.png")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("FairSkin diffusion model demo complete.")
    print("=" * 70)
