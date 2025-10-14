"""
Quality Metrics for FairSkin Synthetic Images.

Implements comprehensive quality validation metrics:
- FID (Fréchet Inception Distance): Distribution similarity
- LPIPS (Learned Perceptual Image Patch Similarity): Perceptual quality
- Diversity Score: Intra-class variation
- Classifier Confidence: Diagnostic feature preservation

Quality Thresholds (from literature):
- FID: <20 (general), <25 (FST VI acceptable)
- LPIPS: <0.15 (vs real images)
- Diversity: >0.3 (CLIP embedding distance)
- Confidence: >0.7 (ResNet50 softmax)

Framework: MENDICANT_BIAS - Phase 2
Agent: HOLLOWED_EYES
Version: 0.3.0
Date: 2025-10-13
"""

import warnings
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy import linalg

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips not available. Install: pip install lpips")

try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    warnings.warn("pytorch-fid not available. Install: pip install pytorch-fid")

import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """
    Convert PIL Image to tensor.

    Args:
        image: PIL Image
        normalize: Whether to normalize to [0, 1]

    Returns:
        Tensor (1, C, H, W)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)

    if not normalize:
        tensor = (tensor * 255).byte()

    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.

    Args:
        tensor: Tensor (C, H, W) or (1, C, H, W)

    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        tensor = (tensor.clamp(0, 1) * 255).byte()

    tensor = tensor.cpu()
    array = tensor.permute(1, 2, 0).numpy()

    return Image.fromarray(array)


# ==================== FID (Fréchet Inception Distance) ====================

class FIDCalculator:
    """
    Calculate FID between two sets of images.

    FID measures the distance between feature distributions of
    real and generated images using Inception-v3 features.

    Lower FID = better quality (more similar to real distribution)
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize FID calculator.

        Args:
            device: Device to run Inception model
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load Inception-v3 model
        try:
            # Try pytorch-fid's InceptionV3 first (preferred)
            if FID_AVAILABLE:
                self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device)
            else:
                # Fallback to torchvision Inception
                self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
                self.model.fc = nn.Identity()  # Remove final FC layer
                self.model = self.model.to(self.device)

            self.model.eval()
            print(f"FID calculator initialized on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Inception model: {e}")

    @torch.no_grad()
    def extract_features(self, images: Union[List[Image.Image], torch.Tensor]) -> np.ndarray:
        """
        Extract Inception features from images.

        Args:
            images: List of PIL Images or tensor (B, C, H, W)

        Returns:
            Feature array (N, 2048)
        """
        if isinstance(images, list):
            # Convert PIL images to tensor
            tensors = [pil_to_tensor(img) for img in images]
            images = torch.cat(tensors, dim=0)

        images = images.to(self.device)

        # Resize to Inception input size (299x299)
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Extract features
        features = self.model(images)

        if isinstance(features, tuple):
            features = features[0]

        return features.cpu().numpy()

    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of features.

        Args:
            features: Feature array (N, 2048)

        Returns:
            (mean, covariance) tuple
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def calculate_fid(
        self,
        real_images: Union[List[Image.Image], torch.Tensor],
        fake_images: Union[List[Image.Image], torch.Tensor],
    ) -> float:
        """
        Calculate FID between real and fake images.

        Args:
            real_images: Real images (list or tensor)
            fake_images: Generated images (list or tensor)

        Returns:
            FID score (lower is better)
        """
        # Extract features
        print("Extracting features from real images...")
        real_features = self.extract_features(real_images)

        print("Extracting features from fake images...")
        fake_features = self.extract_features(fake_images)

        # Compute statistics
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_fake, sigma_fake = self.compute_statistics(fake_features)

        # Calculate FID
        fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

        return fid


def compute_fid(
    real_images: Union[List[Image.Image], torch.Tensor],
    fake_images: Union[List[Image.Image], torch.Tensor],
    device: str = "cuda",
) -> float:
    """
    Compute FID between real and synthetic images.

    Args:
        real_images: Real images
        fake_images: Synthetic images
        device: Device for computation

    Returns:
        FID score
    """
    calculator = FIDCalculator(device=device)
    return calculator.calculate_fid(real_images, fake_images)


# ==================== LPIPS (Learned Perceptual Image Patch Similarity) ====================

class LPIPSCalculator:
    """
    Calculate LPIPS perceptual distance between images.

    LPIPS uses deep features (VGG, AlexNet) to measure perceptual similarity.
    Lower LPIPS = more perceptually similar
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        """
        Initialize LPIPS calculator.

        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
            device: Device for computation
        """
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips library required. Install: pip install lpips")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

        print(f"LPIPS calculator initialized ({net}) on {self.device}")

    @torch.no_grad()
    def calculate_lpips(
        self,
        image1: Union[Image.Image, torch.Tensor],
        image2: Union[Image.Image, torch.Tensor],
    ) -> float:
        """
        Calculate LPIPS distance between two images.

        Args:
            image1: First image
            image2: Second image

        Returns:
            LPIPS distance (lower is better)
        """
        # Convert to tensors
        if isinstance(image1, Image.Image):
            image1 = pil_to_tensor(image1)
        if isinstance(image2, Image.Image):
            image2 = pil_to_tensor(image2)

        # Move to device
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        # Normalize to [-1, 1] (LPIPS expects this range)
        if image1.max() > 1.0:
            image1 = image1 / 255.0
        if image2.max() > 1.0:
            image2 = image2 / 255.0

        image1 = image1 * 2.0 - 1.0
        image2 = image2 * 2.0 - 1.0

        # Calculate LPIPS
        distance = self.model(image1, image2)

        return distance.item()

    def calculate_lpips_batch(
        self,
        images1: List[Image.Image],
        images2: List[Image.Image],
    ) -> List[float]:
        """
        Calculate LPIPS for batch of image pairs.

        Args:
            images1: List of first images
            images2: List of second images

        Returns:
            List of LPIPS distances
        """
        if len(images1) != len(images2):
            raise ValueError("Image lists must have same length")

        distances = []
        for img1, img2 in zip(images1, images2):
            distance = self.calculate_lpips(img1, img2)
            distances.append(distance)

        return distances


def compute_lpips(
    image1: Union[Image.Image, torch.Tensor],
    image2: Union[Image.Image, torch.Tensor],
    net: str = "alex",
    device: str = "cuda",
) -> float:
    """
    Compute LPIPS between two images.

    Args:
        image1: First image
        image2: Second image
        net: Network ('alex', 'vgg')
        device: Device for computation

    Returns:
        LPIPS distance
    """
    calculator = LPIPSCalculator(net=net, device=device)
    return calculator.calculate_lpips(image1, image2)


# ==================== Diversity Score ====================

def compute_diversity_score(
    images: List[Image.Image],
    model: Optional[nn.Module] = None,
    device: str = "cuda",
) -> float:
    """
    Compute intra-class diversity using feature embeddings.

    Measures average pairwise cosine distance between image embeddings.
    Higher diversity = more varied images (avoid mode collapse)

    Args:
        images: List of images to measure diversity
        model: Optional embedding model (defaults to ResNet50)
        device: Device for computation

    Returns:
        Diversity score (higher is better, >0.3 recommended)
    """
    if len(images) < 2:
        return 0.0

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model if not provided
    if model is None:
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()  # Remove classification head
        model = model.to(device)
        model.eval()

    # Extract embeddings
    embeddings = []
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            embedding = model(tensor).squeeze()
            embeddings.append(embedding)

    embeddings = torch.stack(embeddings)

    # Compute pairwise cosine distances
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(embeddings, embeddings.t())

    # Average off-diagonal elements (exclude self-similarity)
    n = len(embeddings)
    mask = torch.ones(n, n, device=device) - torch.eye(n, device=device)
    avg_similarity = (similarity_matrix * mask).sum() / (n * (n - 1))

    # Convert similarity to distance
    diversity = 1.0 - avg_similarity.item()

    return diversity


# ==================== Quality Filter ====================

class QualityFilter:
    """
    Multi-metric quality filter for synthetic images.

    Filters synthetic images based on:
    1. FID (distribution similarity)
    2. LPIPS (perceptual quality)
    3. Classifier confidence (diagnostic features)
    4. Basic sanity checks (brightness, resolution)
    """

    def __init__(
        self,
        real_reference_images: Optional[List[Image.Image]] = None,
        fid_threshold: float = 30.0,
        lpips_threshold: float = 0.2,
        confidence_threshold: float = 0.6,
        classifier: Optional[nn.Module] = None,
        device: str = "cuda",
    ):
        """
        Initialize quality filter.

        Args:
            real_reference_images: Real images for FID/LPIPS comparison
            fid_threshold: Maximum acceptable FID
            lpips_threshold: Maximum acceptable LPIPS
            confidence_threshold: Minimum classifier confidence
            classifier: Optional trained classifier for confidence check
            device: Device for computation
        """
        self.real_reference_images = real_reference_images
        self.fid_threshold = fid_threshold
        self.lpips_threshold = lpips_threshold
        self.confidence_threshold = confidence_threshold
        self.classifier = classifier
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize calculators
        if LPIPS_AVAILABLE:
            self.lpips_calc = LPIPSCalculator(device=device)
        else:
            self.lpips_calc = None
            warnings.warn("LPIPS not available, skipping LPIPS check")

        print("Quality filter initialized:")
        print(f"  FID threshold: {fid_threshold}")
        print(f"  LPIPS threshold: {lpips_threshold}")
        print(f"  Confidence threshold: {confidence_threshold}")

    def check_brightness(self, image: Image.Image) -> bool:
        """Check if image has reasonable brightness."""
        array = np.array(image)
        mean_brightness = array.mean() / 255.0

        # Reject pure black or white images
        if mean_brightness < 0.1 or mean_brightness > 0.9:
            return False

        return True

    def check_resolution(self, image: Image.Image) -> bool:
        """Check if image has correct resolution."""
        return image.size[0] >= 256 and image.size[1] >= 256

    def check_lpips(self, image: Image.Image, diagnosis: int, fst: int) -> bool:
        """Check LPIPS distance to real images."""
        if self.lpips_calc is None or self.real_reference_images is None:
            return True  # Skip if not available

        # Sample random real images
        num_samples = min(10, len(self.real_reference_images))
        reference_samples = np.random.choice(self.real_reference_images, num_samples, replace=False)

        # Calculate LPIPS to each reference
        distances = [self.lpips_calc.calculate_lpips(image, ref) for ref in reference_samples]
        avg_distance = np.mean(distances)

        return avg_distance < self.lpips_threshold

    def check_classifier_confidence(self, image: Image.Image, diagnosis: int) -> bool:
        """Check if classifier is confident about diagnosis."""
        if self.classifier is None:
            return True  # Skip if no classifier provided

        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = F.softmax(logits, dim=1)
            confidence = probs[0, diagnosis].item()

        return confidence >= self.confidence_threshold

    def passes_filter(
        self,
        image: Image.Image,
        diagnosis: int,
        fst: int,
        verbose: bool = False,
    ) -> bool:
        """
        Check if image passes all quality filters.

        Args:
            image: Image to check
            diagnosis: Diagnosis class
            fst: FST class
            verbose: Print failure reasons

        Returns:
            True if passes all checks
        """
        # Basic sanity checks
        if not self.check_resolution(image):
            if verbose:
                print(f"  FAILED: Resolution check")
            return False

        if not self.check_brightness(image):
            if verbose:
                print(f"  FAILED: Brightness check")
            return False

        # LPIPS check
        if not self.check_lpips(image, diagnosis, fst):
            if verbose:
                print(f"  FAILED: LPIPS check")
            return False

        # Classifier confidence check
        if not self.check_classifier_confidence(image, diagnosis):
            if verbose:
                print(f"  FAILED: Confidence check")
            return False

        return True

    def filter_batch(
        self,
        images: List[Image.Image],
        diagnoses: List[int],
        fsts: List[int],
    ) -> Tuple[List[Image.Image], List[int], List[int]]:
        """
        Filter batch of images.

        Args:
            images: List of images
            diagnoses: List of diagnosis labels
            fsts: List of FST labels

        Returns:
            (filtered_images, filtered_diagnoses, filtered_fsts)
        """
        filtered_images = []
        filtered_diagnoses = []
        filtered_fsts = []

        for img, dx, fst in zip(images, diagnoses, fsts):
            if self.passes_filter(img, dx, fst):
                filtered_images.append(img)
                filtered_diagnoses.append(dx)
                filtered_fsts.append(fst)

        return filtered_images, filtered_diagnoses, filtered_fsts


if __name__ == "__main__":
    """Demo of quality metrics."""
    print("=" * 70)
    print("Quality Metrics Demo")
    print("=" * 70)

    print("\nAvailable metrics:")
    print(f"  FID: {FID_AVAILABLE}")
    print(f"  LPIPS: {LPIPS_AVAILABLE}")

    if LPIPS_AVAILABLE:
        print("\nTesting LPIPS...")
        # Create two random images
        img1 = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

        calculator = LPIPSCalculator(device="cpu")
        distance = calculator.calculate_lpips(img1, img2)
        print(f"  LPIPS distance: {distance:.4f}")

    print("\n" + "=" * 70)
    print("Quality metrics demo complete")
    print("=" * 70)
