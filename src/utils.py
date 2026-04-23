"""
Utility functions:
  - Fixed influence functions phi_k
  - PSNR / SSIM metrics
  - Patch extraction helpers
  - Device selection
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Literal


# ---------------------------------------------------------------------------
# Influence functions (fixed phi_k, not learned)
# ---------------------------------------------------------------------------

def soft_threshold(x: Tensor, threshold: float = 0.1) -> Tensor:
    """
    Element-wise soft thresholding.
      phi(x) = sign(x) * max(|x| - threshold, 0)
    Corresponds to L1 / total-variation type regularisation.
    """
    return torch.sign(x) * F.relu(x.abs() - threshold)


def gaussian_deriv(x: Tensor, sigma: float = 1.0) -> Tensor:
    """
    Gaussian derivative influence function.
      phi(x) = x * exp(-x^2 / (2*sigma^2))
    Encourages moderate gradients while suppressing very large ones.
    """
    return x * torch.exp(-x.pow(2) / (2.0 * sigma ** 2))


def lorentzian(x: Tensor, scale: float = 1.0) -> Tensor:
    """
    Lorentzian (Cauchy) influence function.
      phi(x) = x / (1 + (x/scale)^2)
    Robust to outliers; edge-preserving behaviour.
    """
    return x / (1.0 + (x / scale).pow(2))


PHI_REGISTRY: dict[str, callable] = {
    "soft_threshold": soft_threshold,
    "gaussian_deriv": gaussian_deriv,
    "lorentzian":     lorentzian,
}


def get_phi(
    name: Literal["soft_threshold", "gaussian_deriv", "lorentzian"],
    **kwargs,
) -> callable:
    """
    Return the influence function by name, with kwargs bound.

    Usage:
        phi = get_phi("soft_threshold", threshold=0.1)
        out = phi(x)
    """
    if name not in PHI_REGISTRY:
        raise ValueError(f"Unknown influence function '{name}'. "
                         f"Choose from {list(PHI_REGISTRY)}")
    fn = PHI_REGISTRY[name]
    if kwargs:
        import functools
        return functools.partial(fn, **kwargs)
    return fn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Inputs should be in [0, max_val], shape (B, C, H, W) or (C, H, W).
    Returns the mean PSNR over the batch.
    """
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """
    Structural Similarity Index (SSIM).
    Inputs shape: (B, C, H, W), values in [0, max_val].
    Returns mean SSIM over the batch.

    Uses a Gaussian window of the given size.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma=1.5, channels=channels).to(pred.device)

    pad = window_size // 2

    mu1 = F.conv2d(pred,   window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12   = F.conv2d(pred   * target, window, padding=pad, groups=channels) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return (numerator / denominator).mean().item()


def _gaussian_window(size: int, sigma: float, channels: int) -> Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.outer(g)
    # shape: (channels, 1, size, size)
    return window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous()


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(image: Tensor, sigma: float) -> Tensor:
    """
    Add i.i.d. Gaussian noise with std = sigma / 255 (assumes image in [0,1]).
    """
    noise_std = sigma / 255.0
    return image + torch.randn_like(image) * noise_std


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(
    image: Tensor,
    patch_size: int,
    stride: int | None = None,
) -> Tensor:
    """
    Extract all (patch_size x patch_size) patches from a (C, H, W) image.
    stride defaults to patch_size (non-overlapping).
    Returns: (N, C, patch_size, patch_size)
    """
    if stride is None:
        stride = patch_size
    C, H, W = image.shape
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    # patches: (C, n_h, n_w, patch_size, patch_size)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)
    # -> (N, C, patch_size, patch_size)
    return patches.permute(1, 0, 2, 3)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string.
    "auto" selects cuda > mps > cpu in that priority order.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
