"""
"""

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    # Number of convolutional filters in the filter bank per stage
    K: int = 24

    # Spatial size of each filter (filter_size x filter_size)
    filter_size: int = 7

    # Number of unrolled stages T  (NOT epochs — see note above)
    T: int = 5

    # Damping coefficient gamma — FIXED, not learned (per assignment)
    gamma: float = 0.5

    # Time step a = Delta_t — used to initialise alpha/beta
    tau: float = 0.2  # named tau here, corresponds to 'a' in the discretization

    # If True, coeff_curr, coeff_prev, coeff_diff are free learned scalars per stage.
    # If False, they are derived from gamma and tau and held fixed.
    learn_coeffs: bool = True

    # Influence function phi_k — FIXED, not learned (per assignment)
    # Options: "soft_threshold" | "gaussian_deriv" | "lorentzian"
    phi: Literal["soft_threshold", "gaussian_deriv", "lorentzian"] = "lorentzian"

    # phi parameters (fixed)
    phi_threshold: float = 0.01  # Lowered to avoid dead gradients
    phi_sigma: float = 1.0       # for gaussian_deriv
    phi_scale: float = 1.0       # for lorentzian

    # Initial value of data fidelity weight lambda (learned scalar per stage)
    # For Gaussian noise, we need a small anchor to the noisy image.
    lambda_init: float = 0.1

    @property
    def coeff_curr_init(self) -> float:
        """coeff_curr = 2 / (1 + gamma*tau/2)"""
        return 2.0 / (1.0 + self.gamma * self.tau / 2.0)

    @property
    def coeff_prev_init(self) -> float:
        """coeff_prev = -(1 - gamma*tau/2) / (1 + gamma*tau/2)"""
        return -(1.0 - self.gamma * self.tau / 2.0) / (1.0 + self.gamma * self.tau / 2.0)

    @property
    def coeff_diff_init(self) -> float:
        """coeff_diff = tau^2 / (1 + gamma*tau/2)"""
        return (self.tau ** 2) / (1.0 + self.gamma * self.tau / 2.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainParams:
    # Root directory for all outputs (runs, checkpoints, etc)
    output_root: str = "runs"

    # Noise standard deviation sigma. Train one model per noise level.
    sigma: int = 25

    # Patch size for training crops
    patch_size: int = 40

    # Number of patches per batch (TNRD uses 128)
    batch_size: int = 128

    # Number of epochs per greedy stage (one full dataset pass = one epoch)
    epochs_per_stage: int = 30

    # Loss function: "mse" | "l1"
    # Default: MSE, matching the TNRD paper loss (1/2)||u_T - u_gt||^2
    loss_fn: Literal["mse", "l1"] = "mse"

    # Learning rate for convolutional filter parameters
    lr_filters: float = 1e-3

    # Learning rate for scalar parameters (alpha, beta, lambda)
    # Higher than lr_filters because scalars are just 2-3 numbers per stage
    lr_scalars: float = 5e-2

    # Adam betas
    adam_betas: tuple = (0.9, 0.999)

    # Weight decay
    weight_decay: float = 1e-4

    # Gradient clipping max norm (None to disable)
    grad_clip: float | None = 1.0

    # Random seed
    seed: int = 42

    # Checkpoint directory
    checkpoint_dir: str = "checkpoints"

    # How often (in epochs) to save an intermediate checkpoint
    save_every: int = 10

    # Device: "cuda" | "mps" | "cpu" | "auto"
    device: str = "auto"

    # Number of dataloader workers
    num_workers: int = 4

    # Save a sample denoised image to outputs/ every N epochs (0 = disable)
    # This lets you see the diffusion visually during training
    visualize_every: int = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataParams:
    # Root directory where datasets are stored
    data_root: str = "data"

    # Training datasets. Start with cbsd68 (small), then add bsd400/flickr30k.
    # cbsd68  : 68 colour images — good for fast MVP iteration
    # bsd400  : 400 greyscale images — standard TNRD training set
    train_datasets: list[str] = field(default_factory=lambda: ["foe400"])

    # Test datasets
    test_datasets: list[str] = field(default_factory=lambda: ["cbsd68"])

    # BSDS500 archive URL
    bsds500_url: str = (
        "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/"
        "BSR/BSR_bsds500.tgz"
    )

    # CBSD68 via HuggingFace
    cbsd68_hf_name: str = "deepinv/CBSD68"

    # Convert images to greyscale for training
    grayscale: bool = True

    # Data augmentation: random flips + 90-degree rotations
    augment: bool = True

    # Use synthetic training data if no real images available (debug only)
    allow_synthetic: bool = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalParams:
    # Noise levels to evaluate at
    sigma_list: list[int] = field(default_factory=lambda: [15, 25, 50])

    # Save denoised output images to disk
    save_images: bool = True

    # Output directory for saved images
    output_dir: str = "outputs"

    # How many images to save (None = all)
    num_save: int | None = 5

    # Show per-stage diffusion progress images during evaluation
    # This saves one image per stage so you can see the diffusion evolving
    show_diffusion_stages: bool = True


# ---------------------------------------------------------------------------
# Composite config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model: ModelParams = field(default_factory=ModelParams)
    train: TrainParams = field(default_factory=TrainParams)
    data: DataParams = field(default_factory=DataParams)
    eval: EvalParams = field(default_factory=EvalParams)

    # Automatically set during training/eval
    run_dir: str = ""


DEFAULT_CONFIG = Config()
