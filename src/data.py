"""
Dataset classes and download helpers.

Supported datasets
------------------
  bsd400    : 400 greyscale training images from BSDS500
  bsd68     : 68 greyscale test images from BSDS500
  cbsd68    : 68 colour test images (HuggingFace deepinv/CBSD68)
  flickr30k : 30k images from HuggingFace nlphuji/flickr30k (training only)
"""

import tarfile
import urllib.request
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

from .params import DataParams, TrainParams
from .utils import add_gaussian_noise


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_bsds500(data_root: str, url: str) -> None:
    """Download and extract BSDS500 (BSD400 train + BSD68 test)."""
    root = Path(data_root)
    dest = root / "BSR"
    if dest.exists():
        print("BSDS500 already downloaded.")
        return

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "BSR_bsds500.tgz"

    print(f"Downloading BSDS500 ...")
    urllib.request.urlretrieve(url, archive, reporthook=_progress_hook())
    print("\nExtracting ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    archive.unlink()
    print("BSDS500 ready.")


def download_cbsd68(data_root: str, hf_name: str) -> None:
    """Download CBSD68 via HuggingFace datasets."""
    dest = Path(data_root) / "cbsd68"
    if dest.exists():
        print("CBSD68 already downloaded.")
        return

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install the 'datasets' package:  uv add datasets")

    print(f"Downloading CBSD68 from HuggingFace ({hf_name}) ...")

    # Try common splits; deepinv datasets vary between versions
    ds = None
    for split in ("test", "train", "all"):
        try:
            ds = load_dataset(hf_name, split=split)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"Could not load any split from {hf_name}")

    # Detect which key holds the PIL Image (robust to "image", "png", etc.)
    sample0 = ds[0]
    image_key = next((k for k, v in sample0.items() if hasattr(v, "save")), None)
    if image_key is None:
        raise RuntimeError(
            f"No PIL Image column found in {hf_name}. Keys: {list(sample0.keys())}"
        )

    dest.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(ds):
        sample[image_key].save(dest / f"{i:04d}.png")
    print(f"CBSD68 saved ({len(ds)} images, key='{image_key}').")


def download_foe400(data_root: str) -> None:
    """
    Sets up FOE400 training set. 
    Expects 'FoETrainingSets176.zip' in data_root.
    (This dataset is not publicly scrapable via simple URL).
    """
    import zipfile
    root = Path(data_root)
    zip_path = root / "FoETrainingSets176.zip"
    dest = root / "foe400"

    if dest.exists():
        print("FOE400 already exists.")
        return

    if not zip_path.exists():
        print(f"❌ Error: FOE400 requires {zip_path} to be manually placed in {data_root}.")
        print("You can find it in the original TNRD / FoE distribution.")
        return

    print(f"Extracting {zip_path.name} ...")
    _extract_zip(zip_path, dest)
    print(f"FOE400 ready at {dest}")


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    import zipfile
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def download_flickr30k(data_root: str, hf_name: str = "nlphuji/flickr30k") -> None:
    """
    Download Flickr30k images via HuggingFace (nlphuji/flickr30k).
    Only the images are saved; captions are discarded.
    """
    dest = Path(data_root) / "flickr30k"
    if dest.exists():
        print("Flickr30k already downloaded.")
        return

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install the 'datasets' package:  uv add datasets")

    print(f"Downloading Flickr30k from HuggingFace ({hf_name}) ...")
    print("Note: this dataset is ~4 GB and may take a while.")

    # nlphuji/flickr30k has a single "test" split
    ds = None
    for split in ("test", "train", "all"):
        try:
            ds = load_dataset(hf_name, split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f"Could not load any split from {hf_name}")

    # Detect image key
    sample0 = ds[0]
    image_key = next((k for k, v in sample0.items() if hasattr(v, "save")), None)
    if image_key is None:
        raise RuntimeError(
            f"No PIL Image column found in {hf_name}. Keys: {list(sample0.keys())}"
        )

    dest.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(ds):
        if i % 1000 == 0:
            print(f"  {i}/{len(ds)}", end="\r", flush=True)
        sample[image_key].save(dest / f"{i:06d}.jpg")
    print(f"\nFlickr30k saved ({len(ds)} images).")


def _progress_hook():
    prev = [0]

    def hook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            if pct != prev[0]:
                print(f"\r  {pct}%", end="", flush=True)
                prev[0] = pct

    return hook


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _bsds_split_dir(data_root: str, split: str) -> Path:
    return Path(data_root) / "BSR" / "BSDS500" / "data" / "images" / split


def _load_image_paths(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    # Search recursively to find images even if they are in subfolders (common for FOE)
    for p in directory.rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Synthetic dataset (for testing without real images)
# ---------------------------------------------------------------------------

class SyntheticDenoisingDataset(Dataset):
    """Generates synthetic patches on-the-fly for testing/demo purposes."""

    def __init__(
        self,
        sigma: int,
        patch_size: int = 40,
        num_images: int = 100,
        patches_per_image: int = 64,
    ):
        self.sigma = sigma
        self.patch_size = patch_size
        self.num_images = num_images
        self.patches_per_image = patches_per_image

    def __len__(self) -> int:
        return self.num_images * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        clean = torch.rand(1, self.patch_size, self.patch_size)
        for _ in range(3):
            y = torch.randint(0, self.patch_size, (1,)).item()
            x = torch.randint(0, self.patch_size, (1,)).item()
            r = torch.randint(2, 10, (1,)).item()
            clean[0, max(0, y-r):min(self.patch_size, y+r),
                     max(0, x-r):min(self.patch_size, x+r)] = torch.rand(1).item()
        noisy = add_gaussian_noise(clean, self.sigma).clamp(0.0, 1.0)
        return noisy, clean


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    """
    Patch-based training dataset.
    Returns (noisy_patch, clean_patch) of shape (1, patch_size, patch_size).
    Noise is sampled fresh each epoch.
    """

    def __init__(
        self,
        image_paths: list[Path],
        sigma: int,
        patch_size: int,
        augment: bool = True,
        grayscale: bool = True,
        patches_per_image: int = 64,
    ):
        self.paths = image_paths
        self.sigma = sigma
        self.patch_size = patch_size
        self.augment = augment
        self.grayscale = grayscale
        self.patches_per_image = patches_per_image
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.paths[idx // self.patches_per_image])
        img = img.convert("L") if self.grayscale else img.convert("RGB")
        patch = self._random_crop(img)
        if self.augment:
            patch = self._augment(patch)
        clean = self.to_tensor(patch)
        noisy = add_gaussian_noise(clean, self.sigma).clamp(0.0, 1.0)
        return noisy, clean

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        p = self.patch_size
        if w < p or h < p:
            img = TF.resize(img, (max(h, p), max(w, p)))
            w, h = img.size
        x = torch.randint(0, w - p + 1, (1,)).item()
        y = torch.randint(0, h - p + 1, (1,)).item()
        return TF.crop(img, y, x, p, p)

    @staticmethod
    def _augment(img: Image.Image) -> Image.Image:
        if torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
        if torch.rand(1).item() > 0.5:
            img = TF.vflip(img)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            img = TF.rotate(img, 90 * k)
        return img


class TestDataset(Dataset):
    """
    Full-image test dataset.
    Returns (noisy, clean) tensors for a full image.
    """

    def __init__(self, image_paths: list[Path], sigma: int, grayscale: bool = True):
        self.paths = image_paths
        self.sigma = sigma
        self.grayscale = grayscale
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.paths[idx])
        img = img.convert("L") if self.grayscale else img.convert("RGB")
        clean = self.to_tensor(img)
        noisy = add_gaussian_noise(clean, self.sigma).clamp(0.0, 1.0)
        return noisy, clean


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def build_train_loader(data_params: DataParams, train_params: TrainParams) -> DataLoader:
    paths: list[Path] = []

    for name in data_params.train_datasets:
        if name == "bsd400":
            for split in ["train", "test"]:
                d = _bsds_split_dir(data_params.data_root, split)
                if not d.exists():
                    raise RuntimeError(f"BSDS500 split '{split}' not found at {d}")
                paths.extend(_load_image_paths(d))
        elif name == "foe400":
            d = Path(data_params.data_root) / "foe400"
            if not d.exists():
                print(f"⚠️  FOE400 not found at {d}. Skipping...")
                continue
            paths.extend(_load_image_paths(d))
        elif name == "cbsd68":
            d = Path(data_params.data_root) / "cbsd68"
            if not d.exists():
                print(f"⚠️  CBSD68 not found at {d}. Skipping...")
                continue
            paths.extend(_load_image_paths(d))
        elif name == "flickr30k":
            d = Path(data_params.data_root) / "flickr30k"
            if not d.exists():
                print(f"⚠️  Flickr30k not found at {d}. Skipping...")
                continue
            paths.extend(_load_image_paths(d))
        else:
            raise ValueError(f"Unknown train dataset: '{name}'")

    if not paths:
        if data_params.allow_synthetic:
            print("📊 No real images found. Using synthetic training data...")
            dataset = SyntheticDenoisingDataset(
                sigma=train_params.sigma,
                patch_size=train_params.patch_size,
                num_images=100,
                patches_per_image=train_params.batch_size,
            )
            return DataLoader(
                dataset,
                batch_size=train_params.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
        raise FileNotFoundError(
            f"No training images found. Available datasets: {data_params.train_datasets}\n"
            "Run:  uv run main.py download --dataset bsd\n"
            "Or:   uv run main.py train --synthetic"
        )

    dataset = DenoisingDataset(
        image_paths=paths,
        sigma=train_params.sigma,
        patch_size=train_params.patch_size,
        augment=data_params.augment,
        grayscale=data_params.grayscale,
    )
    return DataLoader(
        dataset,
        batch_size=train_params.batch_size,
        shuffle=True,
        num_workers=train_params.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_test_loader(name: str, data_params: DataParams, sigma: int) -> DataLoader | None:
    """Build test loader, returns None if dataset not found."""
    if name == "bsd68":
        d = _bsds_split_dir(data_params.data_root, "test")
        if not d.exists():
            print(f"⚠️  BSD68 not found at {d}. Skipping...")
            return None
        paths = _load_image_paths(d)
        grayscale = True

    elif name == "cbsd68":
        d = Path(data_params.data_root) / "cbsd68"
        if not d.exists():
            print(f"⚠️  CBSD68 not found at {d}. Skipping...")
            return None
        paths = _load_image_paths(d)
        grayscale = data_params.grayscale

    elif name == "flickr30k":
        d = Path(data_params.data_root) / "flickr30k"
        if not d.exists():
            print(f"⚠️  Flickr30k not found at {d}. Skipping...")
            return None
        paths = _load_image_paths(d)
        grayscale = data_params.grayscale

    else:
        raise ValueError(f"Unknown test dataset: '{name}'")

    return DataLoader(
        TestDataset(paths, sigma=sigma, grayscale=grayscale),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        )
