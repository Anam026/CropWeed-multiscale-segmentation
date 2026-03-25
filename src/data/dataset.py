"""
WeedsGalore Dataset Loader
--------------------------
Dataset structure:
  weedsgalore-dataset/
  ├── 2023-05-25/
  │   ├── images/     ← 2023-05-25_NNNN_B/G/R/NIR/RE.png
  │   ├── semantics/  ← 2023-05-25_NNNN.png  (palette PNG: 0=bg, 1=crop, 2=weed)
  │   └── instances/
  ├── 2023-05-30/ ...
  ├── 2023-06-06/ ...
  ├── 2023-06-15/ ...
  └── splits/         ← train.txt / val.txt / test.txt  (image stem per line)
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class WeedsGaloreDataset(Dataset):
    """
    Loads RGB images + semantic masks from the WeedsGalore dataset.

    Args:
        root_dir   : path to weedsgalore-dataset/
        split      : 'train' | 'val' | 'test'
        image_size : square resize target
        transform  : albumentations Compose transform
        bands      : list of band suffixes to stack, e.g. ['R','G','B']
    """

    CLASS_NAMES = ["background", "crop", "weed"]
    NUM_CLASSES = 3
    # Semantic palette: 0=background, 1=crop, 2=weed (rest → ignore)
    IGNORE_INDEX = 255

    DATE_DIRS = ["2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        transform=None,
        bands: List[str] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.bands = bands or ["R", "G", "B"]

        self.samples: List[Tuple[Path, Path]] = []
        self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self):
        """
        Try to load from official splits/ text files first.
        Fall back to scanning all date directories if splits are absent.
        """
        splits_dir = self.root_dir / "splits"
        split_file = splits_dir / f"{self.split}.txt"

        if split_file.exists():
            self._load_from_split_file(split_file)
        else:
            print(f"[WeedsGalore] No splits/{self.split}.txt — scanning all dates.")
            all_samples = self._scan_all_dates()
            self._split_randomly(all_samples)

    def _load_from_split_file(self, split_file: Path):
        """Load samples listed in the official split text file."""
        with open(split_file) as f:
            stems = [l.strip() for l in f if l.strip()]

        for stem in stems:
            # stem format: 2023-05-25_0109
            date = stem[:10]
            date_dir = self.root_dir / date
            img_path  = date_dir / "images"  / f"{stem}_R.png"   # use R-band as proxy
            mask_path = date_dir / "semantics" / f"{stem}.png"

            # Compose the RGB triplet; confirm all three exist
            r = date_dir / "images" / f"{stem}_R.png"
            g = date_dir / "images" / f"{stem}_G.png"
            b = date_dir / "images" / f"{stem}_B.png"
            if r.exists() and g.exists() and b.exists() and mask_path.exists():
                self.samples.append((stem, date_dir, mask_path))
            else:
                print(f"[WeedsGalore] Skipping missing sample: {stem}")

    def _scan_all_dates(self) -> list:
        """Scan all date directories and return list of (stem, date_dir, mask_path)."""
        samples = []
        for date in self.DATE_DIRS:
            date_dir = self.root_dir / date
            if not date_dir.exists():
                continue
            img_dir  = date_dir / "images"
            mask_dir = date_dir / "semantics"
            if not img_dir.exists() or not mask_dir.exists():
                continue

            # Find unique image stems (each stem has 5 band files)
            r_files = sorted(img_dir.glob("*_R.png"))
            for r_file in r_files:
                stem = r_file.stem.replace("_R", "")
                g_file = img_dir / f"{stem}_G.png"
                b_file = img_dir / f"{stem}_B.png"
                mask   = mask_dir / f"{stem}.png"
                if g_file.exists() and b_file.exists() and mask.exists():
                    samples.append((stem, date_dir, mask))
        return samples

    def _split_randomly(self, all_samples: list, train=0.70, val=0.15):
        """Randomly assign samples to train/val/test."""
        import random
        random.seed(42)
        random.shuffle(all_samples)
        n = len(all_samples)
        n_train = int(n * train)
        n_val   = int(n * val)

        if self.split == "train":
            self.samples = all_samples[:n_train]
        elif self.split == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        else:
            self.samples = all_samples[n_train + n_val:]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        stem, date_dir, mask_path = self.samples[idx]

        # 1. Load RGB image from per-band PNG files
        image = self._load_rgb(date_dir / "images", stem)

        # 2. Load semantic mask (palette PNG → integer labels)
        mask = self._load_mask(mask_path)

        # 3. Apply augmentations (albumentations expects HWC numpy uint8)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        # 4. Convert to tensors if not already done by ToTensorV2
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,            # (C, H, W) float32
            "mask": mask,              # (H, W) int64
            "stem": stem,
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_rgb(self, img_dir: Path, stem: str) -> np.ndarray:
        """Load and stack R, G, B band PNGs into an HWC uint8 array."""
        channels = []
        for band in self.bands:
            band_path = img_dir / f"{stem}_{band}.png"
            ch = cv2.imread(str(band_path), cv2.IMREAD_GRAYSCALE)
            if ch is None:
                raise FileNotFoundError(f"Band image not found: {band_path}")
            channels.append(ch)

        # Stack → HWC
        if len(channels) == 1:
            rgb = np.stack([channels[0]] * 3, axis=-1)
        elif len(channels) == 3:
            # Bands are stored as R, G, B — OpenCV expects BGR
            rgb = np.stack(channels, axis=-1)  # HWC in R,G,B order → keep as is
        else:
            rgb = np.stack(channels, axis=-1)
        return rgb.astype(np.uint8)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load semantic mask.
        WeedsGalore semantics are palette PNGs where pixel value = class index.
        Values > NUM_CLASSES-1 are mapped to IGNORE_INDEX.
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Try reading as color and extract red channel (some datasets use RGB palette)
            mask_bgr = cv2.imread(str(mask_path))
            if mask_bgr is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask = mask_bgr[:, :, 2]  # red channel

        # Remap out-of-range values to ignore
        clean = mask.copy()
        clean[mask >= self.NUM_CLASSES] = self.IGNORE_INDEX
        return clean.astype(np.uint8)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights over training set.
        Useful for weighted loss.
        """
        counts = np.zeros(self.NUM_CLASSES, dtype=np.float64)
        for _, _, mask_path in self.samples:
            mask = self._load_mask(mask_path)
            for c in range(self.NUM_CLASSES):
                counts[c] += np.sum(mask == c)
        # Inverse frequency, normalized
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.NUM_CLASSES
        return torch.tensor(weights, dtype=torch.float32)

    def __repr__(self) -> str:
        return (
            f"WeedsGaloreDataset(split={self.split}, "
            f"n_samples={len(self)}, image_size={self.image_size})"
        )
