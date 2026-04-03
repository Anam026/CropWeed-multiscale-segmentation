"""
CropAndWeed Dataset Loader
---------------------------
Dataset structure after running setup.py:
  cropandweed-dataset/
  ├── images/
  │   ├── ave-0001_0001.png   ← RGB images
  │   └── vwg-0328_0001.png
  ├── labelIds/
  │   └── CropsOrWeed/        ← semantic masks (pixel = class ID)
  │       ├── ave-0001_0001.png
  │       └── ...
  └── splits/
      ├── train.txt
      ├── val.txt
      └── test.txt

Class mapping for CropsOrWeed variant:
  0 = background
  1 = crop
  2 = weed
  (higher IDs mapped to ignore=255)
"""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CropAndWeedDataset(Dataset):
    """
    Loads RGB images + semantic masks from the CropAndWeed dataset.

    Args:
        root_dir   : path to cropandweed-dataset/
        split      : 'train' | 'val' | 'test'
        image_size : square resize target
        transform  : albumentations Compose
        variant    : dataset variant folder name under labelIds/
    """

    CLASS_NAMES  = ["background", "crop", "weed"]
    NUM_CLASSES  = 3
    IGNORE_INDEX = 255

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        transform=None,
        variant: str = "CropsOrWeed",
    ):
        super().__init__()
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.image_size = image_size
        self.transform  = transform
        self.variant    = variant

        self.img_dir  = self.root_dir / "images"
        self.mask_dir = self.root_dir / "labelIds" / variant

        self.samples: List[str] = []
        self._load_samples()

    def _load_samples(self):
        splits_dir = self.root_dir / "splits"
        split_file = splits_dir / f"{self.split}.txt"

        if split_file.exists():
            with open(split_file) as f:
                stems = [l.strip() for l in f if l.strip()]
            for stem in stems:
                img  = self.img_dir  / f"{stem}.png"
                mask = self.mask_dir / f"{stem}.png"
                if img.exists() and mask.exists():
                    self.samples.append(stem)
                else:
                    print(f"[CropAndWeed] Skipping missing: {stem}")
        else:
            print(f"[CropAndWeed] No splits/{self.split}.txt — scanning all images.")
            self._split_randomly()

    def _split_randomly(self, train=0.70, val=0.15):
        import random
        all_stems = sorted([
            p.stem for p in self.img_dir.glob("*.png")
            if (self.mask_dir / f"{p.stem}.png").exists()
        ])
        random.seed(42)
        random.shuffle(all_stems)
        n = len(all_stems)
        n_train = int(n * train)
        n_val   = int(n * val)
        if self.split == "train":
            self.samples = all_stems[:n_train]
        elif self.split == "val":
            self.samples = all_stems[n_train:n_train + n_val]
        else:
            self.samples = all_stems[n_train + n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]

        # Load RGB image directly (not split into bands)
        img_path = self.img_dir / f"{stem}.png"
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # Load mask
        mask_path = self.mask_dir / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Map: 0=bg, 1=crop, 2=weed, rest→ignore
        clean = mask.copy().astype(np.uint8)
        clean[mask >= self.NUM_CLASSES] = self.IGNORE_INDEX

        if self.transform is not None:
            aug   = self.transform(image=image, mask=clean)
            image = aug["image"]
            clean = aug["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(clean, np.ndarray):
            clean = torch.from_numpy(clean).long()

        return {"image": image, "mask": clean, "stem": stem}

    def __repr__(self):
        return (f"CropAndWeedDataset(split={self.split}, "
                f"n={len(self)}, variant={self.variant})")