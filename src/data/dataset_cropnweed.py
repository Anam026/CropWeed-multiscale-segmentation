"""
CropAndWeed Dataset Loader 
------------------------------------
Dataset structure:
  data/raw/cropandweed-dataset/data/
  ├── images/          ← RGB images (.jpg)
  ├── labelIds/
  │   └── CropsOrWeed9/  ← semantic masks (.png), pixel = class ID
  └── splits/
      ├── train.txt
      ├── val.txt
      └── test.txt

CropsOrWeed9 class mapping → our 3 classes:
    0        = background
    1        = crop
    2 to 8   = weed species → all mapped to 2
    255      = ignore
"""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CropAndWeedDataset(Dataset):

    CLASS_NAMES  = ["background", "crop", "weed"]
    NUM_CLASSES  = 3
    IGNORE_INDEX = 255

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 512,
        transform=None,
        variant: str = "CropsOrWeed9",
    ):
        super().__init__()
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.image_size = image_size
        self.transform  = transform
        self.variant    = variant

        self.img_dir  = self.root_dir / "images"
        self.mask_dir = self.root_dir / "labelIds" / variant

        self.samples: List[dict] = []
        self._load_samples()

    def _load_samples(self):
        splits_dir = self.root_dir / "splits"
        split_file = splits_dir / f"{self.split}.txt"

        if not split_file.exists():
            print(f"[CropAndWeed] No splits/{self.split}.txt — scanning all images.")
            self._split_randomly()
            return

        stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

        for stem in stems:
            # Try all image extensions
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = self.img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            mask_path = self.mask_dir / f"{stem}.png"

            if img_path is not None and mask_path.exists():
                self.samples.append({"img": img_path, "mask": mask_path, "stem": stem})
            else:
                # Only print first few missing to avoid spam
                if len([s for s in self.samples]) < 5:
                    print(f"[CropAndWeed] Skipping missing: {stem}")

        print(f"[CropAndWeed] {self.split}: {len(self.samples)} samples loaded "
              f"(from {len(stems)} in split file)")

    def _split_randomly(self, train=0.70, val=0.15):
        import random
        all_stems = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img in sorted(self.img_dir.glob(ext)):
                mask = self.mask_dir / f"{img.stem}.png"
                if mask.exists():
                    all_stems.append({"img": img, "mask": mask, "stem": img.stem})

        random.seed(42); random.shuffle(all_stems)
        n = len(all_stems)
        n_train = int(n * train); n_val = int(n * val)

        if self.split == "train":   self.samples = all_stems[:n_train]
        elif self.split == "val":   self.samples = all_stems[n_train:n_train+n_val]
        else:                       self.samples = all_stems[n_train+n_val:]
        print(f"[CropAndWeed] {self.split}: {len(self.samples)} samples (random split)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load image
        img = cv2.imread(str(s["img"]))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {s['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # Load mask and remap to 3 classes
        mask = cv2.imread(str(s["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {s['mask']}")
        mask = self._remap_mask(mask)

        if self.transform is not None:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return {"image": img, "mask": mask, "stem": s["stem"]}

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        CropsOrWeed9 has pixel values 0-8 (+ 255 ignore).
        Map to: 0=background, 1=crop, 2=weed, 255=ignore
        Exact mapping depends on dataset — we use:
            0 → 0  (background)
            1 → 1  (crop)
            2-8 → 2 (weed species)
            255 → 255 (ignore)
        """
        out = np.zeros_like(mask, dtype=np.uint8)
        out[mask == 0]   = 0    # background
        out[mask == 1]   = 1    # crop
        out[(mask >= 2) & (mask < 255)] = 2   # all weed classes → weed
        out[mask == 255] = 255  # ignore
        return out

    def __repr__(self):
        return (f"CropAndWeedDataset(split={self.split}, "
                f"n={len(self)}, variant={self.variant})")