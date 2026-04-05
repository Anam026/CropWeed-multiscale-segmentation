"""
CropAndWeed Dataset Loader
Dataset structure:
  data/raw/cropandweed-dataset/data/
  ├── images/          ← RGB images (.jpg)
  ├── labelIds/
  │   └── CropsOrWeed9/  ← semantic masks (.png), pixel = class ID
  └── splits/
      ├── train.txt
      ├── val.txt
      └── test.txt

CropsOrWeed9 ACTUAL class mapping (verified from pixel analysis):
    9        = background  → 0
    1        = crop        → 1
    3        = weed        → 2
    8        = weed        → 2
    0        = ignore      → 255
    255      = ignore      → 255
"""

from pathlib import Path
from typing import List
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
                if len(self.samples) < 5:
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

        random.seed(42)
        random.shuffle(all_stems)
        n = len(all_stems)
        n_train = int(n * train)
        n_val   = int(n * val)

        if self.split == "train":   self.samples = all_stems[:n_train]
        elif self.split == "val":   self.samples = all_stems[n_train:n_train+n_val]
        else:                       self.samples = all_stems[n_train+n_val:]
        print(f"[CropAndWeed] {self.split}: {len(self.samples)} samples (random split)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = cv2.imread(str(s["img"]))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {s['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        mask = cv2.imread(str(s["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {s['mask']}")
        mask = self._remap_mask(mask)

        if self.transform is not None:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return {"image": img, "mask": mask, "stem": s["stem"]}

    def _remap_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        VERIFIED mapping from pixel distribution analysis:
            9   → 0  (background, 97% of pixels)
            1   → 1  (crop,       0.32% of pixels)
            3   → 2  (weed,       0.09% of pixels)
            8   → 2  (weed,       2.3%  of pixels)
            0   → 255 (ignore/unlabelled)
            255 → 255 (ignore)
        """
        out = np.full_like(mask, 255, dtype=np.uint8)  # default = ignore
        out[mask == 9]   = 0    # background
        out[mask == 1]   = 1    # crop
        out[mask == 3]   = 2    # weed species
        out[mask == 8]   = 2    # weed species
        # mask == 0 and mask == 255 stay as 255 (ignore)
        return out

    def __repr__(self):
        return (f"CropAndWeedDataset(split={self.split}, "
                f"n={len(self)}, variant={self.variant})")