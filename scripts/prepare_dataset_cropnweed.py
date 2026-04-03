"""
prepare_dataset_caw.py
=======================
Downloads and prepares the CropAndWeed dataset.

Steps:
  1. Clones the GitHub repo (scripts only)
  2. Runs their setup.py to download images + annotations
  3. Creates splits/train.txt, val.txt, test.txt
  4. Verifies structure

Usage:
    python scripts/prepare_dataset_caw.py --output_dir data/raw/cropandweed-dataset
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"  ERROR: command failed with code {result.returncode}")
        sys.exit(1)


def create_splits(dataset_dir: Path, train=0.70, val=0.15):
    """Create train/val/test split text files."""
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # Check if official splits already exist
    if (splits_dir / "train.txt").exists():
        print("  Official splits already exist — skipping.")
        return

    img_dir   = dataset_dir / "images"
    mask_dir  = dataset_dir / "labelIds" / "CropsOrWeed"

    if not img_dir.exists():
        print(f"  ERROR: images/ folder not found at {img_dir}")
        sys.exit(1)
    if not mask_dir.exists():
        print(f"  WARNING: labelIds/CropsOrWeed/ not found — trying labelIds/")
        mask_dir = dataset_dir / "labelIds"

    # Get all stems that have both image and mask
    stems = sorted([
        p.stem for p in img_dir.glob("*.png")
        if (mask_dir / f"{p.stem}.png").exists()
    ])

    if not stems:
        print("  WARNING: No matched image-mask pairs found.")
        print("  Make sure setup.py ran successfully and labelIds/CropsOrWeed/ exists.")
        return

    random.seed(42)
    random.shuffle(stems)
    n       = len(stems)
    n_train = int(n * train)
    n_val   = int(n * val)

    splits = {
        "train": stems[:n_train],
        "val":   stems[n_train:n_train + n_val],
        "test":  stems[n_train + n_val:],
    }

    for name, items in splits.items():
        path = splits_dir / f"{name}.txt"
        path.write_text("\n".join(items))
        print(f"  {name}.txt: {len(items)} samples")

    print(f"\n  Total: {n} samples split into train/val/test")


def verify(dataset_dir: Path):
    print("\nVerifying structure...")
    img_dir  = dataset_dir / "images"
    mask_dir = dataset_dir / "labelIds" / "CropsOrWeed"
    splits   = dataset_dir / "splits"

    n_img   = len(list(img_dir.glob("*.png")))  if img_dir.exists()  else 0
    n_mask  = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0

    print(f"  images/              : {n_img} files")
    print(f"  labelIds/CropsOrWeed/: {n_mask} files")

    for split in ["train", "val", "test"]:
        f = splits / f"{split}.txt"
        if f.exists():
            print(f"  splits/{split}.txt   : {len(f.read_text().strip().splitlines())} entries")

    if n_img == 0:
        print("\n  Dataset not downloaded yet. Run setup.py manually:")
        print(f"  cd {dataset_dir}")
        print(f"  python setup.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw/cropandweed-dataset",
                        help="Where to save the dataset")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip git clone + setup.py (if already downloaded)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("Step 1: Cloning CropAndWeed repository...")
        repo_scripts = out / "cropandweed-dataset-scripts"
        if not repo_scripts.exists():
            run(f'git clone https://github.com/cropandweed/cropandweed-dataset "{repo_scripts}"')
        else:
            print("  Repo already cloned.")

        print("\nStep 2: Running setup.py to download dataset...")
        print("  NOTE: This downloads ~8k images. May take 10-30 minutes.")
        run("python setup.py", cwd=str(repo_scripts))

        # Copy downloaded data to our expected location
        downloaded = repo_scripts
        print(f"\nStep 3: Dataset downloaded to {downloaded}")

        # Point to the downloaded location
        out = downloaded

    print("\nStep 3: Creating train/val/test splits...")
    create_splits(out)

    verify(out)

    print(f"\nDataset ready at: {out.absolute()}")
    print("\nUpdate configs/config_caw.yaml:")
    print(f'  raw_dir: "{out.absolute()}"')
    print("\nNext step:")
    print("  python scripts/train_caw.py --config configs/config_caw.yaml")


if __name__ == "__main__":
    main()