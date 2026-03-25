"""
prepare_dataset.py
==================
Downloads the WeedsGalore dataset and prepares the processed split directories.

Usage:
    python scripts/prepare_dataset.py \
        --dataset_url https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip \
        --output_dir data/raw
"""

import argparse
import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            bar.update(size)
    print(f"Saved to {dest}")


def extract_zip(zip_path: Path, dest_dir: Path):
    print(f"Extracting {zip_path.name} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for m in tqdm(members, desc="Extracting"):
            zf.extract(m, dest_dir)
    print("Extraction complete.")


def verify_structure(dataset_dir: Path):
    """Print a summary of what was extracted."""
    print("\nDataset structure:")
    date_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("2023")])
    total_images = 0
    total_masks  = 0
    for d in date_dirs:
        imgs  = list((d / "images").glob("*_R.png")) if (d / "images").exists() else []
        masks = list((d / "semantics").glob("*.png")) if (d / "semantics").exists() else []
        print(f"  {d.name}: {len(imgs)} image stems, {len(masks)} masks")
        total_images += len(imgs)
        total_masks  += len(masks)
    print(f"\nTotal: {total_images} image samples, {total_masks} masks")

    # Check for official splits
    splits_dir = dataset_dir / "splits"
    if splits_dir.exists():
        for f in splits_dir.glob("*.txt"):
            lines = f.read_text().strip().splitlines()
            print(f"  splits/{f.name}: {len(lines)} entries")
    else:
        print("\n  No official splits/ directory found — will use random 70/15/15 split.")


def main():
    parser = argparse.ArgumentParser(description="Prepare WeedsGalore dataset")
    parser.add_argument(
        "--dataset_url",
        default="https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip",
        help="URL of the WeedsGalore zip archive",
    )
    parser.add_argument("--output_dir", default="data/raw", help="Where to save raw data")
    parser.add_argument("--skip_download", action="store_true", help="Skip download if zip already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_path   = output_dir / "weedsgalore-dataset.zip"

    # 1. Download
    if not zip_path.exists() or not args.skip_download:
        download_file(args.dataset_url, zip_path)
    else:
        print(f"Skipping download — {zip_path} already exists.")

    # 2. Extract
    extract_zip(zip_path, output_dir)

    # 3. Find the extracted directory
    dataset_dir = output_dir / "weedsgalore-dataset"
    if not dataset_dir.exists():
        # Some zips nest differently — try to find it
        candidates = list(output_dir.glob("weedsgalore*"))
        candidates = [c for c in candidates if c.is_dir()]
        if candidates:
            dataset_dir = candidates[0]
        else:
            print("ERROR: Could not locate extracted dataset directory.")
            sys.exit(1)

    # 4. Verify
    verify_structure(dataset_dir)
    print(f"\nDataset ready at: {dataset_dir}")
    print("\nNext step:")
    print(f"  python scripts/train.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
