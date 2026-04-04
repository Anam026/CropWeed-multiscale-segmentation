"""
prepare_dataset_cropnweed.py
================================
Creates train/val/test splits for the CropAndWeed dataset.
Usage:
    python scripts/prepare_dataset_cropsnweeds.py \
        --output_dir data/raw/cropandweed-dataset/data \
        --skip_download
"""
import argparse, os, random, sys
from pathlib import Path


def create_splits(dataset_dir: Path, variant: str = "CropsOrWeed9",
                  train=0.70, val=0.15):
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    if (splits_dir / "train.txt").exists():
        print("  Splits already exist — skipping.")
        return

    img_dir  = dataset_dir / "images"
    mask_dir = dataset_dir / "labelIds" / variant

    if not img_dir.exists():
        print(f"  ERROR: images/ not found at {img_dir}"); sys.exit(1)
    if not mask_dir.exists():
        print(f"  ERROR: labelIds/{variant}/ not found at {mask_dir}")
        print(f"  Available variants:")
        lid = dataset_dir / "labelIds"
        if lid.exists():
            for d in sorted(lid.iterdir()):
                print(f"    {d.name}")
        sys.exit(1)

    # Images are .jpg, masks are .png
    stems = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for img_path in sorted(img_dir.glob(ext)):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                stems.append(img_path.stem)

    if not stems:
        print(f"  ERROR: No matched image-mask pairs found.")
        print(f"  Checked images: {img_dir}")
        print(f"  Checked masks:  {mask_dir}")
        n_imgs  = len(list(img_dir.glob("*.*")))
        n_masks = len(list(mask_dir.glob("*.png")))
        print(f"  Images found: {n_imgs}")
        print(f"  Masks found:  {n_masks}")
        # Show sample filenames to debug
        imgs  = list(img_dir.glob("*.*"))[:3]
        masks = list(mask_dir.glob("*.png"))[:3]
        print(f"  Sample images: {[p.name for p in imgs]}")
        print(f"  Sample masks:  {[p.name for p in masks]}")
        sys.exit(1)

    random.seed(42); random.shuffle(stems)
    n = len(stems)
    n_train = int(n * train); n_val = int(n * val)

    split_data = {
        "train": stems[:n_train],
        "val":   stems[n_train:n_train+n_val],
        "test":  stems[n_train+n_val:],
    }
    for name, items in split_data.items():
        (splits_dir / f"{name}.txt").write_text("\n".join(items))
        print(f"  {name}.txt: {len(items)} samples")

    print(f"\n  Total: {n} matched image-mask pairs")


def verify(dataset_dir: Path, variant: str = "CropsOrWeed9"):
    print("\nVerifying structure...")
    img_dir  = dataset_dir / "images"
    mask_dir = dataset_dir / "labelIds" / variant
    splits   = dataset_dir / "splits"

    n_img  = sum(1 for _ in img_dir.glob("*.*"))    if img_dir.exists()  else 0
    n_mask = sum(1 for _ in mask_dir.glob("*.png"))  if mask_dir.exists() else 0
    print(f"  images/                    : {n_img} files")
    print(f"  labelIds/{variant}/ : {n_mask} files")

    for split in ["train", "val", "test"]:
        f = splits / f"{split}.txt"
        if f.exists():
            n = len(f.read_text().strip().splitlines())
            print(f"  splits/{split}.txt          : {n} entries")

    print(f"\nDataset ready at: {dataset_dir.absolute()}")
    print(f"\nNext step:")
    print(f"  python scripts/train_crosnweed.py --config configs/config_cropnweed.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",    default="data/raw/cropandweed-dataset/data")
    parser.add_argument("--variant",       default="CropsOrWeed9")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Creating train/val/test splits for variant: {args.variant}")
    create_splits(out, variant=args.variant)
    verify(out, variant=args.variant)


if __name__ == "__main__":
    main()