"""
evaluate.py — Full evaluation on test split
============================================
Usage:
    python scripts/evaluate.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pth \
        --split test
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import WeedsGaloreDataset
from src.data.transforms import get_val_transforms
from src.models.segmentation_model import build_model
from src.evaluation.metrics import SegmentationMetrics
from src.utils.visualization import save_prediction_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_preds", action="store_true", help="Save prediction grids")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or cfg.get("project", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})

    dataset = WeedsGaloreDataset(
        root_dir   = ds_cfg["raw_dir"],
        split      = args.split,
        image_size = ds_cfg["image_size"],
        transform  = get_val_transforms(ds_cfg["image_size"], aug_cfg),
        bands      = ds_cfg.get("bands", ["R", "G", "B"]),
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Evaluating on {args.split} split: {len(dataset)} samples")

    # Model
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    metrics = SegmentationMetrics(
        num_classes  = ds_cfg["num_classes"],
        ignore_index = ds_cfg.get("ignore_index", 255),
    )

    viz_dir = cfg.get("output", {}).get("viz_dir", "outputs/visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)

            outputs = model(images)
            preds   = outputs["seg"].argmax(dim=1)

            metrics.update(preds, masks)

            if args.save_preds and batch_idx < 5:
                save_prediction_grid(
                    images, masks, preds,
                    save_path=f"{viz_dir}/{args.split}_batch{batch_idx:03d}.png",
                )

    metrics.print_summary(class_names=ds_cfg.get("class_names", ["bg", "crop", "weed"]))
    results = metrics.compute()
    logger.info(f"\nFinal mIoU:  {results['miou']:.4f}")
    logger.info(f"Final mDice: {results['mdice']:.4f}")


if __name__ == "__main__":
    main()
