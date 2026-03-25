"""
train.py — Main training entry point
Usage:
    python scripts/train.py --config configs/config.yaml
"""
import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import WeedsGaloreDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.segmentation_model import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer, build_optimizer, build_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--resume",  default=None)
    parser.add_argument("--device",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("project", {}).get("seed", 42))

    # Device
    device_str = args.device or cfg.get("project", {}).get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})

    train_transform = get_train_transforms(ds_cfg["image_size"], aug_cfg)
    val_transform   = get_val_transforms(ds_cfg["image_size"], aug_cfg)

    train_ds = WeedsGaloreDataset(
        root_dir=ds_cfg["raw_dir"], split="train",
        image_size=ds_cfg["image_size"], transform=train_transform,
        bands=ds_cfg.get("bands", ["R", "G", "B"]),
    )
    val_ds = WeedsGaloreDataset(
        root_dir=ds_cfg["raw_dir"], split="val",
        image_size=ds_cfg["image_size"], transform=val_transform,
        bands=ds_cfg.get("bands", ["R", "G", "B"]),
    )
    logger.info(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    bs = cfg["training"].get("batch_size", 2)
    nw = cfg.get("project", {}).get("num_workers", 0)  # 0 workers on Windows CPU
    pin = device.type == "cuda"                         # only pin memory if GPU

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    logger.info("Building model...")
    model = build_model(cfg)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {total:.2f}M")

    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        cfg=cfg, device=device,
    )

    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()