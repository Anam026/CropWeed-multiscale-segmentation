"""
train_cropnweed.py — Training for CropAndWeed dataset
Usage:
    python scripts/train_cropnweed.py --config configs/config_cropnweed.yaml
"""
import argparse, logging, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset_caw import CropAndWeedDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.segmentation_model import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer, build_optimizer, build_scheduler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config_caw.yaml")
    parser.add_argument("--resume",  default=None)
    parser.add_argument("--device",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("project", {}).get("seed", 42))

    device_str = args.device or cfg.get("project", {}).get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})

    train_ds = CropAndWeedDataset(
        root_dir=ds_cfg["raw_dir"], split="train",
        image_size=ds_cfg["image_size"],
        transform=get_train_transforms(ds_cfg["image_size"], aug_cfg),
        variant=ds_cfg.get("variant", "CropsOrWeed"),
    )
    val_ds = CropAndWeedDataset(
        root_dir=ds_cfg["raw_dir"], split="val",
        image_size=ds_cfg["image_size"],
        transform=get_val_transforms(ds_cfg["image_size"], aug_cfg),
        variant=ds_cfg.get("variant", "CropsOrWeed"),
    )
    logger.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    bs  = cfg["training"].get("batch_size", 2)
    nw  = cfg.get("project", {}).get("num_workers", 0)
    pin = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)

    logger.info("Building model...")
    model     = build_model(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Resumed from {args.resume}")

    trainer = Trainer(model=model, train_loader=train_loader,
                      val_loader=val_loader, criterion=criterion,
                      optimizer=optimizer, scheduler=scheduler,
                      cfg=cfg, device=device)
    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()