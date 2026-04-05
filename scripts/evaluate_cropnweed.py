"""
evaluate_cropsnweeds.py — Evaluation for CropAndWeed dataset
Usage:
    python scripts/evaluate_cropsnweeds.py \
        --config configs/config_cropnweed.yaml \
        --checkpoint outputs/cropandweed/checkpoints/best.pth \
        --split test
"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch, yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset_cropnweed import CropAndWeedDataset
from src.data.transforms import get_val_transforms
from src.models.segmentation_model import build_model
from src.evaluation.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config_cropnweed.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test",
                        choices=["train","val","test"])
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or cfg.get("project",{}).get("device","cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})

    dataset = CropAndWeedDataset(
        root_dir   = ds_cfg["raw_dir"],
        split      = args.split,
        image_size = ds_cfg["image_size"],
        transform  = get_val_transforms(ds_cfg["image_size"], aug_cfg),
        variant    = ds_cfg.get("variant", "CropsOrWeed9"),
    )
    logger.info(f"Evaluating {args.split}: {len(dataset)} samples")

    if len(dataset) == 0:
        logger.error("0 samples loaded! Check raw_dir and variant in config.")
        logger.error(f"raw_dir: {ds_cfg['raw_dir']}")
        logger.error(f"variant: {ds_cfg.get('variant','CropsOrWeed9')}")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        num_workers=0, pin_memory=False)

    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    logger.info(f"Loaded: epoch={ckpt.get('epoch','?')}, "
                f"best_mIoU={ckpt.get('best_miou',0):.4f}")

    metrics = SegmentationMetrics(
        num_classes  = ds_cfg["num_classes"],
        ignore_index = ds_cfg.get("ignore_index", 255),
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks  = batch["mask"].long().to(device)
            preds  = model(images)["seg"].argmax(dim=1)
            metrics.update(preds, masks)

    metrics.print_summary(
        class_names=ds_cfg.get("class_names", ["background","crop","weed"]))
    r = metrics.compute()
    logger.info(f"Final mIoU:  {r['miou']:.4f}")
    logger.info(f"Final mDice: {r['mdice']:.4f}")



if __name__ == "__main__":
    main()