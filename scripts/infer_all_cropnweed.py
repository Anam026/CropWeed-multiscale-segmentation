"""
infer_all_cropnweed.py — Inference on CropAndWeed images
Usage:
    python scripts/infer_all_cropnweed.py \
        --checkpoint outputs/cropandweed/checkpoints/best.pth \
        --config configs/config_cropnweed.yaml \
        --split test \
        --max_images 50 \
        --summary
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

# BUG 1 FIXED: was importing from dataset_caw (wrong file)
from src.data.dataset_cropnweed import CropAndWeedDataset
from src.data.transforms import get_val_transforms
from src.models.segmentation_model import build_model

CLASS_COLORS_RGB = {
    0: (20,  20,  20),   # background
    1: (0,  200,   0),   # crop - green
    2: (200,  0,   0),   # weed - red
}
CLASS_NAMES = ["background", "crop", "weed"]


def mask_to_color(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS_RGB.items():
        out[mask == c] = col
    return out


def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1,2,0).numpy()
    return (np.clip(img*std+mean, 0, 1)*255).astype(np.uint8)


def add_legend(panel_rgb, w):
    leg = np.zeros((36, w, 3), dtype=np.uint8)
    x = 12
    for cid, name in enumerate(CLASS_NAMES):
        r, g, b = CLASS_COLORS_RGB[cid]
        cv2.rectangle(leg, (x,7), (x+20,27), (r,g,b), -1)
        cv2.putText(leg, name, (x+26,22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (210,210,210), 1, cv2.LINE_AA)
        x += 130
    return np.vstack([panel_rgb, leg])


def make_panel(img, gt, pred, edge):
    gt_c   = mask_to_color(gt)
    pred_c = mask_to_color(pred)
    edge_v = (np.stack([edge]*3, -1)*255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.55, pred_c, 0.45, 0)
    panels  = [img, gt_c, pred_c, edge_v, overlay]
    labels  = ["Original","Ground truth","Prediction","Edge map","Overlay"]
    for p, l in zip(panels, labels):
        cv2.putText(p, l, (8,24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255,255,255), 2, cv2.LINE_AA)
    combined = np.concatenate(panels, axis=1)
    return add_legend(combined, combined.shape[1])


def compute_miou(pred, gt, nc=3, ignore=255):
    ious = []
    for c in range(nc):
        v = gt != ignore
        inter = ((pred==c)&(gt==c)&v).sum()
        union = ((pred==c)|(gt==c))&v
        u = union.sum()
        if u > 0: ious.append(inter/u)
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config_cropnweed.yaml")
    parser.add_argument("--split",      default="test",          # BUG 2 FIXED: default was "all"
                        choices=["train","val","test","all"])
    parser.add_argument("--save_dir",
                        default="outputs/cropandweed/visualizations/all_predictions")
    parser.add_argument("--summary",    action="store_true")
    parser.add_argument("--max_summary", default=20, type=int)
    parser.add_argument("--max_images",  default=50, type=int,   # BUG 2 FIXED: new arg to limit saved images
                        help="Max individual panel images to save per split (default 50)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    print("Loading model...")
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}, mIoU={ckpt.get('best_miou',0):.4f}")

    transform = get_val_transforms(ds_cfg["image_size"], cfg.get("augmentation",{}))
    splits    = ["train","val","test"] if args.split=="all" else [args.split]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_pairs = []
    total = 0

    for split in splits:
        ds = CropAndWeedDataset(
            root_dir=ds_cfg["raw_dir"], split=split,
            image_size=ds_cfg["image_size"], transform=transform,
            variant=ds_cfg.get("variant","CropsOrWeed9"),  # BUG 3 FIXED: was "CropsOrWeed" (missing "9")
        )
        split_dir = save_dir / split
        split_dir.mkdir(exist_ok=True)
        print(f"\n{split}: {len(ds)} images (saving max {args.max_images})")

        for i in tqdm(range(len(ds)), desc=split):
            s = ds[i]
            with torch.no_grad():
                out = model(s["image"].unsqueeze(0))
            pred = out["seg"].argmax(1).squeeze(0).cpu().numpy()
            edge = torch.sigmoid(out["edge"]).squeeze().cpu().numpy()
            img  = denorm(s["image"])
            gt   = s["mask"].numpy()

            # Only save up to max_images individual panels
            if total < args.max_images:
                panel = make_panel(img, gt, pred, edge)
                cv2.imwrite(str(split_dir / f"{s['stem']}_result.png"),
                            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            total += 1

            if args.summary and len(summary_pairs) < args.max_summary:
                miou = compute_miou(pred, gt)
                si   = cv2.resize(img, (224,224))
                sp   = cv2.resize(mask_to_color(pred),(224,224),interpolation=cv2.INTER_NEAREST)
                sg   = cv2.resize(mask_to_color(gt),  (224,224),interpolation=cv2.INTER_NEAREST)
                ov   = cv2.addWeighted(si, 0.55, sp, 0.45, 0)
                row  = np.concatenate([si, sg, sp, ov], axis=1)
                cv2.putText(row, f"mIoU={miou:.3f} {s['stem'][-12:]}",
                            (4,18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                summary_pairs.append(row)

    print(f"\nSaved {min(total, args.max_images)} panel images → {save_dir.absolute()}")
    print(f"Ran inference on {total} total images")

    if args.summary and summary_pairs:
        cols = 2
        rows = (len(summary_pairs)+cols-1)//cols
        while len(summary_pairs) < rows*cols:
            summary_pairs.append(np.zeros_like(summary_pairs[0]))
        grid = np.concatenate(
            [np.concatenate(summary_pairs[r*cols:(r+1)*cols], axis=1)
             for r in range(rows)], axis=0)

        tb = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(tb, "CropAndWeed Segmentation Results  |  Original  GT  Prediction  Overlay",
                    (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 1)
        lb = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
        x = 12
        for cid, name in enumerate(CLASS_NAMES):
            r,g,b = CLASS_COLORS_RGB[cid]
            cv2.rectangle(lb,(x,7),(x+20,27),(r,g,b),-1)
            cv2.putText(lb,name,(x+26,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
            x += 130
        grid = np.vstack([tb, grid, lb])

        out_path = "outputs/cropandweed/visualizations/summary_grid.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Summary grid → {out_path}")


if __name__ == "__main__":
    main()