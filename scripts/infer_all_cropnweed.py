"""
infer_all_cropsnweeds.py — Inference on CropAndWeed images
Picks images with the MOST crop/weed pixels so visualizations
show actual plants rather than mostly background.

"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.data.dataset_cropnweed import CropAndWeedDataset
from src.data.transforms import get_val_transforms
from src.models.segmentation_model import build_model

CLASS_COLORS_RGB = {
    0: (20,  20,  20),
    1: (0,  200,   0),   # crop  - green
    2: (200,  0,   0),   # weed  - red
}
CLASS_NAMES = ["background", "crop", "weed"]


def mask_to_color(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS_RGB.items():
        out[mask == c] = col
    return out


def denorm(tensor):
    m = np.array([0.485, 0.456, 0.406])
    s = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().permute(1,2,0).numpy()
    return (np.clip(img*s+m, 0, 1)*255).astype(np.uint8)


def plant_pixel_ratio(mask):
    """Returns fraction of pixels that are crop or weed (not background)."""
    total = mask.size
    plant = ((mask == 1) | (mask == 2)).sum()
    return plant / total if total > 0 else 0.0


def add_legend(panel_rgb, panel_w):
    leg = np.zeros((36, panel_w, 3), dtype=np.uint8)
    x = 12
    for cid, name in enumerate(CLASS_NAMES):
        r, g, b = CLASS_COLORS_RGB[cid]
        cv2.rectangle(leg, (x,7), (x+20,27), (r,g,b), -1)
        cv2.putText(leg, name, (x+26,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210,210,210), 1, cv2.LINE_AA)
        x += 130
    return np.vstack([panel_rgb, leg])


def make_panel(img_rgb, gt_mask, pred_mask, edge_prob):
    gt_color   = mask_to_color(gt_mask)
    pred_color = mask_to_color(pred_mask)
    edge_3ch   = (np.stack([edge_prob]*3, axis=-1)*255).astype(np.uint8)
    overlay    = cv2.addWeighted(img_rgb, 0.55, pred_color, 0.45, 0)
    panels = [img_rgb, gt_color, pred_color, edge_3ch, overlay]
    labels = ["Original", "Ground truth", "Prediction", "Edge map", "Overlay"]
    for p, l in zip(panels, labels):
        cv2.putText(p, l, (8,24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255,255,255), 2, cv2.LINE_AA)
    combined = np.concatenate(panels, axis=1)
    return add_legend(combined, combined.shape[1])


def compute_miou(pred, gt, nc=3, ignore=255):
    ious = []
    for c in range(nc):
        v = gt != ignore
        inter = ((pred==c) & (gt==c) & v).sum()
        union = ((pred==c) | (gt==c)) & v
        if union.sum() > 0:
            ious.append(inter / union.sum())
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config_cropsnweeds.yaml")
    parser.add_argument("--split",      default="all",
                        choices=["train","val","test","all"])
    parser.add_argument("--save_dir",
                        default="outputs/cropandweed/visualizations/all_predictions")
    parser.add_argument("--summary",     action="store_true")
    parser.add_argument("--n_summary",   default=50, type=int,
                        help="Number of images in summary grid")
    parser.add_argument("--min_plant_ratio", default=0.05, type=float,
                        help="Min fraction of plant pixels to include in summary (0-1)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cpu")
    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})

    print("Loading model...")
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}, "
          f"best_mIoU={ckpt.get('best_miou',0):.4f}")

    transform = get_val_transforms(ds_cfg["image_size"], aug_cfg)
    splits    = ["train","val","test"] if args.split=="all" else [args.split]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results with their plant ratio for smart sampling
    all_results = []
    total_saved = 0

    for split in splits:
        dataset = CropAndWeedDataset(
            root_dir   = ds_cfg["raw_dir"],
            split      = split,
            image_size = ds_cfg["image_size"],
            transform  = transform,
            variant    = ds_cfg.get("variant", "CropsOrWeed9"),
        )
        split_dir = save_dir / split
        split_dir.mkdir(exist_ok=True)
        print(f"\n{split}: {len(dataset)} images")

        for i in tqdm(range(len(dataset)), desc=split):
            s    = dataset[i]
            img  = denorm(s["image"])
            gt   = s["mask"].numpy()
            stem = s["stem"]

            with torch.no_grad():
                out = model(s["image"].unsqueeze(0))

            pred = out["seg"].argmax(1).squeeze(0).cpu().numpy()
            edge = torch.sigmoid(out["edge"]).squeeze().cpu().numpy()

            # Save individual result
            panel    = make_panel(img, gt, pred, edge)
            out_path = split_dir / f"{stem}_result.png"
            cv2.imwrite(str(out_path),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            total_saved += 1

            # Compute plant ratio from GT mask for smart selection
            ratio = plant_pixel_ratio(gt)
            all_results.append({
                "img":   img,
                "gt":    gt,
                "pred":  pred,
                "stem":  stem,
                "ratio": ratio,
                "miou":  compute_miou(pred, gt),
            })

    print(f"\nSaved {total_saved} individual images → {save_dir.absolute()}")

    # ── Smart summary grid ────────────────────────────────────────────────────
    if args.summary and all_results:
        print(f"\nSelecting top {args.n_summary} plant-rich images "
              f"(min plant ratio: {args.min_plant_ratio:.0%})...")

        # Filter to images with enough plant pixels
        plant_rich = [r for r in all_results
                      if r["ratio"] >= args.min_plant_ratio]

        print(f"  {len(plant_rich)} images have ≥{args.min_plant_ratio:.0%} plant pixels")

        # Sort by plant pixel ratio descending, pick top n_summary
        plant_rich.sort(key=lambda r: r["ratio"], reverse=True)
        picks = plant_rich[:args.n_summary]

        # If not enough plant-rich images, pad with best remaining
        if len(picks) < args.n_summary:
            remaining = [r for r in all_results if r not in picks]
            remaining.sort(key=lambda r: r["ratio"], reverse=True)
            picks += remaining[:args.n_summary - len(picks)]

        print(f"  Selected {len(picks)} images")
        print(f"  Plant ratio range: "
              f"{picks[-1]['ratio']:.1%} – {picks[0]['ratio']:.1%}")

        # Build summary grid: each row = Original | GT | Prediction | Overlay
        summary_rows = []
        for r in picks:
            small_img  = cv2.resize(r["img"],            (224,224))
            small_gt   = cv2.resize(mask_to_color(r["gt"]),   (224,224),
                                    interpolation=cv2.INTER_NEAREST)
            small_pred = cv2.resize(mask_to_color(r["pred"]), (224,224),
                                    interpolation=cv2.INTER_NEAREST)
            ovly = cv2.addWeighted(small_img, 0.55, small_pred, 0.45, 0)
            row  = np.concatenate([small_img, small_gt, small_pred, ovly], axis=1)

            # Label: stem + plant % + mIoU
            label = (f"{r['stem'][-14:]}  "
                     f"plant:{r['ratio']:.0%}  "
                     f"mIoU:{r['miou']:.2f}")
            cv2.putText(row, label, (4,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
            summary_rows.append(row)

        # Arrange into 2-column grid
        cols = 2
        rows = (len(summary_rows) + cols - 1) // cols
        while len(summary_rows) < rows * cols:
            summary_rows.append(np.zeros_like(summary_rows[0]))

        grid_rows = []
        for r in range(rows):
            grid_rows.append(
                np.concatenate(summary_rows[r*cols:(r+1)*cols], axis=1))
        grid = np.concatenate(grid_rows, axis=0)

        # Title bar
        tb = np.zeros((55, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(tb,
                    f"CropAndWeed Segmentation — Top {len(picks)} plant-rich images  "
                    f"|  Original  GT  Prediction  Overlay",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

        # Legend bar
        lb = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
        x  = 12
        for cid, name in enumerate(CLASS_NAMES):
            r, g, b = CLASS_COLORS_RGB[cid]
            cv2.rectangle(lb, (x,7), (x+20,27), (r,g,b), -1)
            cv2.putText(lb, name, (x+26,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            x += 130

        grid = np.vstack([tb, grid, lb])

        out_path = "outputs/cropandweed/visualizations/summary_grid.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSummary grid → {out_path}")
        print(f"All images show ≥{args.min_plant_ratio:.0%} plant pixels")

    print(f"\nDone! Open: {save_dir.absolute()}")


if __name__ == "__main__":
    main()