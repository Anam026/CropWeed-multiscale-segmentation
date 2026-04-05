"""
infer_all.py — Run inference on ALL images and save results
Usage:
    python scripts/infer_all.py --checkpoint outputs/weedsgalore/checkpoints/best.pth --config configs/config.yaml --split all --summary
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import WeedsGaloreDataset
from src.data.transforms import get_val_transforms
from src.models.segmentation_model import build_model

# Colors in RGB
# 0 = background → black
# 1 = crop       → green
# 2 = weed       → red
CLASS_COLORS_RGB = {
    0: (20,  20,  20),
    1: (0,  200,   0),
    2: (200,   0,   0),
}
CLASS_NAMES = ["background", "crop", "weed"]


def mask_to_color(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, col in CLASS_COLORS_RGB.items():
        out[mask == cls] = col
    return out


def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().permute(1,2,0).numpy()
    return (np.clip(img * std + mean, 0, 1) * 255).astype(np.uint8)


def draw_label(img, text, x=8, y=24, scale=0.65, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255,255,255), thickness, cv2.LINE_AA)
    return img


def add_legend(panel_rgb, panel_w):
    """
    panel_rgb is still in RGB at this point.
    We draw the legend in RGB directly — no BGR flip needed.
    """
    h = 36
    legend = np.zeros((h, panel_w, 3), dtype=np.uint8)
    x = 12
    for cls_id, name in enumerate(CLASS_NAMES):
        r, g, b = CLASS_COLORS_RGB[cls_id]
        # Draw filled box directly in RGB
        cv2.rectangle(legend, (x, 7), (x+20, 27), (r, g, b), -1)
        cv2.putText(legend, name, (x+26, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        x += 130
    return np.vstack([panel_rgb, legend])


def make_panel(img_rgb, gt_mask, pred_mask, edge_prob):
    gt_color   = mask_to_color(gt_mask)
    pred_color = mask_to_color(pred_mask)
    edge_3ch   = (np.stack([edge_prob]*3, axis=-1) * 255).astype(np.uint8)
    overlay    = cv2.addWeighted(img_rgb, 0.55, pred_color, 0.45, 0)

    panels = [img_rgb, gt_color, pred_color, edge_3ch, overlay]
    labels = ["Original", "Ground truth", "Prediction", "Edge map", "Overlay"]

    for panel, lbl in zip(panels, labels):
        draw_label(panel, lbl)

    combined = np.concatenate(panels, axis=1)
    combined = add_legend(combined, combined.shape[1])
    return combined


def compute_miou(pred, gt, num_classes=3, ignore=255):
    ious = []
    for c in range(num_classes):
        valid = gt != ignore
        p = (pred == c) & valid
        g = (gt   == c) & valid
        inter = (p & g).sum()
        union = (p | g).sum()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--split",      default="all",
                        choices=["train","val","test","all"])
    parser.add_argument("--save_dir",   default="outputs/weedsgalore/visualizations/all_predictions")
    parser.add_argument("--summary",    action="store_true")
    parser.add_argument("--max_summary",default=20, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    ds_cfg  = cfg["dataset"]

    print("Loading model...")
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}, best_mIoU={ckpt.get('best_miou',0):.4f}")

    transform = get_val_transforms(ds_cfg["image_size"], cfg.get("augmentation",{}))
    splits    = ["train","val","test"] if args.split == "all" else [args.split]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_pairs = []
    total_saved   = 0

    for split in splits:
        dataset = WeedsGaloreDataset(
            root_dir=ds_cfg["raw_dir"], split=split,
            image_size=ds_cfg["image_size"], transform=transform,
            bands=ds_cfg.get("bands",["R","G","B"]),
        )
        split_dir = save_dir / split
        split_dir.mkdir(exist_ok=True)
        print(f"\n{split} split — {len(dataset)} images")

        for i in tqdm(range(len(dataset)), desc=split):
            s    = dataset[i]
            with torch.no_grad():
                out = model(s["image"].unsqueeze(0))

            pred = out["seg"].argmax(1).squeeze(0).cpu().numpy()
            edge = torch.sigmoid(out["edge"]).squeeze().cpu().numpy()
            img  = denorm(s["image"])
            gt   = s["mask"].numpy()
            stem = s["stem"]

            # panel is RGB
            panel = make_panel(img, gt, pred, edge)

            # convert RGB → BGR only at save time
            out_path = split_dir / f"{stem}_result.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            total_saved += 1

            if args.summary and len(summary_pairs) < args.max_summary:
                miou       = compute_miou(pred, gt)
                small_img  = cv2.resize(img, (224,224))
                small_pred = cv2.resize(mask_to_color(pred), (224,224),
                                        interpolation=cv2.INTER_NEAREST)
                small_gt   = cv2.resize(mask_to_color(gt), (224,224),
                                        interpolation=cv2.INTER_NEAREST)
                ovly  = cv2.addWeighted(small_img, 0.55, small_pred, 0.45, 0)
                strip = np.concatenate([small_img, small_gt, small_pred, ovly], axis=1)
                draw_label(strip, f"mIoU={miou:.3f}  {stem[-12:]}", y=18, scale=0.45)
                summary_pairs.append(strip)

    print(f"\nSaved {total_saved} images → {save_dir.absolute()}")

    if args.summary and summary_pairs:
        cols = 2
        rows = (len(summary_pairs) + cols - 1) // cols
        while len(summary_pairs) < rows * cols:
            summary_pairs.append(np.zeros_like(summary_pairs[0]))

        grid_rows = [np.concatenate(summary_pairs[r*cols:(r+1)*cols], axis=1)
                     for r in range(rows)]
        grid = np.concatenate(grid_rows, axis=0)

        # Title bar
        title_bar = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar,
                    "Multiscale Crop-Weed Segmentation  |  Original  GT  Prediction  Overlay",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 1, cv2.LINE_AA)
        grid = np.vstack([title_bar, grid])

        # Leg end bar — drawn in RGB, flipped at save
        legend_bar = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
        x = 12
        for cls_id, name in enumerate(CLASS_NAMES):
            r, g, b = CLASS_COLORS_RGB[cls_id]
            cv2.rectangle(legend_bar, (x,7), (x+20,27), (r,g,b), -1)
            cv2.putText(legend_bar, name, (x+26,22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            x += 130
        grid = np.vstack([grid, legend_bar])

        summary_path = "outputs/visualizations/summary_grid.png"
        cv2.imwrite(summary_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Summary grid → {summary_path}")

    print(f"\nDone! Open: {save_dir.absolute()}")


if __name__ == "__main__":
    main()   