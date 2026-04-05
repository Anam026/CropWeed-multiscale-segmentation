"""
infer.py — Single-image inference
===================================
Usage:
    python scripts/infer.py \
        --image path/to/image_R.png \
        --checkpoint outputs/weedsgalore/checkpoints/best.pth \
        --config configs/config.yaml \
        --save output.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch
import yaml

from src.models.segmentation_model import build_model
from src.data.transforms import get_test_transforms
from src.utils.visualization import mask_to_color, overlay_mask, denormalize


def load_rgb_from_band(image_path: str) -> np.ndarray:
    """
    Load RGB from a single path. If the path ends in _R.png, also loads _G.png and _B.png.
    Otherwise loads directly as a 3-channel image.
    """
    path = image_path
    if path.endswith("_R.png") or path.endswith("_R.PNG"):
        base = path[:-6]   # strip _R.png
        r = cv2.imread(base + "_R.png", cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(base + "_G.png", cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(base + "_B.png", cv2.IMREAD_GRAYSCALE)
        if r is None or g is None or b is None:
            raise FileNotFoundError(f"Could not find all bands for {base}")
        return np.stack([r, g, b], axis=-1)
    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True, help="Path to R-band image or RGB image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--save",       default="prediction.png", help="Output path")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or cfg.get("project", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    # Load image
    img_rgb = load_rgb_from_band(args.image)
    orig_h, orig_w = img_rgb.shape[:2]

    # Transform
    image_size = cfg["dataset"]["image_size"]
    transform  = get_test_transforms(image_size, cfg.get("augmentation", {}))
    augmented  = transform(image=img_rgb, mask=np.zeros(img_rgb.shape[:2], dtype=np.uint8))
    tensor     = augmented["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)

    pred_mask = outputs["seg"].argmax(dim=1).squeeze(0).cpu().numpy()
    edge_prob = torch.sigmoid(outputs["edge"]).squeeze().cpu().numpy()

    # Resize back to original resolution
    pred_mask_orig = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    edge_orig      = cv2.resize(edge_prob, (orig_w, orig_h))

    # Visualize
    color_pred = mask_to_color(pred_mask_orig)
    overlay    = overlay_mask(img_rgb, pred_mask_orig, alpha=0.5)

    # 4-panel: original | prediction | edge | overlay
    h, w = img_rgb.shape[:2]
    panel = np.zeros((h, w * 4, 3), dtype=np.uint8)
    panel[:, :w]         = img_rgb
    panel[:, w:2*w]      = color_pred
    panel[:, 2*w:3*w]    = (np.stack([edge_orig]*3, axis=-1) * 255).astype(np.uint8)
    panel[:, 3*w:]       = overlay

    # Labels
    for i, lbl in enumerate(["Original", "Prediction", "Edge Map", "Overlay"]):
        cv2.putText(panel, lbl, (i * w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save
    save_path = args.save
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"Saved prediction to: {save_path}")


if __name__ == "__main__":
    main()
