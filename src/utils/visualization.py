"""
Visualization utilities for segmentation results.
"""

from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


# Class colors (BGR for cv2, RGB for matplotlib)
CLASS_COLORS_RGB = np.array([
    [0,   0,   0],    # 0 = background (black)
    [0, 200,   0],    # 1 = crop (green)
    [200, 0,   0],    # 2 = weed (red)
], dtype=np.uint8)

CLASS_NAMES = ["background", "crop", "weed"]


def mask_to_color(mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Convert (H, W) integer mask to (H, W, 3) RGB color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS_RGB):
        color[mask == c] = col
    return color


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a color-coded mask onto an RGB image.
    Both arrays should be (H, W, 3) uint8.
    """
    color_mask = mask_to_color(mask)
    blended = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return blended


def denormalize(tensor: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Convert a normalized (3, H, W) tensor to (H, W, 3) uint8 numpy array."""
    t = tensor.cpu().float()
    for i, (m, s) in enumerate(zip(mean, std)):
        t[i] = t[i] * s + m
    t = t.permute(1, 2, 0).numpy()
    t = (t * 255).clip(0, 255).astype(np.uint8)
    return t


def save_prediction_grid(
    images: torch.Tensor,     # (B, 3, H, W)
    masks: torch.Tensor,      # (B, H, W) GT
    preds: torch.Tensor,      # (B, H, W) predicted
    save_path: str,
    max_samples: int = 4,
):
    """Save a grid: original | GT mask | prediction | overlay."""
    B = min(images.shape[0], max_samples)
    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]

    for i in range(B):
        img_np  = denormalize(images[i])
        gt_np   = masks[i].cpu().numpy()
        pred_np = preds[i].cpu().numpy()

        axes[i, 0].imshow(img_np);               axes[i, 0].set_title("Image")
        axes[i, 1].imshow(mask_to_color(gt_np)); axes[i, 1].set_title("GT Mask")
        axes[i, 2].imshow(mask_to_color(pred_np)); axes[i, 2].set_title("Prediction")
        axes[i, 3].imshow(overlay_mask(img_np, pred_np, alpha=0.5)); axes[i, 3].set_title("Overlay")

        for ax in axes[i]:
            ax.axis("off")

    # Legend
    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c]) / 255, label=CLASS_NAMES[c])
               for c in range(len(CLASS_NAMES))]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curve(log_csv: str, save_path: Optional[str] = None):
    """Plot loss / mIoU training curves from a CSV log file."""
    import pandas as pd
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "train_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train_loss"], label="train loss")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

    if "val_miou" in df.columns:
        axes[1].plot(df["epoch"], df["val_miou"], color="green", label="val mIoU")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU"); axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
