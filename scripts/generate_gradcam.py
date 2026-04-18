"""
generate_gradcam.py
====================
Generates GradCAM activation maps for the 3 MSFCA branches (k=5, k=11, k=17).

Produces publication-ready figure:
    Original | Prediction | k=5 heatmap | k=11 heatmap | k=17 heatmap

Usage (WeedsGalore):
    python scripts/generate_gradcam.py \
        --checkpoint outputs/checkpoints/best.pth \
        --config configs/config.yaml \
        --split test \
        --n_images 4

Usage (CropAndWeed):
    python scripts/generate_gradcam.py \
        --checkpoint outputs/cropandweed/checkpoints/best.pth \
        --config configs/config_cropsnweeds.yaml \
        --split test \
        --n_images 4 \
        --dataset cropandweed

Output:
    outputs/visualizations/results_report/gradcam_msfca.png
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm


# ── Helpers ────────────────────────────────────────────────────────────────────

def denorm(tensor):
    m = np.array([0.485, 0.456, 0.406])
    s = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    return (np.clip(img * s + m, 0, 1) * 255).astype(np.uint8)


def mask_to_color(mask):
    cmap = np.array([[20, 20, 20], [0, 200, 0], [200, 0, 0]], dtype=np.uint8)
    out  = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(cmap):
        out[mask == c] = col
    return out


def apply_heatmap(img_rgb, cam, alpha=0.55):
    """Overlay a GradCAM heatmap on an RGB image."""
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    cam_norm    = (cam_resized - cam_resized.min()) / (
        cam_resized.max() - cam_resized.min() + 1e-8)
    heatmap = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (img_rgb.astype(float) * (1 - alpha) +
               heatmap.astype(float) * alpha).astype(np.uint8)
    return blended, cam_norm


def plant_pixel_ratio(mask):
    total = mask.size
    plant = ((mask == 1) | (mask == 2)).sum()
    return plant / total if total > 0 else 0.0


# ── GradCAM Implementation ─────────────────────────────────────────────────────

class GradCAM:
    """
    GradCAM for a specific layer of the model.
    Hooks into the target layer, captures activations and gradients,
    then computes the weighted activation map.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._fwd_handle  = None
        self._bwd_handle  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Clone to decouple from the computation graph
            self.activations = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """Always call this when done to avoid hook accumulation."""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def compute(self, image_tensor, boxes=None, target_class=2):
        """
        Args:
            image_tensor : (1, 3, H, W) normalized input
            boxes        : None or (1, N, 4) for bbox pipeline
            target_class : 0=bg, 1=crop, 2=weed (default=weed)

        Returns:
            cam : (H', W') numpy float array — raw activation map
        """
        self.model.zero_grad()

        # Clone input to ensure no in-place ops affect the original tensor
        inp = image_tensor.detach().clone().requires_grad_(True)

        # Forward pass
        if boxes is not None:
            out = self.model(inp, boxes)
        else:
            out = self.model(inp)

        seg = out["seg"]  # (1, C, H, W)

        # Score = mean logit for target class
        score = seg[0, target_class].mean()
        score.backward()

        # Pool gradients over spatial dims
        grads   = self.gradients    # (1, C, H', W')
        acts    = self.activations  # (1, C, H', W')
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1).squeeze(0)    # (H', W')
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        return cam


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset(cfg, split, dataset_type):
    from src.data.transforms import get_val_transforms
    transform = get_val_transforms(
        cfg["dataset"]["image_size"], cfg.get("augmentation", {}))

    if dataset_type == "weedsgalore":
        from src.data.dataset import WeedsGaloreDataset
        return WeedsGaloreDataset(
            root_dir   = cfg["dataset"]["raw_dir"],
            split      = split,
            image_size = cfg["dataset"]["image_size"],
            transform  = transform,
            bands      = cfg["dataset"].get("bands", ["R", "G", "B"]),
        )
    else:
        from src.data.dataset_cropnweed import CropAndWeedDataset
        return CropAndWeedDataset(
            root_dir   = cfg["dataset"]["raw_dir"],
            split      = split,
            image_size = cfg["dataset"]["image_size"],
            transform  = transform,
            variant    = cfg["dataset"].get("variant", "CropsOrWeed9"),
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--split",        default="test")
    parser.add_argument("--dataset",      default="weedsgalore",
                        choices=["weedsgalore", "cropandweed"])
    parser.add_argument("--n_images",     default=4, type=int,
                        help="Number of images to visualize")
    parser.add_argument("--target_class", default=2, type=int,
                        help="Class to compute GradCAM for (2=weed)")
    parser.add_argument("--min_plant",    default=0.08, type=float,
                        help="Min plant pixel ratio for image selection")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    # Resolve output dir dynamically if not explicitly set
    if args.out_dir is None:
        args.out_dir = f"outputs/{args.dataset}/visualizations/results_report"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load model ─────────────────────────────────────────────────────────────
    from src.models.segmentation_model import build_model
    print("Loading model...")
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded: epoch={ckpt.get('epoch', '?')}, "
          f"mIoU={ckpt.get('best_miou', 0):.4f}")

    # ── Load dataset & pick plant-rich images ──────────────────────────────────
    print("Loading dataset...")
    dataset = load_dataset(cfg, args.split, args.dataset)
    print(f"  {len(dataset)} samples in {args.split} split")

    print("Selecting plant-rich images...")
    scored = []
    for i in range(len(dataset)):
        s     = dataset[i]
        ratio = plant_pixel_ratio(s["mask"].numpy())
        if ratio >= args.min_plant:
            scored.append((ratio, i, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    picks = scored[:args.n_images]
    print(f"  Selected {len(picks)} images "
          f"(plant ratio {picks[-1][0]:.1%} – {picks[0][0]:.1%})")

    # ── Figure layout ──────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows     = len(picks)
    n_cols     = 5
    col_titles = [
        "Original image",
        "Model prediction",
        "k=5  (fine details)",
        "k=11  (leaf patterns)",
        "k=17  (plant structure)",
    ]

    fig = plt.figure(figsize=(20, n_rows * 3.8 + 1.5))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.06, wspace=0.04,
                            top=0.93, bottom=0.05,
                            left=0.02, right=0.98)

    for j, title in enumerate(col_titles):
        color = "#1a1a2e" if j < 2 else "#7B3F00"
        fig.text(0.02 + j * 0.196, 0.965, title,
                 ha="center", fontsize=11, fontweight="bold", color=color)

    class_name = ["background", "crop", "weed"][args.target_class]

    # ── Per-image loop ─────────────────────────────────────────────────────────
    branch_keys = [
        "k=5  (fine details)",
        "k=11 (leaf patterns)",
        "k=17 (plant structure)",
    ]

    for row_idx, (ratio, ds_idx, sample) in enumerate(tqdm(picks, desc="GradCAM")):
        image_tensor = sample["image"].unsqueeze(0)   # (1, 3, H, W)
        gt_mask      = sample["mask"].numpy()
        stem         = sample["stem"]
        img_rgb      = denorm(image_tensor)

        # ── Prediction (no grad) ───────────────────────────────────────────────
        with torch.no_grad():
            pred_out  = model(image_tensor.clone())
        pred_mask = pred_out["seg"].argmax(1).squeeze(0).cpu().numpy()

        # ── GradCAM — fresh hooks per image per branch ────────────────────────
        branch_layer_map = {
            "k=5  (fine details)":    model.msfca.branches[0].bn,
            "k=11 (leaf patterns)":   model.msfca.branches[1].bn,
            "k=17 (plant structure)": model.msfca.branches[2].bn,
        }

        cams = {}
        for name, layer in branch_layer_map.items():
            gc = GradCAM(model, layer)                # fresh hooks each time
            try:
                cam = gc.compute(
                    image_tensor.clone(),             # always clone input
                    target_class=args.target_class,
                )
                cams[name] = cam
            finally:
                gc.remove_hooks()                     # clean up immediately

        # ── Assemble panels ────────────────────────────────────────────────────
        panels = (
            [img_rgb, mask_to_color(pred_mask)]
            + [apply_heatmap(img_rgb, cams[k])[0] for k in branch_keys]
        )

        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(panel)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{stem[-14:]}\nplant {ratio:.0%}",
                    fontsize=8, rotation=0, labelpad=55,
                    va="center", color="#444",
                )

    # ── Shared colorbar ────────────────────────────────────────────────────────
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cbar_ax = fig.add_axes([0.65, 0.01, 0.32, 0.015])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb   = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap="jet"),
        cax=cbar_ax, orientation="horizontal",
    )
    cb.set_label("Activation intensity (low → high)", fontsize=9)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["low", "medium", "high"])

    fig.suptitle(
        f"GradCAM activation maps — MSFCA branches  "
        f"(target class: {class_name})  ·  "
        f"Warmer = higher activation",
        fontsize=13, fontweight="bold", y=0.99,
    )

    out_path = out_dir / "gradcam_msfca.png"
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {out_path.absolute()}")

    print("\nLaTeX snippet:")
    print(r"\begin{figure}[H]")
    print(r"\centering")
    print(r"\includegraphics[width=\linewidth]{gradcam_msfca.png}")
    print(r"\caption{GradCAM activation maps at the three MSFCA branches.")
    print(r"Left to right: original image, model prediction,")
    print(r"$k{=}5$ (fine-scale leaf margins), $k{=}11$ (leaf patterns),")
    print(r"$k{=}17$ (canopy-level plant structure).")
    print(r"Warmer colours indicate higher activation intensity.}")
    print(r"\label{fig:gradcam_msfca}")
    print(r"\end{figure}")


if __name__ == "__main__":
    main()