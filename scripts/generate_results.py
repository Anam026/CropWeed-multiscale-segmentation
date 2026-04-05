"""
generate_results.py
====================
Generates publication-quality result graphs and summary figures.

Usage:
    python scripts/generate_results.py \
        --checkpoint outputs/weedsgalore/checkpoints/best.pth \
        --config configs/config.yaml

Produces in outputs/visualizations/results_report/:
    1. metrics_bar_chart.png       - Per-class IoU / Dice / Precision / Recall bars
    2. confusion_matrix.png        - Normalized confusion matrix heatmap
    3. class_distribution.png      - Pixel-level class distribution pie chart
    4. prediction_showcase.png     - Best 6 predictions side by side
    5. edge_detection_showcase.png - Edge map comparisons
    6. summary_poster.png          - Single-page publication summary figure
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import yaml
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm

from src.data.dataset import WeedsGaloreDataset
from src.data.transforms import get_val_transforms
from src.models.segmentation_model import build_model
from src.evaluation.metrics import SegmentationMetrics

# ── Visual config ──────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Background", "Crop", "Weed"]
CLASS_COLORS = ["#4a4a4a", "#2ecc71", "#e74c3c"]   # dark gray, green, red
PALETTE      = ["#2c3e50", "#27ae60", "#c0392b"]

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi":   150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

OUT_DIR = Path("outputs/weedsgalore/visualizations/results_report")


# ── Helpers ────────────────────────────────────────────────────────────────────

def mask_to_color(mask):
    color_map = np.array([[74,74,74],[46,204,113],[231,76,60]], dtype=np.uint8)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in enumerate(color_map):
        out[mask == c] = col
    return out

def denorm(tensor):
    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    img  = tensor.cpu().permute(1,2,0).numpy()
    return (np.clip(img*std+mean, 0, 1)*255).astype(np.uint8)

def run_inference(model, dataset, device, max_samples=None):
    """Run model on dataset, return list of dicts with image/gt/pred/edge."""
    results = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in tqdm(range(n), desc="Running inference"):
        s = dataset[i]
        with torch.no_grad():
            out = model(s["image"].unsqueeze(0).to(device))
        pred = out["seg"].argmax(1).squeeze(0).cpu().numpy()
        edge = torch.sigmoid(out["edge"]).squeeze().cpu().numpy()
        results.append({
            "image": denorm(s["image"]),
            "gt":    s["mask"].numpy(),
            "pred":  pred,
            "edge":  edge,
            "stem":  s["stem"],
        })
    return results


# ── Plot 1: Metrics Bar Chart ──────────────────────────────────────────────────

def plot_metrics_bar(metrics_dict, save_path):
    metric_keys = ["iou", "dice", "precision", "recall"]
    metric_labels = ["IoU", "Dice", "Precision", "Recall"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [metrics_dict.get(f"class_{c}_{key}", 0) for c in range(3)]
        bars = ax.bar(x + i*width, vals, width, label=label,
                      color=["#3498db","#e67e22","#9b59b6","#1abc9c"][i],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([f"{n}\n({c})" for n, c in zip(CLASS_NAMES,
                        [CLASS_COLORS[i] for i in range(3)])],
                        fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Segmentation Metrics — Multiscale Crop-Weed Model")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    # Mean line
    miou = metrics_dict.get("miou", 0)
    ax.axhline(miou, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(CLASS_NAMES)-0.1, miou+0.02, f"mIoU={miou:.3f}",
            color="red", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 2: Confusion Matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(conf_matrix, save_path):
    # Normalize
    cm = conf_matrix.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums!=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title, fmt in [
        (ax1, cm_norm, "Normalized Confusion Matrix", ".2f"),
        (ax2, cm,      "Raw Confusion Matrix (pixels)", ".0f"),
    ]:
        cmap = LinearSegmentedColormap.from_list("cw", ["#ffffff","#27ae60"], N=256)
        im = ax.imshow(data, interpolation="nearest", cmap=cmap, vmin=0,
                       vmax=1 if fmt==".2f" else None)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                val = data[i, j]
                txt = format(val, fmt)
                color = "white" if (fmt==".2f" and val > 0.5) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

    plt.suptitle("Confusion Matrix — Crop vs Weed Segmentation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 3: Class Distribution ─────────────────────────────────────────────────

def plot_class_distribution(results, save_path):
    gt_counts   = np.zeros(3)
    pred_counts = np.zeros(3)
    for r in results:
        for c in range(3):
            gt_counts[c]   += (r["gt"]   == c).sum()
            pred_counts[c] += (r["pred"] == c).sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    explode = (0.03, 0.03, 0.06)

    for ax, counts, title in [
        (axes[0], gt_counts,   "Ground Truth Distribution"),
        (axes[1], pred_counts, "Predicted Distribution"),
    ]:
        total = counts.sum()
        pcts  = counts / total * 100
        wedges, texts, autotexts = ax.pie(
            counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
            autopct="%1.1f%%", explode=explode, startangle=90,
            textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i, (name, pct) in enumerate(zip(CLASS_NAMES, pcts)):
            ax.annotate(f"{name}: {counts[i]/1e6:.1f}M px ({pct:.1f}%)",
                        xy=(0, -1.35 - i*0.12), ha="center", fontsize=9,
                        color=CLASS_COLORS[i])

    plt.suptitle("Pixel-Level Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 4: Prediction Showcase ────────────────────────────────────────────────

def compute_iou_sample(pred, gt):
    ious = []
    for c in range(3):
        inter = ((pred==c) & (gt==c)).sum()
        union = ((pred==c) | (gt==c)).sum()
        ious.append(inter/union if union>0 else 0)
    return np.mean(ious)

def plot_prediction_showcase(results, save_path, n_show=6):
    # Pick samples with best IoU spread
    scored = sorted(results, key=lambda r: compute_iou_sample(r["pred"], r["gt"]),
                    reverse=True)
    picks  = scored[:n_show]

    fig = plt.figure(figsize=(18, n_show * 3.2))
    gs  = gridspec.GridSpec(n_show, 4, figure=fig,
                            hspace=0.08, wspace=0.04)

    col_titles = ["Original Image", "Ground Truth Mask",
                  "Model Prediction", "Overlay (Pred on Image)"]
    for j, t in enumerate(col_titles):
        fig.text(0.13 + j*0.215, 0.97, t, ha="center", fontsize=12,
                 fontweight="bold", color="#2c3e50")

    for i, r in enumerate(picks):
        iou = compute_iou_sample(r["pred"], r["gt"])
        overlay = cv2.addWeighted(r["image"], 0.55,
                                   mask_to_color(r["pred"]), 0.45, 0)
        panels = [r["image"], mask_to_color(r["gt"]),
                  mask_to_color(r["pred"]), overlay]

        for j, panel in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(panel)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(f"mIoU={iou:.3f}\n{r['stem'][-12:]}",
                              fontsize=8, rotation=0, labelpad=60,
                              va="center", color="#555")

    # Legend
    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(3)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Multiscale Crop-Weed Segmentation — Prediction Showcase",
                 fontsize=15, fontweight="bold", y=0.995, color="#1a1a2e")

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 5: Edge Detection Showcase ───────────────────────────────────────────

def plot_edge_showcase(results, save_path, n_show=4):
    picks = results[:n_show]
    fig, axes = plt.subplots(n_show, 3, figsize=(12, n_show * 3))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i, r in enumerate(picks):
        # Derive GT edges via Laplacian
        gt_float = r["gt"].astype(np.float32)
        lap = cv2.Laplacian(gt_float, cv2.CV_32F)
        gt_edge = (np.abs(lap) > 0.5).astype(np.uint8) * 255

        axes[i,0].imshow(r["image"])
        axes[i,0].set_title("Original" if i==0 else "")
        axes[i,1].imshow(gt_edge, cmap="gray")
        axes[i,1].set_title("GT Edge Map" if i==0 else "")
        axes[i,2].imshow(r["edge"], cmap="hot")
        axes[i,2].set_title("Predicted Edge Map" if i==0 else "")
        for ax in axes[i]: ax.axis("off")

    fig.suptitle("Edge Detection Results — Crop/Weed Boundaries",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot 6: Summary Poster ─────────────────────────────────────────────────────

def plot_summary_poster(metrics_dict, results, conf_matrix, save_path):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.3,
                           top=0.88, bottom=0.08, left=0.05, right=0.97)

    # ── Title ──────────────────────────────────────────────────────
    fig.text(0.5, 0.94,
             "Multiscale Semantic Segmentation of Crops and Weeds",
             ha="center", fontsize=18, fontweight="bold", color="white")
    fig.text(0.5, 0.905,
             "MSFCA + Global Transformer + UNet++ Decoder  |  WeedsGalore Dataset",
             ha="center", fontsize=12, color="#aaaacc")

    # ── Metric cards (row 0, cols 0-3) ─────────────────────────────
    card_metrics = [
        ("mIoU",      metrics_dict.get("miou",0),       "#3498db"),
        ("mDice",     metrics_dict.get("mdice",0),      "#27ae60"),
        ("Precision", metrics_dict.get("mprecision",0), "#e67e22"),
        ("Recall",    metrics_dict.get("mrecall",0),    "#9b59b6"),
    ]
    for col, (name, val, color) in enumerate(card_metrics):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(color)
        ax.text(0.5, 0.55, f"{val:.4f}", transform=ax.transAxes,
                ha="center", va="center", fontsize=26, fontweight="bold",
                color="white")
        ax.text(0.5, 0.18, name, transform=ax.transAxes,
                ha="center", va="center", fontsize=13, color="white", alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # ── Per-class IoU bar (row 1, cols 0-1) ────────────────────────
    ax_bar = fig.add_subplot(gs[1, :2])
    ax_bar.set_facecolor("#16213e")
    metric_groups = ["iou", "dice", "precision", "recall"]
    group_labels  = ["IoU", "Dice", "Prec", "Rec"]
    x = np.arange(3); w = 0.18
    colors_bar = ["#3498db","#27ae60","#e67e22","#9b59b6"]
    for gi, (mk, ml) in enumerate(zip(metric_groups, group_labels)):
        vals = [metrics_dict.get(f"class_{c}_{mk}", 0) for c in range(3)]
        bars = ax_bar.bar(x + gi*w, vals, w, label=ml,
                          color=colors_bar[gi], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.01, f"{v:.2f}",
                        ha="center", fontsize=7, color="white")
    ax_bar.set_xticks(x + w*1.5)
    ax_bar.set_xticklabels(CLASS_NAMES, color="white", fontsize=11)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title("Per-Class Metrics", color="white", fontsize=12)
    ax_bar.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
                  loc="upper right")
    ax_bar.tick_params(colors="white")
    ax_bar.set_facecolor("#16213e")
    for sp in ax_bar.spines.values(): sp.set_color("#444")
    ax_bar.grid(axis="y", alpha=0.2, color="white")

    # ── Confusion matrix (row 1, cols 2-3) ─────────────────────────
    ax_cm = fig.add_subplot(gs[1, 2:])
    ax_cm.set_facecolor("#16213e")
    cm = conf_matrix.astype(float)
    rs = cm.sum(1, keepdims=True)
    cm_n = np.divide(cm, rs, where=rs!=0)
    cmap = LinearSegmentedColormap.from_list("g", ["#16213e","#27ae60"], N=256)
    im = ax_cm.imshow(cm_n, cmap=cmap, vmin=0, vmax=1)
    ax_cm.set_xticks(range(3)); ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(CLASS_NAMES, color="white", fontsize=9)
    ax_cm.set_yticklabels(CLASS_NAMES, color="white", fontsize=9)
    ax_cm.set_xlabel("Predicted", color="white"); ax_cm.set_ylabel("GT", color="white")
    ax_cm.set_title("Confusion Matrix", color="white", fontsize=12)
    for ii in range(3):
        for jj in range(3):
            ax_cm.text(jj, ii, f"{cm_n[ii,jj]:.2f}", ha="center", va="center",
                       fontsize=11, color="white" if cm_n[ii,jj]>0.4 else "#aaa",
                       fontweight="bold")

    # ── Sample predictions (row 2, all cols) ───────────────────────
    picks = sorted(results, key=lambda r: compute_iou_sample(r["pred"],r["gt"]),
                   reverse=True)[:4]
    for col, r in enumerate(picks):
        iou  = compute_iou_sample(r["pred"], r["gt"])
        ovly = cv2.addWeighted(r["image"], 0.5, mask_to_color(r["pred"]), 0.5, 0)
        combined = np.concatenate([r["image"], mask_to_color(r["pred"]), ovly], axis=1)
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(combined)
        ax.set_title(f"mIoU={iou:.3f}", color="white", fontsize=9)
        ax.axis("off")

    fig.text(0.02, 0.03,
             "Original | Prediction | Overlay  (Green=Crop  Red=Weed  Dark=Background)",
             color="#aaaacc", fontsize=9)

    patches = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(3)]
    fig.legend(handles=patches, loc="lower right", ncol=3,
               fontsize=10, facecolor="#16213e", labelcolor="white",
               bbox_to_anchor=(0.98, 0.02))

    plt.savefig(save_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--split",      default="test",
                        choices=["train","val","test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("Loading model...")
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    best_miou = ckpt.get("best_miou", 0)
    print(f"Model loaded — best mIoU: {best_miou:.4f}")

    ds_cfg = cfg["dataset"]
    dataset = WeedsGaloreDataset(
        root_dir=ds_cfg["raw_dir"], split=args.split,
        image_size=ds_cfg["image_size"],
        transform=get_val_transforms(ds_cfg["image_size"], cfg.get("augmentation",{})),
        bands=ds_cfg.get("bands",["R","G","B"]),
    )
    print(f"Dataset: {len(dataset)} samples ({args.split} split)")

    # Run inference + compute metrics
    metrics_tracker = SegmentationMetrics(num_classes=3, ignore_index=255)
    results = []
    print("Running inference...")
    for i in tqdm(range(len(dataset))):
        s = dataset[i]
        with torch.no_grad():
            out = model(s["image"].unsqueeze(0))
        pred = out["seg"].argmax(1).squeeze(0).cpu().numpy()
        edge = torch.sigmoid(out["edge"]).squeeze().cpu().numpy()
        metrics_tracker.update(
            torch.from_numpy(pred).unsqueeze(0),
            s["mask"].unsqueeze(0),
        )
        results.append({
            "image": denorm(s["image"]),
            "gt":    s["mask"].numpy(),
            "pred":  pred,
            "edge":  edge,
            "stem":  s["stem"],
        })

    metrics = metrics_tracker.compute()
    conf_matrix = metrics_tracker.get_confusion_matrix()
    metrics_tracker.print_summary(class_names=CLASS_NAMES)

    print("\nGenerating plots...")
    plot_metrics_bar(metrics,      OUT_DIR / "1_metrics_bar_chart.png")
    plot_confusion_matrix(conf_matrix, OUT_DIR / "2_confusion_matrix.png")
    plot_class_distribution(results,   OUT_DIR / "3_class_distribution.png")
    plot_prediction_showcase(results,  OUT_DIR / "4_prediction_showcase.png")
    plot_edge_showcase(results,        OUT_DIR / "5_edge_detection_showcase.png")
    plot_summary_poster(metrics, results, conf_matrix, OUT_DIR / "6_summary_poster.png")

    print(f"\nAll results saved to: {OUT_DIR.absolute()}")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()