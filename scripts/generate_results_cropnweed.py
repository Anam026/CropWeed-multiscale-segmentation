"""
generate_results_caw.py — Publication graphs for CropAndWeed dataset
Usage:
    python scripts/generate_results_caw.py \
        --checkpoint outputs/cropandweed/checkpoints/best.pth \
        --config configs/config_caw.yaml
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch, yaml, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm

from src.data.dataset_cropnweed import CropAndWeedDataset
from src.data.transforms import get_val_transforms
from src.models.segmentation_model import build_model
from src.evaluation.metrics import SegmentationMetrics

CLASS_NAMES  = ["Background", "Crop", "Weed"]
CLASS_COLORS = ["#141414", "#00c800", "#c80000"]

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                     "axes.titlesize":13,"axes.titleweight":"bold",
                     "figure.dpi":150,"savefig.bbox":"tight"})

OUT_DIR = Path("outputs/cropandweed/visualizations/results_report")


def mask_to_color(mask):
    cmap = np.array([[20,20,20],[0,200,0],[200,0,0]], dtype=np.uint8)
    out  = np.zeros((*mask.shape,3), dtype=np.uint8)
    for c, col in enumerate(cmap):
        out[mask==c] = col
    return out

def denorm(tensor):
    m = np.array([0.485,0.456,0.406]); s = np.array([0.229,0.224,0.225])
    img = tensor.cpu().permute(1,2,0).numpy()
    return (np.clip(img*s+m,0,1)*255).astype(np.uint8)

def compute_iou(pred, gt):
    ious = []
    for c in range(3):
        v = gt != 255
        inter = ((pred==c)&(gt==c)&v).sum()
        union = ((pred==c)|(gt==c))
        u = (union&v).sum()
        if u > 0: ious.append(inter/u)
    return np.mean(ious) if ious else 0.0


def plot_metrics_bar(metrics, save_path):
    keys   = ["iou","dice","precision","recall"]
    labels = ["IoU","Dice","Precision","Recall"]
    x = np.arange(3); w = 0.2
    fig, ax = plt.subplots(figsize=(10,5))
    for i,(k,l) in enumerate(zip(keys,labels)):
        vals = [metrics.get(f"class_{c}_{k}",0) for c in range(3)]
        bars = ax.bar(x+i*w, vals, w, label=l,
                      color=["#3498db","#e67e22","#9b59b6","#1abc9c"][i],
                      alpha=0.85, edgecolor="white")
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0,1.12); ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics — CropAndWeed Dataset")
    ax.legend(loc="upper right"); ax.grid(axis="y",alpha=0.4,linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    miou = metrics.get("miou",0)
    ax.axhline(miou, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(2.6, miou+0.02, f"mIoU={miou:.3f}", color="red", fontsize=9)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, save_path):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    rs  = cm.sum(1,keepdims=True)
    cmn = np.divide(cm.astype(float), rs, where=rs!=0)
    cmap = LinearSegmentedColormap.from_list("g",["#ffffff","#27ae60"],N=256)
    for ax,data,title,fmt in [
        (ax1,cmn,"Normalized",":.2f"),(ax2,cm.astype(float),"Raw (pixels)",":.0f")]:
        im = ax.imshow(data,cmap=cmap,vmin=0,vmax=1 if fmt==":.2f" else None)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth"); ax.set_title(title)
        for i in range(3):
            for j in range(3):
                v = data[i,j]; t = format(v,fmt[1:])
                ax.text(j,i,t,ha="center",va="center",fontsize=11,fontweight="bold",
                        color="white" if (fmt==":.2f" and v>0.5) else "black")
    plt.suptitle("Confusion Matrix — CropAndWeed Segmentation",fontsize=14,fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  Saved: {save_path}")


def plot_showcase(results, save_path, n=6):
    picks = sorted(results,key=lambda r:compute_iou(r["pred"],r["gt"]),reverse=True)[:n]
    fig   = plt.figure(figsize=(18, n*3.2))
    gs    = gridspec.GridSpec(n,4,figure=fig,hspace=0.08,wspace=0.04)
    for j,t in enumerate(["Original","Ground Truth","Prediction","Overlay"]):
        fig.text(0.13+j*0.215,0.97,t,ha="center",fontsize=12,fontweight="bold")
    for i,r in enumerate(picks):
        iou = compute_iou(r["pred"],r["gt"])
        ovl = cv2.addWeighted(r["image"],0.55,mask_to_color(r["pred"]),0.45,0)
        for j,panel in enumerate([r["image"],mask_to_color(r["gt"]),
                                    mask_to_color(r["pred"]),ovl]):
            ax = fig.add_subplot(gs[i,j]); ax.imshow(panel); ax.axis("off")
            if j==0:
                ax.set_ylabel(f"mIoU={iou:.3f}\n{r['stem'][-12:]}",
                              fontsize=8,rotation=0,labelpad=60,va="center")
    patches = [mpatches.Patch(color=CLASS_COLORS[c],label=CLASS_NAMES[c]) for c in range(3)]
    fig.legend(handles=patches,loc="lower center",ncol=3,fontsize=11,bbox_to_anchor=(0.5,0.01))
    fig.suptitle("CropAndWeed Segmentation — Prediction Showcase",
                 fontsize=15,fontweight="bold",y=0.995)
    plt.savefig(save_path,dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_poster(metrics, results, cm, save_path):
    fig = plt.figure(figsize=(20,14))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(3,4,figure=fig,hspace=0.38,wspace=0.3,
                            top=0.88,bottom=0.08,left=0.05,right=0.97)

    fig.text(0.5,0.94,"Multiscale Semantic Segmentation — CropAndWeed Dataset",
             ha="center",fontsize=18,fontweight="bold",color="white")
    fig.text(0.5,0.905,"MSFCA + Global Transformer + UNet++ Decoder",
             ha="center",fontsize=12,color="#aaaacc")

    cards = [("mIoU",metrics.get("miou",0),"#3498db"),
             ("mDice",metrics.get("mdice",0),"#27ae60"),
             ("Precision",metrics.get("mprecision",0),"#e67e22"),
             ("Recall",metrics.get("mrecall",0),"#9b59b6")]
    for col,(name,val,color) in enumerate(cards):
        ax = fig.add_subplot(gs[0,col]); ax.set_facecolor(color)
        ax.text(0.5,0.55,f"{val:.4f}",transform=ax.transAxes,
                ha="center",va="center",fontsize=26,fontweight="bold",color="white")
        ax.text(0.5,0.18,name,transform=ax.transAxes,
                ha="center",va="center",fontsize=13,color="white",alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    ax_b = fig.add_subplot(gs[1,:2]); ax_b.set_facecolor("#16213e")
    x = np.arange(3); w = 0.18
    for gi,(mk,ml) in enumerate(zip(["iou","dice","precision","recall"],
                                     ["IoU","Dice","Prec","Rec"])):
        vals = [metrics.get(f"class_{c}_{mk}",0) for c in range(3)]
        bars = ax_b.bar(x+gi*w,vals,w,label=ml,
                        color=["#3498db","#27ae60","#e67e22","#9b59b6"][gi],alpha=0.85)
        for bar,v in zip(bars,vals):
            ax_b.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                      f"{v:.2f}",ha="center",fontsize=7,color="white")
    ax_b.set_xticks(x+w*1.5); ax_b.set_xticklabels(CLASS_NAMES,color="white")
    ax_b.set_ylim(0,1.15); ax_b.set_title("Per-Class Metrics",color="white")
    ax_b.legend(fontsize=8,facecolor="#1a1a2e",labelcolor="white",loc="upper right")
    ax_b.tick_params(colors="white"); ax_b.grid(axis="y",alpha=0.2,color="white")
    for sp in ax_b.spines.values(): sp.set_color("#444")

    ax_cm = fig.add_subplot(gs[1,2:])
    rs  = cm.sum(1,keepdims=True)
    cmn = np.divide(cm.astype(float),rs,where=rs!=0)
    cmap2 = LinearSegmentedColormap.from_list("g",["#16213e","#27ae60"],N=256)
    ax_cm.imshow(cmn,cmap=cmap2,vmin=0,vmax=1)
    ax_cm.set_xticks(range(3)); ax_cm.set_yticks(range(3))
    ax_cm.set_xticklabels(CLASS_NAMES,color="white",fontsize=9)
    ax_cm.set_yticklabels(CLASS_NAMES,color="white",fontsize=9)
    ax_cm.set_title("Confusion Matrix",color="white")
    for ii in range(3):
        for jj in range(3):
            ax_cm.text(jj,ii,f"{cmn[ii,jj]:.2f}",ha="center",va="center",
                       fontsize=11,fontweight="bold",
                       color="white" if cmn[ii,jj]>0.4 else "#aaa")

    picks = sorted(results,key=lambda r:compute_iou(r["pred"],r["gt"]),reverse=True)[:4]
    for col,r in enumerate(picks):
        iou = compute_iou(r["pred"],r["gt"])
        ovl = cv2.addWeighted(r["image"],0.5,mask_to_color(r["pred"]),0.5,0)
        combo = np.concatenate([r["image"],mask_to_color(r["pred"]),ovl],axis=1)
        ax = fig.add_subplot(gs[2,col])
        ax.imshow(combo); ax.set_title(f"mIoU={iou:.3f}",color="white",fontsize=9)
        ax.axis("off")

    fig.text(0.02,0.03,"Original | Prediction | Overlay  (Green=Crop  Red=Weed  Dark=Background)",
             color="#aaaacc",fontsize=9)
    patches = [mpatches.Patch(color=CLASS_COLORS[c],label=CLASS_NAMES[c]) for c in range(3)]
    fig.legend(handles=patches,loc="lower right",ncol=3,fontsize=10,
               facecolor="#16213e",labelcolor="white",bbox_to_anchor=(0.98,0.02))
    plt.savefig(save_path,facecolor=fig.get_facecolor(),dpi=150)
    plt.close(); print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config_cropnweed.yaml")
    parser.add_argument("--split",      default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = build_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Best mIoU: {ckpt.get('best_miou',0):.4f}")

    ds_cfg  = cfg["dataset"]
    dataset = CropAndWeedDataset(
        root_dir=ds_cfg["raw_dir"], split=args.split,
        image_size=ds_cfg["image_size"],
        transform=get_val_transforms(ds_cfg["image_size"],cfg.get("augmentation",{})),
        variant=ds_cfg.get("variant","CropsOrWeed"),
    )
    print(f"Dataset: {len(dataset)} samples")

    tracker = SegmentationMetrics(num_classes=3, ignore_index=255)
    results = []
    print("Running inference...")
    for i in tqdm(range(len(dataset))):
        s = dataset[i]
        with torch.no_grad():
            out = model(s["image"].unsqueeze(0))
        pred = out["seg"].argmax(1).squeeze(0).cpu().numpy()
        edge = torch.sigmoid(out["edge"]).squeeze().cpu().numpy()
        tracker.update(torch.from_numpy(pred).unsqueeze(0), s["mask"].unsqueeze(0))
        results.append({"image":denorm(s["image"]),"gt":s["mask"].numpy(),
                        "pred":pred,"edge":edge,"stem":s["stem"]})

    metrics = tracker.compute()
    cm      = tracker.get_confusion_matrix()
    tracker.print_summary(class_names=CLASS_NAMES)

    print("\nGenerating plots...")
    plot_metrics_bar(metrics,  OUT_DIR/"1_metrics_bar_chart.png")
    plot_confusion_matrix(cm,  OUT_DIR/"2_confusion_matrix.png")
    plot_showcase(results,     OUT_DIR/"3_prediction_showcase.png")
    plot_summary_poster(metrics, results, cm, OUT_DIR/"4_summary_poster.png")

    print(f"\nAll saved to: {OUT_DIR.absolute()}")


if __name__ == "__main__":
    main()