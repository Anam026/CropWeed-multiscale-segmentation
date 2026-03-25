"""
Segmentation Evaluation Metrics
================================
Computes per-class and mean IoU, Dice, Precision, Recall, F1
using a confusion-matrix-based approach (numerically stable).
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class SegmentationMetrics:
    """
    Accumulates predictions and targets in a confusion matrix,
    then computes standard segmentation metrics.

    Args:
        num_classes  : number of semantic classes
        ignore_index : label to exclude (typically 255)
    """

    def __init__(self, num_classes: int = 3, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.conf_matrix  = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.conf_matrix[:] = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   : (B, H, W) long   — predicted class indices
            targets : (B, H, W) long   — ground-truth class indices
        """
        preds   = preds.cpu().numpy().astype(np.int64).ravel()
        targets = targets.cpu().numpy().astype(np.int64).ravel()

        valid = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Clip to valid range
        preds   = preds.clip(0, self.num_classes - 1)
        targets = targets.clip(0, self.num_classes - 1)

        np.add.at(self.conf_matrix, (targets, preds), 1)

    def compute(self) -> Dict[str, float]:
        cm = self.conf_matrix.astype(np.float64)

        # Per-class TP, FP, FN
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp   # column sum − diag
        fn = cm.sum(axis=1) - tp   # row sum − diag

        eps = 1e-7

        # IoU per class
        iou = tp / (tp + fp + fn + eps)
        # Dice per class
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        # Precision per class
        precision = tp / (tp + fp + eps)
        # Recall per class
        recall = tp / (tp + fn + eps)
        # F1 per class (= Dice)
        f1 = 2 * precision * recall / (precision + recall + eps)

        # Mean over classes (only include classes that appear in GT)
        present = cm.sum(axis=1) > 0   # classes with at least one GT pixel

        metrics = {
            "miou"      : float(iou[present].mean()),
            "mdice"     : float(dice[present].mean()),
            "mprecision": float(precision[present].mean()),
            "mrecall"   : float(recall[present].mean()),
            "mf1"       : float(f1[present].mean()),
        }

        # Per-class metrics
        for c in range(self.num_classes):
            tag = f"class_{c}"
            metrics[f"{tag}_iou"]       = float(iou[c])
            metrics[f"{tag}_dice"]      = float(dice[c])
            metrics[f"{tag}_precision"] = float(precision[c])
            metrics[f"{tag}_recall"]    = float(recall[c])

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        return self.conf_matrix.copy()

    def print_summary(self, class_names: List[str] = None):
        metrics = self.compute()
        names = class_names or [f"class_{i}" for i in range(self.num_classes)]

        print("\n" + "=" * 60)
        print(f"{'Class':<20} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Rec':>8}")
        print("-" * 60)
        for i, name in enumerate(names):
            print(
                f"{name:<20} "
                f"{metrics[f'class_{i}_iou']:>8.4f} "
                f"{metrics[f'class_{i}_dice']:>8.4f} "
                f"{metrics[f'class_{i}_precision']:>8.4f} "
                f"{metrics[f'class_{i}_recall']:>8.4f}"
            )
        print("-" * 60)
        print(f"{'Mean':<20} {metrics['miou']:>8.4f} {metrics['mdice']:>8.4f} "
              f"{metrics['mprecision']:>8.4f} {metrics['mrecall']:>8.4f}")
        print("=" * 60)
