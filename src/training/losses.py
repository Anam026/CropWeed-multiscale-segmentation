"""
Loss Functions: Dice + Focal + Edge
L_total = (dice_w*Dice + focal_w*Focal) + edge_w*EdgeLoss
Now supports class_weights for handling class imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, ignore_index=255, smooth=1.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth
        self.weight       = weight  # per-class weights (num_classes,)

    def forward(self, logits, targets):
        targets = targets.long()
        probs   = F.softmax(logits, dim=1)
        B, C    = probs.shape[:2]
        valid   = (targets != self.ignore_index)
        t       = targets.clone(); t[~valid] = 0
        t_oh    = F.one_hot(t, C).permute(0,3,1,2).float()
        t_oh    = t_oh * valid.unsqueeze(1).float()

        losses = []
        for c in range(C):
            p = probs[:,c].reshape(B,-1)
            g = t_oh[:,c].reshape(B,-1)
            inter = (p*g).sum(1); denom = p.sum(1)+g.sum(1)
            dc = (1-(2*inter+self.smooth)/(denom+self.smooth)).mean()
            if self.weight is not None:
                dc = dc * self.weight[c].to(logits.device)
            losses.append(dc)
        return torch.stack(losses).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.weight       = weight

    def forward(self, logits, targets):
        targets = targets.long()
        B,C,H,W = logits.shape
        lf = logits.permute(0,2,3,1).reshape(-1,C)
        tf = targets.reshape(-1)
        valid = tf != self.ignore_index
        lf = lf[valid]; tf = tf[valid]
        if tf.numel() == 0: return logits.sum()*0.0

        # Apply class weights to cross entropy
        w = self.weight.to(logits.device) if self.weight is not None else None
        log_p = F.log_softmax(lf, dim=1)
        ce    = F.nll_loss(log_p, tf, weight=w, reduction="none")
        p_t   = log_p.exp()[torch.arange(len(tf)), tf]
        return ((1-p_t)**self.gamma * ce).mean()


class EdgeLoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, edge_logits, edge_gt):
        pw = torch.tensor([self.pos_weight], device=edge_logits.device)
        return F.binary_cross_entropy_with_logits(
            edge_logits, edge_gt.float(), pos_weight=pw)


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=3, dice_weight=0.6, focal_weight=0.4,
                 edge_weight=0.4, focal_gamma=2.0, ignore_index=255,
                 class_weights=None, **kwargs):
        super().__init__()
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.edge_weight  = edge_weight

        # Build class weight tensor if provided
        cw = None
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32)

        self.dice_loss  = DiceLoss(num_classes=num_classes,
                                   ignore_index=ignore_index, weight=cw)
        self.focal_loss = FocalLoss(gamma=focal_gamma,
                                    ignore_index=ignore_index, weight=cw)
        self.edge_loss  = EdgeLoss()

    def forward(self, seg_logits, seg_targets, edge_logits, edge_gt):
        seg_targets = seg_targets.long()
        l_dice  = self.dice_loss(seg_logits, seg_targets)
        l_focal = self.focal_loss(seg_logits, seg_targets)
        l_seg   = self.dice_weight*l_dice + self.focal_weight*l_focal
        l_edge  = self.edge_loss(edge_logits, edge_gt)
        l_total = l_seg + self.edge_weight*l_edge
        return {"total":l_total, "seg":l_seg, "dice":l_dice,
                "focal":l_focal, "edge":l_edge}


def build_loss(cfg):
    lc = cfg.get("training",{}).get("loss",{})
    dc = cfg.get("dataset",{})
    return CombinedLoss(
        num_classes   = dc.get("num_classes", 3),
        dice_weight   = lc.get("dice_weight", 0.6),
        focal_weight  = lc.get("focal_weight", 0.4),
        edge_weight   = lc.get("edge_weight", 0.4),
        focal_gamma   = lc.get("focal_gamma", 2.0),
        ignore_index  = dc.get("ignore_index", 255),
        class_weights = lc.get("class_weights", None),
    )