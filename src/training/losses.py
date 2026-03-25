"""
Loss Functions: Dice + Focal + Edge
L_total = (0.5*Dice + 0.5*Focal) + 0.3*EdgeLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth       = smooth

    def forward(self, logits, targets):
        targets = targets.long()
        probs   = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape

        # mask out ignore pixels
        valid = (targets != self.ignore_index)
        t = targets.clone()
        t[~valid] = 0          # safe placeholder, will be zeroed by valid mask

        # one-hot: (B, H, W, C) -> (B, C, H, W)
        t_oh = F.one_hot(t, num_classes=C).permute(0, 3, 1, 2).float()
        t_oh = t_oh * valid.unsqueeze(1).float()

        dice_losses = []
        for c in range(C):
            p = probs[:, c].reshape(B, -1)
            g = t_oh[:, c].reshape(B, -1)
            inter = (p * g).sum(1)
            denom = p.sum(1) + g.sum(1)
            dice_losses.append((1.0 - (2*inter + self.smooth) / (denom + self.smooth)).mean())

        return torch.stack(dice_losses).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        targets = targets.long()
        B, C, H, W = logits.shape

        # flatten
        logits_f  = logits.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)
        targets_f = targets.reshape(-1)                          # (N,)

        # only keep valid pixels
        valid = targets_f != self.ignore_index
        logits_f  = logits_f[valid]
        targets_f = targets_f[valid]

        if targets_f.numel() == 0:
            return logits.sum() * 0.0

        # standard cross-entropy on valid pixels
        log_p = F.log_softmax(logits_f, dim=1)                   # (M, C)
        ce    = F.nll_loss(log_p, targets_f, reduction="none")   # (M,)

        # focal weight
        p_t     = log_p.exp()[torch.arange(len(targets_f)), targets_f]
        focal_w = (1.0 - p_t) ** self.gamma

        return (focal_w * ce).mean()


class EdgeLoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, edge_logits, edge_gt):
        pw = torch.tensor([self.pos_weight], device=edge_logits.device)
        return F.binary_cross_entropy_with_logits(edge_logits, edge_gt.float(), pos_weight=pw)


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=3, dice_weight=0.5, focal_weight=0.5,
                 edge_weight=0.3, focal_gamma=2.0, ignore_index=255, **kwargs):
        super().__init__()
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.edge_weight  = edge_weight
        self.dice_loss    = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.focal_loss   = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index)
        self.edge_loss    = EdgeLoss()

    def forward(self, seg_logits, seg_targets, edge_logits, edge_gt):
        seg_targets = seg_targets.long()
        l_dice  = self.dice_loss(seg_logits, seg_targets)
        l_focal = self.focal_loss(seg_logits, seg_targets)
        l_seg   = self.dice_weight * l_dice + self.focal_weight * l_focal
        l_edge  = self.edge_loss(edge_logits, edge_gt)
        l_total = l_seg + self.edge_weight * l_edge
        return {"total": l_total, "seg": l_seg, "dice": l_dice, "focal": l_focal, "edge": l_edge}


def build_loss(cfg):
    loss_cfg = cfg.get("training", {}).get("loss", {})
    ds_cfg   = cfg.get("dataset", {})
    return CombinedLoss(
        num_classes  = ds_cfg.get("num_classes", 3),
        dice_weight  = loss_cfg.get("dice_weight", 0.5),
        focal_weight = loss_cfg.get("focal_weight", 0.5),
        edge_weight  = loss_cfg.get("edge_weight", 0.3),
        focal_gamma  = loss_cfg.get("focal_gamma", 2.0),
        ignore_index = ds_cfg.get("ignore_index", 255),
    )