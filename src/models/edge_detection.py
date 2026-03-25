"""
Edge Detection Head
===================
A lightweight head that predicts binary edge maps from
the finest decoder feature map. Edges are computed using
a Sobel-like learned filter or via a small CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDetectionHead(nn.Module):
    """
    Predicts a binary edge map from the finest decoder features.

    The edge map is used for:
      1. Auxiliary edge loss (L_edge) during training
      2. Feature refinement via edge-guided attention (optional)

    Args:
        in_channels  : channels from the decoder's finest feature map
        mid_channels : intermediate channel count
    """

    def __init__(self, in_channels: int = 32, mid_channels: int = 32):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
        )
        # Output: 1-channel binary edge probability
        self.out_conv = nn.Conv2d(mid_channels // 2, 1, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, in_channels, H, W)

        Returns:
            edge_logits: (B, 1, H, W) — raw logits, apply sigmoid for probability
        """
        x = self.conv1(feat)
        x = self.conv2(x)
        return self.out_conv(x)


def build_edge_gt(mask: torch.Tensor, dilation: int = 1) -> torch.Tensor:
    """
    Derive binary edge ground truth from a semantic mask using morphological dilation.

    Args:
        mask : (B, H, W) long tensor, class labels
        dilation: edge thickness in pixels

    Returns:
        edge_gt: (B, 1, H, W) float32  — 1 at boundaries, 0 elsewhere
    """
    # Work in float for Laplacian
    b, h, w = mask.shape
    mask_f = mask.float().unsqueeze(1)  # (B, 1, H, W)

    # Simple Laplacian edge detection on the mask
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32,
        device=mask.device,
    ).view(1, 1, 3, 3)

    edges = F.conv2d(mask_f, laplacian_kernel, padding=1).abs()
    edges = (edges > 0.5).float()

    # Optional dilation to thicken edges
    if dilation > 1:
        kernel = torch.ones(1, 1, dilation * 2 + 1, dilation * 2 + 1,
                            device=mask.device)
        edges = F.conv2d(edges, kernel, padding=dilation).clamp(0, 1)

    return edges  # (B, 1, H, W)
