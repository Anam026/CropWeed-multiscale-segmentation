"""
MSFCA — Multi-Scale Feature Channel Attention Module
=====================================================
Architecture (per the pipeline diagram):
  Input feature map (B, C_in, H, W)
  ↓  3 parallel asymmetric conv branches:
     branch_k:  Conv(1×k) → Conv(k×1) → BN → ReLU
     k ∈ {5, 11, 17}
  ↓  Channel attention squeeze-excitation on concatenated branch outputs
  ↓  Output (B, C_out, H, W)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricConvBranch(nn.Module):
    """
    Asymmetric convolution: 1×k followed by k×1.
    Both preserve spatial dimensions (same padding).
    """

    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.conv1xk = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, k),
            padding=(0, k // 2),
            bias=False,
        )
        self.convkx1 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(k, 1),
            padding=(k // 2, 0),
            bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)          # ← fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1xk(x)
        x = self.convkx1(x)
        x = self.bn(x)
        return self.act(x)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=False),                # ← fixed
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.fc(self.gap(x)).view(b, c, 1, 1)
        return x * scale


class MSFCAModule(nn.Module):
    """
    Multi-Scale Feature Channel Attention Module.

    Args:
        in_channels     : channels of the input feature map (e.g. 2048 from ResNet50)
        out_channels    : desired output channels (e.g. 512)
        kernel_sizes    : list of asymmetric kernel values (default [5, 11, 17])
        reduction_ratio : SE channel squeeze factor (default 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = None,
        reduction_ratio: int = 16,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes or [5, 11, 17]
        n_branches = len(self.kernel_sizes)

        branch_channels = out_channels // n_branches

        # 1×1 projection before branches
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),                # ← fixed
        )

        # Parallel asymmetric branches
        self.branches = nn.ModuleList([
            AsymmetricConvBranch(out_channels, branch_channels, k)
            for k in self.kernel_sizes
        ])

        # Channel attention on concatenated branches
        concat_channels = branch_channels * n_branches
        self.channel_att = ChannelAttention(concat_channels, reduction=reduction_ratio)

        # Final 1×1 fusion
        self.output_proj = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),                # ← fixed
        )

        # Residual shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, out_channels, H, W)
        """
        residual = self.shortcut(x)

        x = self.input_proj(x)

        branch_outs = [branch(x) for branch in self.branches]
        concat = torch.cat(branch_outs, dim=1)

        concat = self.channel_att(concat)

        out = self.output_proj(concat)

        return F.relu(out + residual, inplace=False)   # ← fixed