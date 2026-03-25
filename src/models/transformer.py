"""
Global Transformer — Global Context Modeling
=============================================
Applies multi-head self-attention over spatial tokens derived from
the feature map, then reshapes back to (B, C, H, W).
Uses standard PyTorch TransformerEncoderLayer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


class GlobalTransformer(nn.Module):
    """
    Transformer-based global context module.

    Args:
        in_channels : input/output channel count
        num_heads   : number of attention heads
        mlp_ratio   : expansion ratio for the FFN hidden dim
        dropout     : attention + FFN dropout
        num_layers  : number of stacked transformer layers
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Positional encoding is learned; max 32×32 = 1024 tokens
        self.max_seq_len = 1024
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, in_channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=int(in_channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection (preserves channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)  — same shape, global context infused
        """
        B, C, H, W = x.shape
        seq_len = H * W
        assert seq_len <= self.max_seq_len, (
            f"GlobalTransformer: spatial size {H}×{W}={seq_len} exceeds "
            f"max_seq_len={self.max_seq_len}. Reduce image size or increase max_seq_len."
        )

        # Flatten spatial dims → sequence: (B, H*W, C)
        tokens = rearrange(x, "b c h w -> b (h w) c")

        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :seq_len, :]

        # Global self-attention
        tokens = self.transformer(tokens)

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        out = rearrange(tokens, "b (h w) c -> b c h w", h=H, w=W)

        return self.out_proj(out)