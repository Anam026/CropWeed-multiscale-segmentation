"""
CropWeedSegmentationModel
==========================
Full pipeline assembling:
    CNN Encoder → MSFCA → Global Transformer → Fusion → UNet++ Decoder + Edge Head

Forward pass returns:
    {
        "seg"  : (B, num_classes, H, W)   segmentation logits
        "edge" : (B, 1, H, W)             edge logits
    }
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CNNEncoder
from .msfca import MSFCAModule
from .transformer import GlobalTransformer
from .unetpp_decoder import UNetPPDecoder
from .edge_detection import EdgeDetectionHead


class FusionModule(nn.Module):
    """Concatenate MSFCA and Transformer features, project to out_channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, msfca_feat: torch.Tensor, transformer_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([msfca_feat, transformer_feat], dim=1)
        return self.proj(combined)


class CropWeedSegmentationModel(nn.Module):
    """
    End-to-end crop/weed segmentation model.

    Args:
        encoder_name    : timm backbone (resnet50 / resnet101 / vgg16)
        pretrained      : use pretrained weights for the encoder
        num_classes     : number of output segmentation classes (default 3)
        msfca_out_ch    : MSFCA output channels (default 512)
        transformer_heads: number of attention heads
        transformer_layers: stacked transformer encoder layers
        kernel_sizes    : MSFCA asymmetric conv kernel sizes
        encoder_channels: override encoder channel list
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 3,
        msfca_out_ch: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        kernel_sizes: List[int] = None,
        encoder_channels: List[int] = None,
        freeze_bn: bool = False,
    ):
        super().__init__()

        # 1. Encoder
        self.encoder = CNNEncoder(
            name=encoder_name,
            pretrained=pretrained,
            freeze_bn=freeze_bn,
        )
        enc_ch = encoder_channels or self.encoder.encoder_channels
        bottleneck_ch = enc_ch[-1]  # deepest encoder channels (e.g. 2048)

        # 2. MSFCA Module (applied to bottleneck features)
        self.msfca = MSFCAModule(
            in_channels=bottleneck_ch,
            out_channels=msfca_out_ch,
            kernel_sizes=kernel_sizes or [5, 11, 17],
        )

        # 3. Global Transformer
        self.transformer = GlobalTransformer(
            in_channels=msfca_out_ch,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
        )

        # 4. Fusion
        self.fusion = FusionModule(
            in_channels=msfca_out_ch * 2,
            out_channels=msfca_out_ch,
        )

        # Adapt fused features back to bottleneck channels expected by decoder
        self.bottleneck_adapt = nn.Sequential(
            nn.Conv2d(msfca_out_ch, bottleneck_ch, 1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        # 5. UNet++ Decoder
        self.decoder = UNetPPDecoder(
            encoder_channels=enc_ch,
            decoder_channels=[256, 128, 64, 32],
            num_classes=num_classes,
        )

        # 6. Edge Detection Head (receives finest decoder features)
        self.edge_head = EdgeDetectionHead(in_channels=32, mid_channels=32)

        # Track config for summary
        self.num_classes = num_classes
        self.encoder_name = encoder_name

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            {
                "seg"  : (B, num_classes, H, W)
                "edge" : (B, 1, H, W)
            }
        """
        input_size = x.shape[-2:]

        # ── Encoder ──────────────────────────────────────────────────
        features = self.encoder(x)          # [f0, f1, f2, f3, f4]
        f0, f1, f2, f3, f4 = features       # spatial: H/2..H/32

        # ── MSFCA on bottleneck ───────────────────────────────────────
        msfca_out = self.msfca(f4)           # (B, msfca_out_ch, H/32, W/32)

        # ── Global Transformer ────────────────────────────────────────
        transformer_out = self.transformer(msfca_out)  # same shape

        # ── Fusion ────────────────────────────────────────────────────
        fused = self.fusion(msfca_out, transformer_out)
        fused = self.bottleneck_adapt(fused)   # back to bottleneck_ch

        # Replace encoder bottleneck with enriched fused features
        features_enriched = [f0, f1, f2, f3, fused]

        # ── UNet++ Decoder ────────────────────────────────────────────
        decoder_out = self.decoder(features_enriched)
        seg_logits  = decoder_out["seg"]       # (B, num_classes, H, W)
        edge_feat   = decoder_out["edge_feat"] # finest decoder features

        # ── Edge Head ─────────────────────────────────────────────────
        edge_logits = self.edge_head(edge_feat)  # (B, 1, H', W')

        # Ensure outputs match input spatial resolution
        if seg_logits.shape[-2:] != input_size:
            seg_logits = F.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)
        if edge_logits.shape[-2:] != input_size:
            edge_logits = F.interpolate(edge_logits, size=input_size, mode="bilinear", align_corners=False)

        return {
            "seg":  seg_logits,
            "edge": edge_logits,
        }

    def get_params(self):
        """Return (encoder_params, other_params) for differential LR."""
        enc_params   = list(self.encoder.parameters())
        other_params = (
            list(self.msfca.parameters()) +
            list(self.transformer.parameters()) +
            list(self.fusion.parameters()) +
            list(self.bottleneck_adapt.parameters()) +
            list(self.decoder.parameters()) +
            list(self.edge_head.parameters())
        )
        return enc_params, other_params


def build_model(cfg: dict) -> CropWeedSegmentationModel:
    """Construct model from config dict."""
    m = cfg.get("model", cfg)
    enc = m.get("encoder", {})
    msfca = m.get("msfca", {})
    trans = m.get("transformer", {})
    decoder = m.get("unetpp_decoder", {})
    ds = cfg.get("dataset", {})

    return CropWeedSegmentationModel(
        encoder_name     = enc.get("name", "resnet50"),
        pretrained       = enc.get("pretrained", True),
        num_classes      = ds.get("num_classes", 3),
        msfca_out_ch     = msfca.get("out_channels", 512),
        transformer_heads= trans.get("num_heads", 8),
        transformer_layers= trans.get("num_layers", 2),
        kernel_sizes     = msfca.get("kernel_sizes", [5, 11, 17]),
        encoder_channels = decoder.get("encoder_channels", None),
        freeze_bn        = enc.get("freeze_bn", False),
    )
