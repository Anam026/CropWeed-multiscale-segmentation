"""
CNN Encoder — ResNet50 / ResNet101 / VGG16 via timm.
Returns a list of feature maps at 4 scales for skip connections.
"""

from typing import List

import torch
import torch.nn as nn
import timm


class CNNEncoder(nn.Module):
    """
    Wraps a timm backbone and exposes multi-scale feature maps
    for use in UNet++ skip connections.

    ResNet50 output channels per stage:
        stage1: 256   (1/4  of input)
        stage2: 512   (1/8)
        stage3: 1024  (1/16)
        stage4: 2048  (1/32)
    Also returns the initial stem output:
        stem:    64   (1/2)
    """

    # Map model name → expected channel list [stem, s1, s2, s3, s4]
    CHANNEL_MAP = {
        "resnet50":  [64, 256,  512, 1024, 2048],
        "resnet101": [64, 256,  512, 1024, 2048],
        "vgg16":     [64, 128,  256,  512,  512],
    }

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()
        self.name = name

        # Load backbone
        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        self.out_channels: List[int] = self.CHANNEL_MAP.get(name, [64, 256, 512, 1024, 2048])

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    @property
    def encoder_channels(self) -> List[int]:
        return self.out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input RGB

        Returns:
            list of 5 feature maps [f0, f1, f2, f3, f4]
            with spatial sizes [H/2, H/4, H/8, H/16, H/32]
        """
        features = self.backbone(x)
        return features
