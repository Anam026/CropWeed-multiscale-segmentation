"""
UNet++ Decoder — Nested Skip Connections
Channel arithmetic verified for ResNet50:
  f0:64  f1:256  f2:512  f3:1024  f4:2048
  d[0]=256  d[1]=128  d[2]=64  d[3]=32
"""
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),  nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


def _up(src, ref):
    return F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)


class UNetPPDecoder(nn.Module):
    def __init__(self, encoder_channels=None, decoder_channels=None, num_classes=3):
        super().__init__()
        e = encoder_channels or [64, 256, 512, 1024, 2048]
        d = decoder_channels or [256, 128, 64, 32]

        # depth-3: up(f4)+f3 -> d0
        self.x31 = VGGBlock(e[4] + e[3],              d[0], d[0])
        # depth-2
        self.x21 = VGGBlock(e[3] + e[2],              d[1], d[1])
        self.x22 = VGGBlock(d[0] + d[1] + e[2],       d[1], d[1])
        # depth-1
        self.x11 = VGGBlock(e[2] + e[1],              d[2], d[2])
        self.x12 = VGGBlock(d[1] + d[2] + e[1],       d[2], d[2])
        self.x13 = VGGBlock(d[1] + d[2]*2 + e[1],     d[2], d[2])
        # depth-0
        self.x01 = VGGBlock(e[1] + e[0],              d[3], d[3])
        self.x02 = VGGBlock(d[2] + d[3] + e[0],       d[3], d[3])
        self.x03 = VGGBlock(d[2] + d[3]*2 + e[0],     d[3], d[3])
        self.x04 = VGGBlock(d[2] + d[3]*3 + e[0],     d[3], d[3])

        self.seg_head = nn.Conv2d(d[3], num_classes, 1)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, features):
        f0, f1, f2, f3, f4 = features

        x31 = self.x31(torch.cat([_up(f4, f3), f3], 1))

        x21 = self.x21(torch.cat([_up(f3,  f2), f2], 1))
        x22 = self.x22(torch.cat([_up(x31, f2), x21, f2], 1))

        x11 = self.x11(torch.cat([_up(f2,  f1), f1], 1))
        x12 = self.x12(torch.cat([_up(x21, f1), x11, f1], 1))
        x13 = self.x13(torch.cat([_up(x22, f1), x11, x12, f1], 1))

        x01 = self.x01(torch.cat([_up(f1,  f0), f0], 1))
        x02 = self.x02(torch.cat([_up(x11, f0), x01, f0], 1))
        x03 = self.x03(torch.cat([_up(x12, f0), x01, x02, f0], 1))
        x04 = self.x04(torch.cat([_up(x13, f0), x01, x02, x03, f0], 1))

        avg = (x01 + x02 + x03 + x04) / 4.0
        return {"seg": self.final_up(self.seg_head(avg)), "edge_feat": avg}