import torch
import torch.nn as nn

from .blocks import ConvGNAct, ResBlock


class SharedBackbone(nn.Module):
    """
    共享特征编码器。

    输入:
        x: [B, 3, H, W]
    输出:
        z: [B, feat_ch, H/4, W/4]
    """

    def __init__(self, feat_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(3, 32, stride=2),
            ResBlock(32),
            ConvGNAct(32, 64, stride=2),
            ResBlock(64),
            ConvGNAct(64, 128, stride=1),
            ResBlock(128),
            ConvGNAct(128, feat_ch, kernel_size=1, padding=0),
            ResBlock(feat_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入x: [B, 3, H, W]，输出z: [B, feat_ch, H/4, W/4]。"""
        return self.net(x)
