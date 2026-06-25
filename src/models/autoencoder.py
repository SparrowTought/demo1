import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, ResBlock


class TinyAutoEncoder(nn.Module):
    """
    轻量潜空间自编码器。

    该模块先通过pretrain_autoencoder.py预训练。训练CIGN-CD主模型时，默认冻结其参数。
    """

    def __init__(self, latent_ch: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(3, 32, stride=2),
            ResBlock(32),
            ConvGNAct(32, 64, stride=2),
            ResBlock(64),
            ConvGNAct(64, 128, stride=2),
            ResBlock(128),
            nn.Conv2d(128, latent_ch, 1),
        )
        self.dec_in = nn.Sequential(ConvGNAct(latent_ch, 128), ResBlock(128))
        self.dec_mid1 = nn.Sequential(ConvGNAct(128, 64), ResBlock(64))
        self.dec_mid2 = nn.Sequential(ConvGNAct(64, 32), ResBlock(32))
        self.dec_out = nn.Sequential(ConvGNAct(32, 32), nn.Conv2d(32, 3, 3, padding=1), nn.Tanh())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """输入x: [B, 3, H, W]，输出latent: [B, latent_ch, H/8, W/8]。"""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """输入latent: [B, latent_ch, H/8, W/8]，输出rec: [B, 3, H, W]。"""
        h = self.dec_in(latent)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec_mid1(h)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec_mid2(h)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """输入x: [B, 3, H, W]，输出latent与rec。"""
        latent = self.encode(x)
        rec = self.decode(latent)
        return {"latent": latent, "rec": rec}
