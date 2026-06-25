import torch
import torch.nn as nn

from src.utils.fft import fft_lowpass


class T2StyleEncoder(nn.Module):
    """
    从T2的backbone特征Z2中提取低频观测风格条件c2。

    输入:
        z2: [B, feat_ch, H/4, W/4]
    输出:
        style_vec: [B, style_dim]
        z2_low: [B, feat_ch, H/4, W/4]
    """

    def __init__(self, feat_ch: int = 128, style_dim: int = 128, keep_ratio: float = 0.25):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.mlp = nn.Sequential(
            nn.Linear(feat_ch, style_dim),
            nn.SiLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, z2: torch.Tensor) -> dict[str, torch.Tensor]:
        """输入z2: [B, C, Hs, Ws]，输出低频特征与风格向量。"""
        z2_low = fft_lowpass(z2, keep_ratio=self.keep_ratio)
        pooled = z2_low.mean(dim=(2, 3))
        style_vec = self.mlp(pooled)
        return {"style_vec": style_vec, "z2_low": z2_low}
