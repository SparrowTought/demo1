import torch
import torch.nn as nn

from src.utils.warp import warp_with_offset

from .blocks import ConvGNAct, ResBlock


class ReferenceBoundaryAligner(nn.Module):
    """
    只对无变化参考特征Z2_ref进行边界对齐，不对真实Z2进行warp。
    """

    def __init__(self, feat_ch: int = 128):
        super().__init__()
        self.offset_net = nn.Sequential(
            ConvGNAct(feat_ch * 2 + 1, feat_ch),
            ResBlock(feat_ch),
            nn.Conv2d(feat_ch, 2, 3, padding=1),
        )

    def forward(self, z1: torch.Tensor, z2_ref: torch.Tensor, w1: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        输入:
            z1: [B, feat_ch, H/4, W/4]
            z2_ref: [B, feat_ch, H/4, W/4]
            w1: [B, 1, H/4, W/4]
        输出:
            z2_ref_align: [B, feat_ch, H/4, W/4]
            offset: [B, 2, H/4, W/4]
        """
        offset = 3.0 * torch.tanh(self.offset_net(torch.cat([z1, z2_ref, w1], dim=1)))
        z2_ref_align = warp_with_offset(z2_ref, offset)
        return {"z2_ref_align": z2_ref_align, "offset": offset}
