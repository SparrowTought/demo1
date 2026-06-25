import torch
import torch.nn.functional as F


def warp_with_offset(x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    使用像素单位偏移场对特征进行双线性采样。

    输入:
        x: [B, C, H, W]
        offset: [B, 2, H, W]，第0通道为x方向偏移，第1通道为y方向偏移
    输出:
        warped: [B, C, H, W]
    """
    b, c, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    norm_x = offset[:, 0] * (2.0 / max(w - 1, 1))
    norm_y = offset[:, 1] * (2.0 / max(h - 1, 1))
    flow = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(x, grid + flow, mode="bilinear", padding_mode="border", align_corners=True)
