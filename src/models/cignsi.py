import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, ResBlock


class CompactIGNSIControlEncoder(nn.Module):
    """
    紧凑IGNSI结构控制编码器。

    该模块只从T1特征Z1中提取结构先验P1和结构权重W1，不输出basis，
    不做特征值分解，不引入正交损失。
    """

    def __init__(self, feat_ch: int = 128, hidden_dim: int = 24, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(feat_ch, hidden_dim, 1)
        self.residual_mapper = nn.Sequential(ConvGNAct(hidden_dim, feat_ch), ResBlock(feat_ch))
        self.weight_net = nn.Sequential(
            ConvGNAct(2, 16),
            nn.Conv2d(16, 1, 1),
        )
        self.prior_net = nn.Sequential(
            ConvGNAct(feat_ch * 3, feat_ch),
            ResBlock(feat_ch),
            nn.Conv2d(feat_ch, feat_ch, 1),
        )

    def _local_mean(self, y: torch.Tensor) -> torch.Tensor:
        """使用F.unfold计算局部高斯加权均值。"""
        b, d, h, w = y.shape
        k = self.patch_size
        pad = k // 2
        coords = torch.arange(k, device=y.device, dtype=y.dtype) - pad
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        sigma = max(float(k) / 3.0, 1.0)
        weights = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
        weights = (weights / weights.sum()).view(1, 1, k * k, 1)
        patches = F.unfold(y, kernel_size=k, padding=pad).view(b, d, k * k, h * w)
        mean = (patches * weights).sum(dim=2).view(b, d, h, w)
        return mean

    def forward(self, z1: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        输入:
            z1: [B, C, Hs, Ws]
        输出:
            prior: [B, C, Hs, Ws]
            weight: [B, 1, Hs, Ws]
            residual_energy: [B, 1, Hs, Ws]
            gradient_energy: [B, 1, Hs, Ws]
        """
        eps = 1e-6
        y1 = self.proj(z1)
        mu = self._local_mean(y1)
        residual = y1 - mu
        residual_energy = torch.sqrt(residual.square().mean(dim=1, keepdim=True) + eps)

        grad_x = z1[..., :, 1:] - z1[..., :, :-1]
        grad_y = z1[..., 1:, :] - z1[..., :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        gradient_energy = torch.sqrt((grad_x.square() + grad_y.square()).mean(dim=1, keepdim=True) + eps)

        weight = torch.sigmoid(self.weight_net(torch.cat([residual_energy, gradient_energy], dim=1)))
        residual_hat = self.residual_mapper(residual)
        prior_delta = self.prior_net(torch.cat([z1, residual_hat, weight * z1], dim=1))
        prior = z1 + prior_delta
        return {
            "prior": prior,
            "weight": weight,
            "residual_energy": residual_energy,
            "gradient_energy": gradient_energy,
        }


class ControlAdapter(nn.Module):
    """
    将Compact-IGNSI输出的结构先验P1和权重W1转换为潜空间编辑器使用的结构条件C1。
    """

    def __init__(self, feat_ch: int = 128, control_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(feat_ch + 1, feat_ch),
            ResBlock(feat_ch),
            ConvGNAct(feat_ch, control_ch),
        )

    def forward(self, prior: torch.Tensor, weight: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        输入:
            prior: [B, feat_ch, H/4, W/4]
            weight: [B, 1, H/4, W/4]
            target_size: (H/8, W/8)
        输出:
            c1: [B, control_ch, H/8, W/8]
        """
        c1 = self.net(torch.cat([prior, weight], dim=1))
        return F.interpolate(c1, size=target_size, mode="bilinear", align_corners=False)
