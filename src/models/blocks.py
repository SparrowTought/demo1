import math

import torch
import torch.nn as nn


def valid_group_count(channels: int, max_groups: int = 8) -> int:
    """返回能整除通道数的GroupNorm分组数。"""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    """卷积、GroupNorm与SiLU激活的基础模块。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(valid_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入输出均为[B, C, H, W]格式。"""
        return self.net(x)


class ResBlock(nn.Module):
    """轻量残差块，保持空间尺寸不变。"""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(valid_group_count(ch), ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入x: [B, C, H, W]，输出: [B, C, H, W]。"""
        return self.act(x + self.conv2(self.conv1(x)))


class TimeEmbedding(nn.Module):
    """将扩散时间步编码为连续条件向量。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """输入t: [B]，输出: [B, dim]。"""
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.proj(emb)


class FiLMResBlock(nn.Module):
    """带FiLM调制的残差块，用时间与风格条件调制latent特征。"""

    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(valid_group_count(ch), ch)
        self.norm2 = nn.GroupNorm(valid_group_count(ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, ch * 4)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """输入x: [B, C, H, W]，cond: [B, D]，输出: [B, C, H, W]。"""
        gamma1, beta1, gamma2, beta2 = self.cond(cond).chunk(4, dim=1)
        gamma1 = gamma1[:, :, None, None]
        beta1 = beta1[:, :, None, None]
        gamma2 = gamma2[:, :, None, None]
        beta2 = beta2[:, :, None, None]
        h = self.norm1(x) * (1.0 + gamma1) + beta1
        h = self.conv1(self.act(h))
        h = self.norm2(h) * (1.0 + gamma2) + beta2
        h = self.conv2(self.act(h))
        return x + h
