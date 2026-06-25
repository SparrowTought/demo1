import torch
import torch.nn.functional as F


def sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算Sobel梯度。

    输入x: [B, C, H, W]，输出gx和gy，shape相同。
    """
    c = x.shape[1]
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device) / 8.0
    kernel_y = kernel_x.t()
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kernel_x, padding=1, groups=c)
    gy = F.conv2d(x, kernel_y, padding=1, groups=c)
    return gx, gy


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """返回Sobel梯度幅值，shape为[B, C, H, W]。"""
    gx, gy = sobel_gradients(x)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)
