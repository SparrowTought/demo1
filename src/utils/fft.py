import torch


def fft_lowpass(x: torch.Tensor, keep_ratio: float = 0.25) -> torch.Tensor:
    """
    对特征图做中心低通滤波。

    输入:
        x: [B, C, H, W]
    输出:
        low: [B, C, H, W]
    """
    b, c, h, w = x.shape
    ratio = float(max(min(keep_ratio, 1.0), 0.01))
    keep_h = max(1, int(h * ratio))
    keep_w = max(1, int(w * ratio))
    freq = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=(-2, -1)), dim=(-2, -1))
    mask = torch.zeros((h, w), dtype=torch.float32, device=x.device)
    h0 = (h - keep_h) // 2
    w0 = (w - keep_w) // 2
    mask[h0 : h0 + keep_h, w0 : w0 + keep_w] = 1.0
    freq = freq * mask.view(1, 1, h, w)
    low = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), dim=(-2, -1)).real
    return low.to(dtype=x.dtype)
