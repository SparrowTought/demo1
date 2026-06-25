import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """二值Dice损失，输入logits与mask均为[B, 1, H, W]。"""
    prob = torch.sigmoid(logits)
    inter = torch.sum(prob * mask)
    union = torch.sum(prob) + torch.sum(mask)
    return 1.0 - (2.0 * inter + eps) / (union + eps)


def bce_dice_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits与Dice损失之和。"""
    return F.binary_cross_entropy_with_logits(logits, mask) + dice_loss(logits, mask)


def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """简化SSIM损失，不依赖外部库，输入范围建议为[-1, 1]。"""
    c1 = 0.01**2
    c2 = 0.03**2
    x = (x + 1.0) * 0.5
    y = (y + 1.0) * 0.5
    mu_x = F.avg_pool2d(x, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, 3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(x * x, 3, stride=1, padding=1) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, 3, stride=1, padding=1) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, 3, stride=1, padding=1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    return torch.clamp((1.0 - ssim) * 0.5, 0.0, 1.0).mean()


def autoencoder_loss(x: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
    """自编码器预训练损失: L1 + 0.2 * SSIM loss。"""
    return F.l1_loss(rec, x) + 0.2 * ssim_loss(rec, x)


def reference_consistency_loss(outputs: dict[str, torch.Tensor], t2: torch.Tensor, mask: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """只在未变化区域约束T2_ref与真实T2的一致性。"""
    unchanged = 1.0 - mask
    img_loss = torch.sum(torch.abs(outputs["t2_ref"] - t2) * unchanged) / (torch.sum(unchanged) * t2.shape[1] + 1e-6)
    mask_s = F.interpolate(unchanged, size=outputs["z2"].shape[-2:], mode="nearest")
    feat_loss = torch.sum(torch.abs(outputs["z2_ref"] - outputs["z2"]) * mask_s) / (torch.sum(mask_s) * outputs["z2"].shape[1] + 1e-6)
    return img_loss + beta * feat_loss


def cign_cd_loss(
    outputs: dict[str, torch.Tensor],
    t1: torch.Tensor,
    t2: torch.Tensor,
    mask: torch.Tensor,
    lambda_ref: float = 1.0,
    lambda_diff: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    简化后的总损失:
        L = L_cd + lambda_ref * L_ref + lambda_diff * L_diff

    其中L_cd监督变化检测，L_ref约束未变化区域参考一致性，
    L_diff是完整latent diffusion的噪声预测损失。
    """
    del t1
    cd_loss = bce_dice_loss(outputs["logits"], mask)
    ref_loss = reference_consistency_loss(outputs, t2, mask)
    diff_loss = outputs.get("diffusion_loss", torch.zeros((), device=cd_loss.device, dtype=cd_loss.dtype))
    total = cd_loss + lambda_ref * ref_loss + lambda_diff * diff_loss
    return {
        "loss": total,
        "cd_loss": cd_loss,
        "ref_loss": ref_loss,
        "diff_loss": diff_loss,
    }
