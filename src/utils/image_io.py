from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """将[-1,1]或[0,1]张量转换为HWC uint8图像。"""
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 1:
        arr = x[0].clamp(0, 1).numpy()
        return (arr * 255).astype(np.uint8)
    x = ((x.clamp(-1, 1) + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def save_tensor_image(x: torch.Tensor, path: str | Path) -> None:
    """保存张量图像到PNG。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_uint8_image(x)
    Image.fromarray(arr).save(path)


def load_rgb_tensor(path: str | Path, image_size: int | None = None) -> torch.Tensor:
    """读取RGB图像并归一化到[-1,1]，返回[3,H,W]。"""
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0
