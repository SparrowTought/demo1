from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .image_io import tensor_to_uint8_image


def _gray_panel(x: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    arr = x[0]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = F.interpolate(arr[None, None], size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def _rgb_panel(x: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    arr = tensor_to_uint8_image(x)
    img = Image.fromarray(arr).convert("RGB")
    return img.resize((size[1], size[0]), Image.BILINEAR)


def save_prediction_grid(
    t1: torch.Tensor,
    t2: torch.Tensor,
    mask: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    save_path: str | Path,
    max_items: int = 4,
) -> None:
    """保存T1、T2、GT、T2_ref、Pred、A_exp、A_cert、Structure_Weight可视化网格。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(t1.shape[0], max_items)
    h, w = t1.shape[-2:]
    titles = ["T1", "T2", "GT", "T2_ref", "Pred", "A_exp", "A_cert", "Structure_Weight"]
    cell_h, cell_w = h, w
    title_h = 20
    canvas = Image.new("RGB", (cell_w * len(titles), (cell_h + title_h) * n), "white")
    draw = ImageDraw.Draw(canvas)
    prob = outputs["prob"].detach()
    pred = (prob > 0.5).float()
    for i in range(n):
        panels = [
            _rgb_panel(t1[i], (cell_h, cell_w)),
            _rgb_panel(t2[i], (cell_h, cell_w)),
            _gray_panel(mask[i], (cell_h, cell_w)),
            _rgb_panel(outputs["t2_ref"][i], (cell_h, cell_w)),
            _gray_panel(pred[i], (cell_h, cell_w)),
            _gray_panel(outputs["a_exp"][i], (cell_h, cell_w)),
            _gray_panel(outputs["a_cert"][i], (cell_h, cell_w)),
            _gray_panel(outputs["structure_weight"][i], (cell_h, cell_w)),
        ]
        y0 = i * (cell_h + title_h)
        for j, panel in enumerate(panels):
            x0 = j * cell_w
            draw.text((x0 + 4, y0 + 3), titles[j], fill=(0, 0, 0))
            canvas.paste(panel, (x0, y0 + title_h))
    canvas.save(save_path)
