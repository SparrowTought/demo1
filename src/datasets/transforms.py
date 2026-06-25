import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image


def normalize_image(img: Image.Image) -> torch.Tensor:
    """将PIL RGB图像转换为[-1, 1]范围的float32张量。"""
    tensor = TF.to_tensor(img.convert("RGB"))
    return tensor * 2.0 - 1.0


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """将mask中大于127的像素视为变化类别。"""
    tensor = TF.to_tensor(mask.convert("L"))
    return (tensor > 0.5).float()


class PairedCDTransform:
    """
    双时相变化检测配对增强。

    训练阶段包含resize、随机裁剪、水平翻转、垂直翻转和90度旋转；
    验证测试阶段只做resize和归一化。
    """

    def __init__(self, image_size: int = 256, train: bool = True):
        self.image_size = int(image_size)
        self.train = train

    def __call__(self, t1: Image.Image, t2: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.image_size
        t1 = t1.convert("RGB").resize((size, size), Image.BILINEAR)
        t2 = t2.convert("RGB").resize((size, size), Image.BILINEAR)
        mask = mask.convert("L").resize((size, size), Image.NEAREST)

        if self.train:
            crop_size = random.randint(max(size // 2, 16), size)
            i, j, h, w = self._crop_params(size, crop_size)
            t1 = TF.crop(t1, i, j, h, w).resize((size, size), Image.BILINEAR)
            t2 = TF.crop(t2, i, j, h, w).resize((size, size), Image.BILINEAR)
            mask = TF.crop(mask, i, j, h, w).resize((size, size), Image.NEAREST)
            if random.random() < 0.5:
                t1, t2, mask = TF.hflip(t1), TF.hflip(t2), TF.hflip(mask)
            if random.random() < 0.5:
                t1, t2, mask = TF.vflip(t1), TF.vflip(t2), TF.vflip(mask)
            k = random.randint(0, 3)
            if k:
                angle = 90 * k
                t1, t2, mask = TF.rotate(t1, angle), TF.rotate(t2, angle), TF.rotate(mask, angle)

        return normalize_image(t1), normalize_image(t2), mask_to_tensor(mask)

    @staticmethod
    def _crop_params(size: int, crop_size: int) -> tuple[int, int, int, int]:
        top = random.randint(0, size - crop_size)
        left = random.randint(0, size - crop_size)
        return top, left, crop_size, crop_size


class ImageOnlyTransform:
    """自编码器预训练用单图像增强与归一化。"""

    def __init__(self, image_size: int = 256, train: bool = True):
        self.image_size = int(image_size)
        self.train = train

    def __call__(self, img: Image.Image) -> torch.Tensor:
        size = self.image_size
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            k = random.randint(0, 3)
            if k:
                img = TF.rotate(img, 90 * k)
        return normalize_image(img)
