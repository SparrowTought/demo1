from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .transforms import ImageOnlyTransform, PairedCDTransform


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ChangeDetectionDataset(Dataset):
    """
    二值遥感变化检测数据集。

    返回:
        t1: [3, H, W], float32, range [-1, 1]
        t2: [3, H, W], float32, range [-1, 1]
        mask: [1, H, W], float32, values {0, 1}
        name: 文件名
    """

    def __init__(self, root: str, split: str = "train", image_size: int = 256, train: bool | None = None):
        self.root = Path(root)
        self.split = split
        self.a_dir = self.root / split / "A"
        self.b_dir = self.root / split / "B"
        self.label_dir = self.root / split / "label"
        if not self.a_dir.exists():
            raise FileNotFoundError(f"找不到A目录: {self.a_dir}")
        if not self.b_dir.exists():
            raise FileNotFoundError(f"找不到B目录: {self.b_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"找不到label目录: {self.label_dir}")
        self.names = sorted(p.name for p in self.a_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.names:
            raise RuntimeError(f"数据集中没有图像: {self.a_dir}")
        for name in self.names:
            if not (self.b_dir / name).exists():
                raise FileNotFoundError(f"B目录缺少对应图像: {name}")
            if not (self.label_dir / name).exists():
                raise FileNotFoundError(f"label目录缺少对应图像: {name}")
        if train is None:
            train = split == "train"
        self.transform = PairedCDTransform(image_size=image_size, train=train)

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        """读取一个样本并返回t1、t2、mask与文件名。"""
        name = self.names[index]
        t1 = Image.open(self.a_dir / name)
        t2 = Image.open(self.b_dir / name)
        mask = Image.open(self.label_dir / name)
        t1_t, t2_t, mask_t = self.transform(t1, t2, mask)
        return {"t1": t1_t, "t2": t2_t, "mask": mask_t, "name": name}


class AutoEncoderImageDataset(Dataset):
    """读取训练集中A和B两类图像，作为自编码器重建样本。"""

    def __init__(self, root: str, split: str = "train", image_size: int = 256, train: bool = True):
        split_dir = Path(root) / split
        images = []
        for sub in ("A", "B"):
            cur_dir = split_dir / sub
            if not cur_dir.exists():
                raise FileNotFoundError(f"找不到图像目录: {cur_dir}")
            images.extend(sorted(p for p in cur_dir.iterdir() if p.suffix.lower() in IMG_EXTS))
        if not images:
            raise RuntimeError(f"没有找到自编码器预训练图像: {split_dir}")
        self.images = images
        self.transform = ImageOnlyTransform(image_size=image_size, train=train)

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        """返回单张归一化图像。"""
        path = self.images[index]
        img = self.transform(Image.open(path))
        return {"image": img, "name": path.name}
