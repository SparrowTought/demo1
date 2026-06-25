import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_background(size: int) -> Image.Image:
    """生成带纹理的随机遥感风格背景。"""
    arr = np.random.randint(50, 180, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=1.2))
    draw = ImageDraw.Draw(img)
    for _ in range(18):
        x0 = random.randint(0, size - 20)
        y0 = random.randint(0, size - 20)
        x1 = min(size, x0 + random.randint(15, 70))
        y1 = min(size, y0 + random.randint(8, 50))
        color = tuple(random.randint(60, 210) for _ in range(3))
        draw.rectangle([x0, y0, x1, y1], fill=color)
    return img.filter(ImageFilter.GaussianBlur(radius=0.4))


def add_changes(a_img: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    """在B图像中添加矩形和圆形变化区域，同时生成label。"""
    b_img = a_img.copy()
    label = Image.new("L", (size, size), 0)
    draw_b = ImageDraw.Draw(b_img)
    draw_l = ImageDraw.Draw(label)
    for _ in range(random.randint(2, 5)):
        x0 = random.randint(10, size - 60)
        y0 = random.randint(10, size - 60)
        w = random.randint(20, 70)
        h = random.randint(20, 70)
        color = tuple(random.randint(170, 255) for _ in range(3))
        if random.random() < 0.5:
            box = [x0, y0, min(size - 1, x0 + w), min(size - 1, y0 + h)]
            draw_b.rectangle(box, fill=color)
            draw_l.rectangle(box, fill=255)
        else:
            box = [x0, y0, min(size - 1, x0 + w), min(size - 1, y0 + w)]
            draw_b.ellipse(box, fill=color)
            draw_l.ellipse(box, fill=255)
    return b_img, label


def generate_split(root: Path, split: str, count: int, size: int) -> None:
    """生成一个数据划分。"""
    for sub in ("A", "B", "label"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        name = f"{idx:04d}.png"
        a_img = make_background(size)
        b_img, label = add_changes(a_img, size)
        a_img.save(root / split / "A" / name)
        b_img.save(root / split / "B" / name)
        label.save(root / split / "label" / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成CIGN-CD最小可运行假数据集")
    parser.add_argument("--out", default="./dummy_cd_dataset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.out)
    generate_split(root, "train", args.train, args.size)
    generate_split(root, "val", args.val, args.size)
    generate_split(root, "test", args.test, args.size)
    print(f"已生成假数据集: {root.resolve()}")


if __name__ == "__main__":
    main()
