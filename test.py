import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.datasets import ChangeDetectionDataset
from src.engine.evaluator import evaluate
from src.models.full_model import CIGNCDModel
from src.utils.checkpoint import load_checkpoint


# =========================
# 新手主要改这里：测试超参数
# =========================
HYPERPARAMS = {
    "data_root": "./dummy_cd_dataset",
    "checkpoint": "./outputs/cign_cd/checkpoints/best.pth",
    "save_dir": "./outputs/cign_cd/test_predictions",
    "image_size": 256,
    "batch_size": 4,
    # Windows电脑如果DataLoader卡住，把num_workers改成0。
    "num_workers": 4,
    "device": "cuda",
    "threshold": 0.5,

    # 模型结构参数必须和训练时保持一致。
    "feat_ch": 128,
    "control_ch": 64,
    "latent_ch": 4,
    "style_dim": 128,
    "hidden_ch": 128,

    # 完整Latent Diffusion推理参数。
    "diffusion_steps": 1000,
    "inference_steps": 50,
    "sample_method": "ddim",  # 可选: "ddim" 或 "ddpm"
    "edit_strength": 0.65,
}


def make_device(device_name: str) -> torch.device:
    """选择测试设备；没有GPU时自动回到CPU。"""
    return torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main() -> None:
    hp = HYPERPARAMS
    device = make_device(hp["device"])
    print(f"当前设备: {device}")

    model = CIGNCDModel(
        feat_ch=hp["feat_ch"],
        control_ch=hp["control_ch"],
        latent_ch=hp["latent_ch"],
        style_dim=hp["style_dim"],
        hidden_ch=hp["hidden_ch"],
        diffusion_steps=hp["diffusion_steps"],
        inference_steps=hp["inference_steps"],
        sample_method=hp["sample_method"],
        edit_strength=hp["edit_strength"],
        freeze_autoencoder=True,
    ).to(device)

    load_checkpoint(hp["checkpoint"], model, map_location=str(device))
    test_set = ChangeDetectionDataset(hp["data_root"], split="test", image_size=hp["image_size"], train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp["num_workers"],
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(model, test_loader, device, threshold=hp["threshold"], save_dir=hp["save_dir"])
    print("测试指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"预测图已保存到: {hp['save_dir']}")


if __name__ == "__main__":
    main()
