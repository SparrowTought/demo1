import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.datasets import AutoEncoderImageDataset, ChangeDetectionDataset
from src.engine.trainer_ae import train_autoencoder
from src.engine.trainer_cd import train_cign_cd
from src.models.autoencoder import TinyAutoEncoder
from src.models.full_model import CIGNCDModel
from src.utils.seed import set_seed


# =========================
# 新手主要改这里：训练超参数
# =========================
HYPERPARAMS = {
    # 数据路径。换成自己的数据集时，只改这里即可。
    "data_root": "./dummy_cd_dataset",
    "image_size": 256,
    "batch_size": 4,
    # Windows电脑如果DataLoader卡住，把num_workers改成0。
    "num_workers": 4,
    "device": "cuda",
    "seed": 42,

    # CIGN-CD主模型训练参数。
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "amp": True,
    "resume": None,

    # 模型宽度参数。显存不够时可以适当调小feat_ch和hidden_ch。
    "feat_ch": 128,
    "control_ch": 64,
    "latent_ch": 4,
    "style_dim": 128,
    "hidden_ch": 128,
    "freeze_autoencoder": True,

    # 完整Latent Diffusion超参数。
    "diffusion_steps": 1000,
    "inference_steps": 50,
    "sample_method": "ddim",  # 可选: "ddim" 或 "ddpm"
    "edit_strength": 0.65,

    # 损失权重。
    "lambda_ref": 1.0,
    "lambda_diff": 0.1,

    # 输出路径。
    "output_dir": "./outputs/cign_cd",

    # AutoEncoder预训练参数。
    "pretrain_ae": True,
    "ae_epochs": 50,
    "ae_lr": 2e-4,
    "ae_output_dir": "./outputs/ae_pretrain",
    "ae_checkpoint": "./outputs/ae_pretrain/checkpoints/best_ae.pth",
}


def make_device(device_name: str) -> torch.device:
    """选择训练设备；没有GPU时自动回到CPU。"""
    return torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")


def build_ae_cfg(hp: dict) -> dict:
    """把超参数整理为AE训练配置。"""
    return {
        "seed": hp["seed"],
        "device": hp["device"],
        "data": {
            "root": hp["data_root"],
            "image_size": hp["image_size"],
            "batch_size": hp["batch_size"],
            "num_workers": hp["num_workers"],
        },
        "train": {
            "epochs": hp["ae_epochs"],
            "lr": hp["ae_lr"],
            "weight_decay": hp["weight_decay"],
            "amp": hp["amp"],
        },
        "output": {"dir": hp["ae_output_dir"]},
        "model": {"latent_ch": hp["latent_ch"]},
    }


def build_cd_cfg(hp: dict) -> dict:
    """把超参数整理为CIGN-CD训练配置。"""
    return {
        "seed": hp["seed"],
        "device": hp["device"],
        "data": {
            "root": hp["data_root"],
            "image_size": hp["image_size"],
            "batch_size": hp["batch_size"],
            "num_workers": hp["num_workers"],
        },
        "model": {
            "feat_ch": hp["feat_ch"],
            "control_ch": hp["control_ch"],
            "latent_ch": hp["latent_ch"],
            "style_dim": hp["style_dim"],
            "hidden_ch": hp["hidden_ch"],
            "diffusion_steps": hp["diffusion_steps"],
            "inference_steps": hp["inference_steps"],
            "sample_method": hp["sample_method"],
            "edit_strength": hp["edit_strength"],
            "freeze_autoencoder": hp["freeze_autoencoder"],
            "ae_checkpoint": hp["ae_checkpoint"],
        },
        "train": {
            "epochs": hp["epochs"],
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"],
            "amp": hp["amp"],
        },
        "loss": {
            "lambda_ref": hp["lambda_ref"],
            "lambda_diff": hp["lambda_diff"],
        },
        "output": {"dir": hp["output_dir"]},
    }


def train_ae_if_needed(hp: dict, device: torch.device) -> None:
    """如果需要，先训练TinyAutoEncoder。"""
    ae_ckpt = Path(hp["ae_checkpoint"])
    if ae_ckpt.exists() and not hp["pretrain_ae"]:
        print(f"发现已有AE权重，跳过AE预训练: {ae_ckpt}")
        return

    print("开始预训练TinyAutoEncoder...")
    cfg = build_ae_cfg(hp)
    train_set = AutoEncoderImageDataset(hp["data_root"], split="train", image_size=hp["image_size"], train=True)
    val_set = AutoEncoderImageDataset(hp["data_root"], split="val", image_size=hp["image_size"], train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=hp["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = TinyAutoEncoder(latent_ch=hp["latent_ch"]).to(device)
    train_autoencoder(model, train_loader, val_loader, cfg, device)


def train_change_detector(hp: dict, device: torch.device) -> None:
    """训练完整CIGN-CD模型。"""
    cfg = build_cd_cfg(hp)
    train_set = ChangeDetectionDataset(hp["data_root"], split="train", image_size=hp["image_size"], train=True)
    val_set = ChangeDetectionDataset(hp["data_root"], split="val", image_size=hp["image_size"], train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=hp["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp["num_workers"],
        pin_memory=device.type == "cuda",
    )

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
        freeze_autoencoder=hp["freeze_autoencoder"],
    ).to(device)

    ae_ckpt = Path(hp["ae_checkpoint"])
    if ae_ckpt.exists():
        model.load_autoencoder_weights(str(ae_ckpt), strict=True)
        if hp["freeze_autoencoder"]:
            model.freeze_autoencoder()
        print(f"已加载AE权重: {ae_ckpt}")
    else:
        print(f"未找到AE权重，将使用随机初始化AE: {ae_ckpt}")

    metrics = train_cign_cd(model, train_loader, val_loader, cfg, device, resume=hp["resume"])
    print("训练完成，最后一轮指标:")
    print(metrics)


def main() -> None:
    hp = HYPERPARAMS
    set_seed(hp["seed"])
    device = make_device(hp["device"])
    print(f"当前设备: {device}")
    train_ae_if_needed(hp, device)
    train_change_detector(hp, device)


if __name__ == "__main__":
    main()
