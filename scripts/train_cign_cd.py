import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import ChangeDetectionDataset
from src.engine.trainer_cd import train_cign_cd
from src.models.full_model import CIGNCDModel
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="训练CIGN-CD变化检测模型")
    parser.add_argument("--config", default="configs/cign_cd.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device_name = cfg.get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    train_set = ChangeDetectionDataset(cfg["data"]["root"], split="train", image_size=cfg["data"]["image_size"], train=True)
    val_set = ChangeDetectionDataset(cfg["data"]["root"], split="val", image_size=cfg["data"]["image_size"], train=False)
    train_loader = DataLoader(train_set, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda")
    model_cfg = cfg["model"]
    model = CIGNCDModel(
        feat_ch=model_cfg["feat_ch"],
        control_ch=model_cfg["control_ch"],
        latent_ch=model_cfg["latent_ch"],
        style_dim=model_cfg["style_dim"],
        hidden_ch=model_cfg["hidden_ch"],
        diffusion_steps=model_cfg.get("diffusion_steps", 1000),
        inference_steps=model_cfg.get("inference_steps", 50),
        sample_method=model_cfg.get("sample_method", "ddim"),
        edit_strength=model_cfg.get("edit_strength", 0.65),
        freeze_autoencoder=model_cfg["freeze_autoencoder"],
    ).to(device)
    ae_ckpt = Path(model_cfg["ae_checkpoint"])
    if ae_ckpt.exists():
        model.load_autoencoder_weights(str(ae_ckpt), strict=True)
        if model_cfg["freeze_autoencoder"]:
            model.freeze_autoencoder()
    else:
        print(f"未找到AE权重，将使用随机初始化AE: {ae_ckpt}")
    metrics = train_cign_cd(model, train_loader, val_loader, cfg, device, resume=args.resume)
    print(metrics)


if __name__ == "__main__":
    main()
