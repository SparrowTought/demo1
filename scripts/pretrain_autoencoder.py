import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import AutoEncoderImageDataset
from src.engine.trainer_ae import train_autoencoder
from src.models.autoencoder import TinyAutoEncoder
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="预训练TinyAutoEncoder")
    parser.add_argument("--config", default="configs/ae_pretrain.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device_name = cfg.get("device", "cuda")
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    train_set = AutoEncoderImageDataset(cfg["data"]["root"], split="train", image_size=cfg["data"]["image_size"], train=True)
    val_set = AutoEncoderImageDataset(cfg["data"]["root"], split="val", image_size=cfg["data"]["image_size"], train=False)
    train_loader = DataLoader(train_set, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda")
    model = TinyAutoEncoder(latent_ch=cfg["model"]["latent_ch"]).to(device)
    metrics = train_autoencoder(model, train_loader, val_loader, cfg, device)
    print(metrics)


if __name__ == "__main__":
    main()
