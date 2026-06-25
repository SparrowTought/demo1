import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import ChangeDetectionDataset
from src.models.full_model import CIGNCDModel
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.visualization import save_prediction_grid


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="保存一批CIGN-CD可视化结果")
    parser.add_argument("--config", default="configs/cign_cd.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="./outputs/cign_cd/visualization/batch.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(Path(cfg["output"]["dir"]) / "checkpoints" / "best.pth")
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
    load_checkpoint(ckpt_path, model, map_location=str(device))
    model.eval()
    dataset = ChangeDetectionDataset(cfg["data"]["root"], split=args.split, image_size=cfg["data"]["image_size"], train=False)
    loader = DataLoader(dataset, batch_size=min(cfg["data"]["batch_size"], 4), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    t1 = batch["t1"].to(device)
    t2 = batch["t2"].to(device)
    mask = batch["mask"].to(device)
    outputs = model(t1, t2)
    save_prediction_grid(t1.cpu(), t2.cpu(), mask.cpu(), {k: v.cpu() for k, v in outputs.items() if torch.is_tensor(v)}, args.out)
    print(f"可视化图已保存到: {args.out}")


if __name__ == "__main__":
    main()
