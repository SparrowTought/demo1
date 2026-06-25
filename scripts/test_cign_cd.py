import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import ChangeDetectionDataset
from src.engine.evaluator import evaluate
from src.models.full_model import CIGNCDModel
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="在test集评估CIGN-CD")
    parser.add_argument("--config", default="configs/cign_cd.yaml")
    parser.add_argument("--checkpoint", default=None)
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
    test_set = ChangeDetectionDataset(cfg["data"]["root"], split="test", image_size=cfg["data"]["image_size"], train=False)
    test_loader = DataLoader(test_set, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda")
    pred_dir = Path(cfg["output"]["dir"]) / "test_predictions"
    metrics = evaluate(model, test_loader, device, save_dir=pred_dir)
    print(metrics)
    print(f"预测图已保存到: {pred_dir}")


if __name__ == "__main__":
    main()
