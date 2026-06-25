import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.cd_dataset import IMG_EXTS
from src.models.full_model import CIGNCDModel
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.image_io import load_rgb_tensor, save_tensor_image


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="对文件夹中的T1/T2图像进行批量推理")
    parser.add_argument("--config", default="configs/cign_cd.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t1_dir", required=True)
    parser.add_argument("--t2_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
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
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()
    t1_dir = Path(args.t1_dir)
    t2_dir = Path(args.t2_dir)
    out_dir = Path(args.out_dir)
    for sub in ("prob", "pred", "t2_ref", "a_exp", "a_cert", "structure_weight"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in t1_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for name in names:
        t2_path = t2_dir / name
        if not t2_path.exists():
            print(f"跳过缺失T2图像: {name}")
            continue
        t1 = load_rgb_tensor(t1_dir / name, image_size=cfg["data"]["image_size"]).unsqueeze(0).to(device)
        t2 = load_rgb_tensor(t2_path, image_size=cfg["data"]["image_size"]).unsqueeze(0).to(device)
        outputs = model(t1, t2)
        prob = outputs["prob"][0]
        pred = (prob > args.threshold).float()
        save_tensor_image(prob, out_dir / "prob" / name)
        save_tensor_image(pred, out_dir / "pred" / name)
        save_tensor_image(outputs["t2_ref"][0], out_dir / "t2_ref" / name)
        save_tensor_image(outputs["a_exp"][0], out_dir / "a_exp" / name)
        save_tensor_image(outputs["a_cert"][0], out_dir / "a_cert" / name)
        save_tensor_image(outputs["structure_weight"][0], out_dir / "structure_weight" / name)
    print(f"推理结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
