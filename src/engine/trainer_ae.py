from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.losses.losses import autoencoder_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import JsonlLogger


def train_autoencoder(model, train_loader, val_loader, cfg: dict, device: torch.device) -> dict[str, float]:
    """训练TinyAutoEncoder并保存best与last checkpoint。"""
    out_dir = Path(cfg["output"]["dir"])
    ckpt_dir = out_dir / "checkpoints"
    logger = JsonlLogger(out_dir / "train_log.jsonl")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"].get("amp", True) and device.type == "cuda")
    best_loss = float("inf")
    last_metrics = {"val_loss": best_loss}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"ae train {epoch}", leave=False):
            x = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                outputs = model(x)
                loss = autoencoder_loss(x, outputs["rec"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)
        val_loss = validate_autoencoder(model, val_loader, device)
        last_metrics = {"train_loss": train_loss, "val_loss": val_loss}
        logger.log({"epoch": epoch, **last_metrics})
        save_checkpoint(ckpt_dir / "last_ae.pth", model, optimizer=optimizer, epoch=epoch, best_metric=-best_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(ckpt_dir / "best_ae.pth", model, optimizer=optimizer, epoch=epoch, best_metric=-best_loss)
    return last_metrics


@torch.no_grad()
def validate_autoencoder(model, loader, device: torch.device) -> float:
    """计算自编码器验证损失。"""
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc="ae val", leave=False):
        x = batch["image"].to(device)
        outputs = model(x)
        total += autoencoder_loss(x, outputs["rec"]).item()
    return total / max(len(loader), 1)
