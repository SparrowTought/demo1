from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.engine.evaluator import evaluate
from src.losses.losses import cign_cd_loss
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logger import JsonlLogger


def train_cign_cd(model, train_loader, val_loader, cfg: dict, device: torch.device, resume: str | None = None) -> dict[str, float]:
    """训练CIGN-CD主模型，按验证F1保存best checkpoint。"""
    out_dir = Path(cfg["output"]["dir"])
    ckpt_dir = out_dir / "checkpoints"
    logger = JsonlLogger(out_dir / "train_log.jsonl")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg["train"]["epochs"], 1))
    scaler = GradScaler(enabled=cfg["train"].get("amp", True) and device.type == "cuda")
    start_epoch = 1
    best_f1 = 0.0
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer=optimizer, scheduler=scheduler, map_location=str(device))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_metric", 0.0))
    last_metrics = {"f1": best_f1}
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_cd = 0.0
        for batch in tqdm(train_loader, desc=f"cd train {epoch}", leave=False):
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                outputs = model(t1, t2)
                losses = cign_cd_loss(
                    outputs,
                    t1,
                    t2,
                    mask,
                    lambda_ref=cfg["loss"]["lambda_ref"],
                    lambda_diff=cfg["loss"].get("lambda_diff", 0.1),
                )
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += losses["loss"].item()
            total_cd += losses["cd_loss"].item()
        scheduler.step()
        train_loss = total_loss / max(len(train_loader), 1)
        train_cd = total_cd / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device)
        last_metrics = {"train_loss": train_loss, "train_cd_loss": train_cd, **metrics}
        logger.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0], **last_metrics})
        save_checkpoint(ckpt_dir / "last.pth", model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_f1)
        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_f1)
    return last_metrics
