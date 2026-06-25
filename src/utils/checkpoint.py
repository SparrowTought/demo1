from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, epoch: int = 0, best_metric: float = 0.0, extra: dict | None = None) -> None:
    """保存包含模型、优化器、epoch和best metric的checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location: str = "cpu") -> dict:
    """加载checkpoint并恢复可选优化器和学习率策略。"""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
