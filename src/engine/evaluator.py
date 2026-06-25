from pathlib import Path

import torch
from tqdm import tqdm

from src.metrics.binary_metrics import confusion_matrix_binary, merge_confusion
from src.utils.image_io import save_tensor_image


@torch.no_grad()
def evaluate(model, loader, device: torch.device, threshold: float = 0.5, save_dir: str | Path | None = None) -> dict[str, float]:
    """在验证或测试集上评估变化检测指标，可选保存预测图。"""
    model.eval()
    cm_total = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="eval", leave=False):
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        mask = batch["mask"].to(device)
        outputs = model(t1, t2)
        pred = (outputs["prob"] > threshold).float()
        cm_total = merge_confusion(cm_total, confusion_matrix_binary(pred.cpu(), mask.cpu()))
        if save_dir is not None:
            names = batch["name"]
            for i, name in enumerate(names):
                save_tensor_image(pred[i], save_dir / name)
    metrics = {}
    metrics["precision"] = cm_total["tp"] / (cm_total["tp"] + cm_total["fp"] + 1e-6)
    metrics["recall"] = cm_total["tp"] / (cm_total["tp"] + cm_total["fn"] + 1e-6)
    metrics["f1"] = 2.0 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"] + 1e-6)
    metrics["iou"] = cm_total["tp"] / (cm_total["tp"] + cm_total["fp"] + cm_total["fn"] + 1e-6)
    total = cm_total["tp"] + cm_total["tn"] + cm_total["fp"] + cm_total["fn"]
    metrics["oa"] = (cm_total["tp"] + cm_total["tn"]) / (total + 1e-6)
    pe = ((cm_total["tp"] + cm_total["fp"]) * (cm_total["tp"] + cm_total["fn"]) + (cm_total["fn"] + cm_total["tn"]) * (cm_total["fp"] + cm_total["tn"])) / (total * total + 1e-6)
    metrics["kappa"] = (metrics["oa"] - pe) / (1.0 - pe + 1e-6)
    return metrics
