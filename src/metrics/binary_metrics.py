import torch


def confusion_matrix_binary(pred: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """计算二值变化检测混淆矩阵元素。"""
    pred = (pred > 0.5).bool()
    mask = (mask > 0.5).bool()
    tp = torch.logical_and(pred, mask).sum().item()
    tn = torch.logical_and(~pred, ~mask).sum().item()
    fp = torch.logical_and(pred, ~mask).sum().item()
    fn = torch.logical_and(~pred, mask).sum().item()
    return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


def precision(cm: dict[str, float]) -> float:
    """Precision。"""
    return cm["tp"] / (cm["tp"] + cm["fp"] + 1e-6)


def recall(cm: dict[str, float]) -> float:
    """Recall。"""
    return cm["tp"] / (cm["tp"] + cm["fn"] + 1e-6)


def f1_score(cm: dict[str, float]) -> float:
    """F1分数。"""
    p = precision(cm)
    r = recall(cm)
    return 2.0 * p * r / (p + r + 1e-6)


def iou(cm: dict[str, float]) -> float:
    """变化类IoU。"""
    return cm["tp"] / (cm["tp"] + cm["fp"] + cm["fn"] + 1e-6)


def oa(cm: dict[str, float]) -> float:
    """Overall Accuracy。"""
    total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
    return (cm["tp"] + cm["tn"]) / (total + 1e-6)


def kappa(cm: dict[str, float]) -> float:
    """Cohen's Kappa。"""
    total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
    po = oa(cm)
    pe = ((cm["tp"] + cm["fp"]) * (cm["tp"] + cm["fn"]) + (cm["fn"] + cm["tn"]) * (cm["fp"] + cm["tn"])) / (total * total + 1e-6)
    return (po - pe) / (1.0 - pe + 1e-6)


def merge_confusion(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """合并两个混淆矩阵字典。"""
    return {key: a.get(key, 0.0) + b.get(key, 0.0) for key in ("tp", "tn", "fp", "fn")}


def evaluate_binary_change(pred: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """输入pred与mask，返回Precision、Recall、F1、IoU、OA、Kappa。"""
    cm = confusion_matrix_binary(pred, mask)
    return {
        "precision": precision(cm),
        "recall": recall(cm),
        "f1": f1_score(cm),
        "iou": iou(cm),
        "oa": oa(cm),
        "kappa": kappa(cm),
    }
