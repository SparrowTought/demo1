import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses.losses import autoencoder_loss, cign_cd_loss
from src.metrics.binary_metrics import evaluate_binary_change
from src.models.full_model import CIGNCDModel


def test_autoencoder_loss_backward():
    x = torch.randn(2, 3, 32, 32)
    rec = torch.tanh(torch.randn(2, 3, 32, 32, requires_grad=True))
    loss = autoencoder_loss(x, rec)
    assert loss.ndim == 0
    loss.backward()
    assert rec.grad is None or torch.isfinite(loss)


def test_cign_cd_loss_backward():
    model = CIGNCDModel(feat_ch=16, control_ch=8, latent_ch=2, style_dim=16, hidden_ch=16, freeze_autoencoder=False)
    t1 = torch.randn(2, 3, 64, 64)
    t2 = torch.randn(2, 3, 64, 64)
    mask = (torch.rand(2, 1, 64, 64) > 0.7).float()
    outputs = model(t1, t2)
    losses = cign_cd_loss(outputs, t1, t2, mask)
    for key in ("loss", "cd_loss", "ref_loss", "diff_loss"):
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])
    losses["loss"].backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_sum > 0


def test_metrics_compute():
    pred = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.float32)
    mask = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32)
    metrics = evaluate_binary_change(pred, mask)
    for key in ("precision", "recall", "f1", "iou", "oa", "kappa"):
        assert key in metrics
        assert isinstance(metrics[key], float)
