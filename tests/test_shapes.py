import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.autoencoder import TinyAutoEncoder
from src.models.cignsi import CompactIGNSIControlEncoder
from src.models.decoder import CertifiedResidualGatedDecoder
from src.models.full_model import CIGNCDModel


def test_autoencoder_shape():
    model = TinyAutoEncoder(latent_ch=2)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out["latent"].shape == (2, 2, 8, 8)
    assert out["rec"].shape == (2, 3, 64, 64)


def test_cignsi_shape():
    model = CompactIGNSIControlEncoder(feat_ch=16, hidden_dim=8, patch_size=5)
    z = torch.randn(2, 16, 16, 16)
    out = model(z)
    assert out["prior"].shape == z.shape
    assert out["weight"].shape == (2, 1, 16, 16)
    assert out["residual_energy"].shape == (2, 1, 16, 16)
    assert out["gradient_energy"].shape == (2, 1, 16, 16)


def test_decoder_shape():
    model = CertifiedResidualGatedDecoder(feat_ch=16, hidden_ch=16)
    d_obs = torch.randn(2, 16, 16, 16)
    d_ref = torch.randn(2, 16, 16, 16)
    d_cert = torch.randn(2, 16, 16, 16)
    w1 = torch.rand(2, 1, 16, 16)
    out = model(d_obs, d_ref, d_cert, w1, out_size=(64, 64))
    assert out["logits"].shape == (2, 1, 64, 64)
    assert out["a_exp"].shape == (2, 1, 16, 16)
    assert out["a_cert"].shape == (2, 1, 16, 16)


def test_full_model_forward_and_backward():
    model = CIGNCDModel(feat_ch=16, control_ch=8, latent_ch=2, style_dim=16, hidden_ch=16, freeze_autoencoder=False)
    t1 = torch.randn(2, 3, 64, 64)
    t2 = torch.randn(2, 3, 64, 64)
    out = model(t1, t2)
    assert out["logits"].shape == (2, 1, 64, 64)
    assert out["prob"].shape == (2, 1, 64, 64)
    assert out["t2_ref"].shape == (2, 3, 64, 64)
    assert out["eps_pred"].shape == out["noise"].shape
    assert out["diffusion_loss"].ndim == 0
    loss = out["logits"].mean() + out["t2_ref"].mean()
    loss.backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_sum > 0
