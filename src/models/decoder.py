import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, ResBlock


class CertifiedResidualGatedDecoder(nn.Module):
    """
    认证残差门控解码器。

    D_ref负责解释，D_cert负责认证，D_obs作为辅助观测差异。
    """

    def __init__(self, feat_ch: int = 128, hidden_ch: int = 128):
        super().__init__()
        self.exp_gate = nn.Sequential(ConvGNAct(3, 16), nn.Conv2d(16, 1, 1))
        self.cert_gate = nn.Sequential(ConvGNAct(3, 16), nn.Conv2d(16, 1, 1))
        self.obs_proj = nn.Sequential(ConvGNAct(feat_ch, hidden_ch), ResBlock(hidden_ch))
        self.cert_proj = nn.Sequential(ConvGNAct(feat_ch, hidden_ch), ResBlock(hidden_ch))
        self.fuse = nn.Sequential(
            ConvGNAct(hidden_ch * 2 + 3, hidden_ch),
            ResBlock(hidden_ch),
            ConvGNAct(hidden_ch, hidden_ch // 2),
        )
        self.main_head = nn.Sequential(ConvGNAct(hidden_ch // 2, hidden_ch // 4), nn.Conv2d(hidden_ch // 4, 1, 1))
        self.cert_head = nn.Sequential(ConvGNAct(hidden_ch, hidden_ch // 2), nn.Conv2d(hidden_ch // 2, 1, 1))

    def forward(
        self,
        d_obs: torch.Tensor,
        d_ref: torch.Tensor,
        d_cert: torch.Tensor,
        w1: torch.Tensor,
        out_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """
        输入:
            d_obs/d_ref/d_cert: [B, feat_ch, H/4, W/4]
            w1: [B, 1, H/4, W/4]
            out_size: (H, W)
        输出:
            logits: [B, 1, H, W]，以及a_exp、a_cert、s_exp。
        """
        dot = torch.sum(d_obs * d_ref, dim=1, keepdim=True)
        norm_obs = torch.norm(d_obs, p=2, dim=1, keepdim=True)
        norm_ref = torch.norm(d_ref, p=2, dim=1, keepdim=True)
        s_exp = dot / (norm_obs * norm_ref + 1e-6)

        ref_mag = torch.norm(d_ref, p=2, dim=1, keepdim=True)
        cert_mag = torch.norm(d_cert, p=2, dim=1, keepdim=True)
        a_exp = torch.sigmoid(self.exp_gate(torch.cat([ref_mag, s_exp, w1], dim=1)))
        a_cert = torch.sigmoid(self.cert_gate(torch.cat([cert_mag, 1.0 - a_exp, w1], dim=1)))

        f_obs = self.obs_proj(d_obs)
        f_cert = self.cert_proj(d_cert)
        obs_mod = (1.0 - a_exp) * f_obs
        cert_mod = (1.0 + a_cert) * f_cert
        fused = self.fuse(torch.cat([cert_mod, obs_mod, a_cert, a_exp, w1], dim=1))
        l_main = self.main_head(fused)
        l_cert = self.cert_head(cert_mod)
        logits = F.interpolate(l_main + l_cert, size=out_size, mode="bilinear", align_corners=False)
        return {"logits": logits, "a_exp": a_exp, "a_cert": a_cert, "s_exp": s_exp}
