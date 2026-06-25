import torch
import torch.nn as nn

from .alignment import ReferenceBoundaryAligner
from .autoencoder import TinyAutoEncoder
from .backbone import SharedBackbone
from .cignsi import CompactIGNSIControlEncoder, ControlAdapter
from .decoder import CertifiedResidualGatedDecoder
from .latent_editor import T1OnlyLatentEditor
from .style_encoder import T2StyleEncoder


class CIGNCDModel(nn.Module):
    """CIGN-CD完整模型。"""

    def __init__(
        self,
        feat_ch: int = 128,
        control_ch: int = 64,
        latent_ch: int = 4,
        style_dim: int = 128,
        hidden_ch: int = 128,
        freeze_autoencoder: bool = True,
        diffusion_steps: int = 1000,
        inference_steps: int = 50,
        sample_method: str = "ddim",
        edit_strength: float = 0.65,
    ):
        super().__init__()
        self.backbone = SharedBackbone(feat_ch=feat_ch)
        self.cignsi = CompactIGNSIControlEncoder(feat_ch=feat_ch)
        self.control_adapter = ControlAdapter(feat_ch=feat_ch, control_ch=control_ch)
        self.style_encoder = T2StyleEncoder(feat_ch=feat_ch, style_dim=style_dim)
        self.autoencoder = TinyAutoEncoder(latent_ch=latent_ch)
        self.latent_editor = T1OnlyLatentEditor(
            autoencoder=self.autoencoder,
            control_ch=control_ch,
            latent_ch=latent_ch,
            style_dim=style_dim,
            hidden_ch=hidden_ch,
            num_steps=diffusion_steps,
            inference_steps=inference_steps,
            sample_method=sample_method,
            edit_strength=edit_strength,
        )
        self.ref_aligner = ReferenceBoundaryAligner(feat_ch=feat_ch)
        self.decoder = CertifiedResidualGatedDecoder(feat_ch=feat_ch, hidden_ch=hidden_ch)
        if freeze_autoencoder:
            self.freeze_autoencoder()

    def freeze_autoencoder(self) -> None:
        """冻结自编码器参数，主模型训练时仅作为T1-only latent编辑空间使用。"""
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def load_autoencoder_weights(self, checkpoint_path: str, strict: bool = True) -> None:
        """从checkpoint加载自编码器权重。"""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        ae_state = {k.replace("autoencoder.", ""): v for k, v in state.items() if k.startswith("autoencoder.")}
        if not ae_state:
            ae_state = state
        self.autoencoder.load_state_dict(ae_state, strict=strict)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        输入:
            t1: [B, 3, H, W]
            t2: [B, 3, H, W]
        输出:
            包含logits、prob、参考图像、差异特征、门控图和结构先验的字典。
        """
        z1 = self.backbone(t1)
        z2 = self.backbone(t2)

        struct = self.cignsi(z1)
        p1 = struct["prior"]
        w1 = struct["weight"]

        target_latent_size = (t1.shape[-2] // 8, t1.shape[-1] // 8)
        c1 = self.control_adapter(p1, w1, target_size=target_latent_size)

        style = self.style_encoder(z2)
        style_vec = style["style_vec"]

        edit = self.latent_editor(t1, c1, style_vec)
        t2_ref = edit["t2_ref"]

        z2_ref = self.backbone(t2_ref)

        align = self.ref_aligner(z1, z2_ref, w1)
        z2_ref_align = align["z2_ref_align"]
        offset = align["offset"]

        d_obs = z2 - z1
        d_ref = z2_ref_align - z1
        d_cert = z2 - z2_ref_align

        dec = self.decoder(d_obs, d_ref, d_cert, w1, out_size=t1.shape[-2:])
        logits = dec["logits"]
        prob = torch.sigmoid(logits)

        return {
            "logits": logits,
            "prob": prob,
            "t2_ref": t2_ref,
            "x1": edit["x1"],
            "x_t": edit["x_t"],
            "x_ref": edit["x_ref"],
            "x0_pred": edit["x0_pred"],
            "diffusion_t": edit["t"],
            "sample_start_t": edit["sample_start_t"],
            "noise": edit["noise"],
            "eps_pred": edit["eps_pred"],
            "diffusion_loss": edit["diffusion_loss"],
            "z1": z1,
            "z2": z2,
            "z2_ref": z2_ref,
            "z2_ref_align": z2_ref_align,
            "offset": offset,
            "d_obs": d_obs,
            "d_ref": d_ref,
            "d_cert": d_cert,
            "a_exp": dec["a_exp"],
            "a_cert": dec["a_cert"],
            "s_exp": dec["s_exp"],
            "structure_weight": w1,
            "structure_prior": p1,
            "style_vec": style_vec,
            "style_low": style["z2_low"],
        }
