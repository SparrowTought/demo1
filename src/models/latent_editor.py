import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import TinyAutoEncoder
from .blocks import ConvGNAct, FiLMResBlock, TimeEmbedding


def _extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """按照时间步t取出扩散系数，并reshape成[B,1,1,1]。"""
    out = values.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


class LatentUNetDenoiser(nn.Module):
    """
    轻量latent UNet去噪器。

    输入:
        x_t: [B, latent_ch, H/8, W/8]
        c1: [B, control_ch, H/8, W/8]
        t: [B]
        style_vec: [B, style_dim]
    输出:
        eps_pred: [B, latent_ch, H/8, W/8]
    """

    def __init__(self, latent_ch: int = 4, control_ch: int = 64, style_dim: int = 128, hidden_ch: int = 128):
        super().__init__()
        self.control_proj = nn.Conv2d(control_ch, latent_ch, 1)
        self.in_proj = ConvGNAct(latent_ch * 2, hidden_ch)
        self.down1 = ConvGNAct(hidden_ch, hidden_ch, stride=2)
        self.down2 = ConvGNAct(hidden_ch, hidden_ch, stride=2)
        self.time_emb = TimeEmbedding(hidden_ch)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_ch, hidden_ch),
        )
        self.mid = nn.ModuleList([FiLMResBlock(hidden_ch, hidden_ch) for _ in range(4)])
        self.up1 = ConvGNAct(hidden_ch * 2, hidden_ch)
        self.up2 = ConvGNAct(hidden_ch * 2, hidden_ch)
        self.refine = nn.ModuleList([FiLMResBlock(hidden_ch, hidden_ch) for _ in range(2)])
        self.out = nn.Conv2d(hidden_ch, latent_ch, 3, padding=1)

    def forward(self, x_t: torch.Tensor, c1: torch.Tensor, t: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        """预测当前噪声eps_pred。"""
        control = self.control_proj(c1)
        h0 = self.in_proj(torch.cat([x_t, control], dim=1))
        h1 = self.down1(h0)
        h2 = self.down2(h1)
        cond = self.time_emb(t) + self.style_mlp(style_vec)
        h = h2
        for block in self.mid:
            h = block(h, cond)
        h = F.interpolate(h, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up1(torch.cat([h, h1], dim=1))
        h = F.interpolate(h, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up2(torch.cat([h, h0], dim=1))
        for block in self.refine:
            h = block(h, cond)
        return self.out(h)


class T1OnlyLatentEditor(nn.Module):
    """
    完整T1-only latent diffusion编辑器。

    该模块不编码T2，也不以T2 latent作为扩散目标。训练阶段使用标准DDPM噪声预测目标，
    学习在T1结构条件C1和T2低频风格c2调制下恢复T1结构latent；变化检测总损失和
    未变化区域参考一致性损失会进一步把采样结果推向无变化后时相参考T2_ref。
    """

    def __init__(
        self,
        autoencoder: TinyAutoEncoder,
        control_ch: int = 64,
        latent_ch: int = 4,
        style_dim: int = 128,
        hidden_ch: int = 128,
        num_steps: int = 1000,
        inference_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        sample_method: str = "ddim",
        edit_strength: float = 0.65,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.num_steps = int(num_steps)
        self.inference_steps = int(inference_steps)
        self.sample_method = sample_method
        self.edit_strength = float(edit_strength)
        self.denoiser = LatentUNetDenoiser(
            latent_ch=latent_ch,
            control_ch=control_ch,
            style_dim=style_dim,
            hidden_ch=hidden_ch,
        )

        betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """前向扩散: q(x_t | x_0)。"""
        sqrt_ab = _extract(self.sqrt_alpha_bars, t, x0.shape).to(dtype=x0.dtype)
        sqrt_omb = _extract(self.sqrt_one_minus_alpha_bars, t, x0.shape).to(dtype=x0.dtype)
        return sqrt_ab * x0 + sqrt_omb * noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, c1: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        """条件去噪器预测噪声epsilon。"""
        return self.denoiser(x_t, c1, t, style_vec)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """由x_t和预测噪声反推x_0估计。"""
        sqrt_ab = _extract(self.sqrt_alpha_bars, t, x_t.shape).to(dtype=x_t.dtype)
        sqrt_omb = _extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape).to(dtype=x_t.dtype)
        return (x_t - sqrt_omb * eps_pred) / (sqrt_ab + 1e-8)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, c1: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        """DDPM单步反向采样: p(x_{t-1} | x_t)。"""
        eps_pred = self.predict_noise(x_t, t, c1, style_vec)
        beta_t = _extract(self.betas, t, x_t.shape).to(dtype=x_t.dtype)
        sqrt_one_minus_ab = _extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape).to(dtype=x_t.dtype)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, t, x_t.shape).to(dtype=x_t.dtype)
        model_mean = sqrt_recip_alpha * (x_t - beta_t * eps_pred / (sqrt_one_minus_ab + 1e-8))
        posterior_var = _extract(self.posterior_variance, t, x_t.shape).to(dtype=x_t.dtype)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(t.shape[0], *([1] * (x_t.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, c1: torch.Tensor, style_vec: torch.Tensor) -> torch.Tensor:
        """确定性DDIM单步采样，eta=0。"""
        eps_pred = self.predict_noise(x_t, t, c1, style_vec)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        ab_prev = _extract(self.alpha_bars, t_prev.clamp(min=0), x_t.shape).to(dtype=x_t.dtype)
        eps_coef = torch.sqrt(torch.clamp(1.0 - ab_prev, min=0.0))
        x_prev = torch.sqrt(ab_prev) * x0_pred + eps_coef * eps_pred
        done_mask = (t_prev < 0).float().view(t.shape[0], *([1] * (x_t.ndim - 1)))
        return done_mask * x0_pred + (1.0 - done_mask) * x_prev

    @torch.no_grad()
    def sample_reference_latent(self, x1: torch.Tensor, c1: torch.Tensor, style_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从加噪T1 latent开始进行多步反向采样，得到参考latent。

        返回:
            x_ref: [B, latent_ch, H/8, W/8]
            sample_start_t: [B]
        """
        b = x1.shape[0]
        max_t = max(1, min(self.num_steps - 1, int((self.num_steps - 1) * self.edit_strength)))
        steps = max(2, min(self.inference_steps, max_t + 1))
        schedule = torch.linspace(max_t, 0, steps, device=x1.device).round().long().unique_consecutive()
        if schedule[-1].item() != 0:
            schedule = torch.cat([schedule, torch.zeros(1, device=x1.device, dtype=torch.long)])
        start_t = torch.full((b,), int(schedule[0].item()), device=x1.device, dtype=torch.long)
        x_t = self.q_sample(x1, start_t, torch.randn_like(x1))
        for idx, cur in enumerate(schedule):
            t = torch.full((b,), int(cur.item()), device=x1.device, dtype=torch.long)
            if self.sample_method == "ddpm":
                next_x = self.p_sample(x_t, t, c1, style_vec)
            else:
                prev_value = int(schedule[idx + 1].item()) if idx + 1 < len(schedule) else -1
                t_prev = torch.full((b,), prev_value, device=x1.device, dtype=torch.long)
                next_x = self.ddim_step(x_t, t, t_prev, c1, style_vec)
            x_t = next_x
        return x_t, start_t

    def forward(self, t1: torch.Tensor, c1: torch.Tensor, style_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        输入:
            t1: [B, 3, H, W]
            c1: [B, control_ch, H/8, W/8]
            style_vec: [B, style_dim]
        输出:
            t2_ref: [B, 3, H, W]，以及DDPM训练和采样所需的中间量。
        """
        b = t1.shape[0]
        x1 = self.autoencoder.encode(t1)
        t = torch.randint(0, self.num_steps, (b,), device=t1.device)
        noise = torch.randn_like(x1)
        x_t = self.q_sample(x1, t, noise)
        eps_pred = self.predict_noise(x_t, t, c1, style_vec)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)

        if self.training:
            x_ref = x0_pred
            sample_start_t = t
        else:
            x_ref, sample_start_t = self.sample_reference_latent(x1, c1, style_vec)

        t2_ref = self.autoencoder.decode(x_ref)
        diffusion_loss = F.mse_loss(eps_pred, noise)
        return {
            "t2_ref": t2_ref,
            "x1": x1,
            "x_t": x_t,
            "x_ref": x_ref,
            "x0_pred": x0_pred,
            "t": t,
            "sample_start_t": sample_start_t,
            "noise": noise,
            "eps_pred": eps_pred,
            "diffusion_loss": diffusion_loss,
        }
