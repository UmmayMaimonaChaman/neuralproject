"""
diffusion.py  –  Placeholder Diffusion Model (optional advanced extension)
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

This file provides a minimal score-based diffusion model stub.
Not part of the graded tasks (Tasks 1-4 use AE, VAE, Transformer, RLHF).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import NUM_PITCHES, SEQUENCE_LENGTH


class SimpleDenoisingNetwork(nn.Module):
    """
    Minimal UNet-style denoising network for diffusion.
    Predicts noise ε given noisy input x_t and diffusion timestep t.
    """
    def __init__(self, input_dim: int = NUM_PITCHES, hidden: int = 256):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.net = nn.Sequential(
            nn.Linear(input_dim + 64, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, input_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, T, P),  t: (B, 1) normalised timestep."""
        te = self.time_embed(t.unsqueeze(-1).float())           # (B, 64)
        te = te.unsqueeze(1).expand(-1, x.size(1), -1)          # (B, T, 64)
        return self.net(torch.cat([x, te], dim=-1))              # (B, T, P)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion with cosine noise schedule.
    Placeholder – not used in main tasks.
    """
    def __init__(self, model: SimpleDenoisingNetwork, T: int = 1000):
        super().__init__()
        self.model = model
        self.T     = T
        # Cosine schedule beta_t
        steps  = torch.arange(T + 1, dtype=torch.float) / T
        alphas = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        betas  = 1 - alphas[1:] / alphas[:-1]
        self.register_buffer('betas', betas.clamp(0.0001, 0.9999))
        self.register_buffer('alphas_bar', torch.cumprod(1 - betas, dim=0))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """Forward diffusion: q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)"""
        a_bar = self.alphas_bar[t].view(-1, 1, 1)
        eps   = torch.randn_like(x0)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps, eps

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute diffusion training loss L = E‖ε − ε_θ(x_t, t)‖²"""
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps = self.q_sample(x0, t)
        t_norm = (t.float() / self.T).unsqueeze(-1)
        eps_pred = self.model(x_t, t_norm.unsqueeze(1).expand_as(x_t[..., :1]).squeeze(-1))
        return F.mse_loss(eps_pred, eps)
