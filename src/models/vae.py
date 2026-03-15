"""
vae.py  –  Task 2: Variational Autoencoder (VAE) for Multi-Genre Music Generation
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Mathematical Formulation:
    Encoder:   q_φ(z|X) = N(μ(X), σ²(X))
    Sampling:  z = μ + σ ⊙ ε,    ε ~ N(0, I)     (reparameterisation trick)
    Decoder:   p_θ(X̂|z)
    Loss:      L_VAE = L_recon + β·D_KL(q_φ(z|X) ‖ p(z))
    KL:        D_KL = -½ Σ (1 + log σ² - μ² - σ²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import (VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_NUM_LAYERS,
                        VAE_DROPOUT, VAE_BETA, NUM_PITCHES, SEQUENCE_LENGTH,
                        NUM_GENRES)


class VAEEncoder(nn.Module):
    """
    Encodes X → (μ, log σ²) for the variational posterior q_φ(z|X).

    Architecture:
        X → Linear → Bi-LSTM → Linear → (μ HEAD, log-σ² HEAD)
    """
    def __init__(self, input_dim: int = NUM_PITCHES,
                 hidden_dim: int = VAE_HIDDEN_DIM,
                 latent_dim: int = VAE_LATENT_DIM,
                 num_layers: int = VAE_NUM_LAYERS,
                 dropout: float = VAE_DROPOUT,
                 num_genres: int = NUM_GENRES):
        super().__init__()
        self.genre_embed = nn.Embedding(num_genres, 32)
        self.input_proj  = nn.Linear(input_dim + 32, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, genre: torch.Tensor):
        """
        Args:
            x:     (B, T, P)
            genre: (B,)  integer genre labels
        Returns:
            mu, log_var: each (B, latent_dim)
        """
        g = self.genre_embed(genre)                         # (B, 32)
        g = g.unsqueeze(1).expand(-1, x.size(1), -1)       # (B, T, 32)
        h = F.relu(self.input_proj(torch.cat([x, g], dim=-1)))  # (B, T, H)
        _, (hidden, _) = self.lstm(h)
        fwd, bwd = hidden[-2], hidden[-1]
        shared   = self.fc_shared(torch.cat([fwd, bwd], dim=-1))
        mu       = self.fc_mu(shared)
        log_var  = self.fc_log_var(shared)
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    Decodes z + genre_embedding → reconstructed sequence X̂.
    """
    def __init__(self, input_dim: int = NUM_PITCHES,
                 hidden_dim: int = VAE_HIDDEN_DIM,
                 latent_dim: int = VAE_LATENT_DIM,
                 num_layers: int = VAE_NUM_LAYERS,
                 dropout: float = VAE_DROPOUT,
                 seq_len: int = SEQUENCE_LENGTH,
                 num_genres: int = NUM_GENRES):
        super().__init__()
        self.seq_len = seq_len
        self.genre_embed = nn.Embedding(num_genres, 32)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, genre: torch.Tensor) -> torch.Tensor:
        g   = self.genre_embed(genre)                     # (B, 32)
        inp = self.latent_proj(torch.cat([z, g], dim=-1)) # (B, H)
        rep = inp.unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.lstm(rep)
        return self.out_proj(out)                          # (B, T, P)


class MusicVAE(nn.Module):
    """
    Full VAE for multi-genre music generation.
    Task 2 (Medium): diverse generation with latent interpolation.

    L_VAE = L_recon + β·D_KL
    D_KL  = -½ Σ (1 + log σ² - μ² - σ²)   (closed-form for Gaussian prior)
    """
    def __init__(self, input_dim: int = NUM_PITCHES,
                 hidden_dim: int = VAE_HIDDEN_DIM,
                 latent_dim: int = VAE_LATENT_DIM,
                 num_layers: int = VAE_NUM_LAYERS,
                 dropout: float = VAE_DROPOUT,
                 beta: float = VAE_BETA,
                 seq_len: int = SEQUENCE_LENGTH,
                 num_genres: int = NUM_GENRES):
        super().__init__()
        self.beta      = beta
        self.latent_dim = latent_dim
        self.num_genres = num_genres
        self.encoder   = VAEEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout, num_genres)
        self.decoder   = VAEDecoder(input_dim, hidden_dim, latent_dim, num_layers, dropout, seq_len, num_genres)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick:  z = μ + σ ⊙ ε,  ε ~ N(0, I)
        Enables gradient flow through sampling.
        """
        std = torch.exp(0.5 * log_var)     # σ = exp(log σ / 2)
        eps = torch.randn_like(std)         # ε ~ N(0, I)
        return mu + std * eps              # z

    def forward(self, x: torch.Tensor, genre: torch.Tensor) -> tuple:
        """
        Returns:
            x_hat:   (B, T, P)
            mu:      (B, latent_dim)
            log_var: (B, latent_dim)
        """
        mu, log_var = self.encoder(x, genre)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decoder(z, genre)
        return x_hat, mu, log_var

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor) -> dict:
        """
        VAE loss = Reconstruction loss + β·KL divergence
        KL  = -½ Σ_j (1 + log σ²_j - μ²_j - σ²_j)
        """
        L_recon = F.mse_loss(x_hat, x, reduction='mean')
        # Closed-form KL divergence against standard Gaussian N(0,I)
        L_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        L_kl = L_kl / (x.size(0) * x.size(1))    # per element normalisation
        L_total = L_recon + self.beta * L_kl
        return {"loss": L_total, "recon": L_recon, "kl": L_kl}

    def generate(self, n_samples: int = 1, genre: int = 0,
                 device: str = 'cpu') -> torch.Tensor:
        """
        Generate music by sampling z ~ N(0, I).
        """
        self.eval()
        with torch.no_grad():
            z   = torch.randn(n_samples, self.latent_dim, device=device)
            gen = torch.full((n_samples,), genre, dtype=torch.long, device=device)
            return self.decoder(z, gen)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    genre: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """
        Latent space interpolation between two music segments.
        z_interp = (1-α)·z₁ + α·z₂   for α ∈ {0, 1/(n-1), …, 1}
        """
        self.eval()
        with torch.no_grad():
            mu1, _ = self.encoder(x1, genre)
            mu2, _ = self.encoder(x2, genre)
            alphas  = torch.linspace(0, 1, steps, device=x1.device)
            outputs = []
            for a in alphas:
                z_interp = (1 - a) * mu1 + a * mu2
                out      = self.decoder(z_interp, genre)
                outputs.append(out)
            return torch.stack(outputs, dim=1)  # (B, steps, T, P)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MusicVAE()
    print(f"MusicVAE — Trainable parameters: {model.count_parameters():,}")
    B, T, P = 4, SEQUENCE_LENGTH, NUM_PITCHES
    x = torch.rand(B, T, P)
    g = torch.randint(0, NUM_GENRES, (B,))
    x_hat, mu, log_var = model(x, g)
    losses = model.compute_loss(x, x_hat, mu, log_var)
    print(f"L_total={losses['loss']:.4f}  L_recon={losses['recon']:.4f}  L_KL={losses['kl']:.4f}")
    gen = model.generate(3, genre=0)
    print(f"Generated shape: {gen.shape}")
