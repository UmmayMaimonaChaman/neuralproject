"""
autoencoder.py  –  Task 1: LSTM Autoencoder for Single-Genre Music Generation
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Mathematical Formulation:
    Encoder:   z  = f_φ(X)           (LSTM hidden state)
    Decoder:   X̂  = g_θ(z)           (LSTM sequential reconstruction)
    Loss:      L_AE = Σ‖x_t − x̂_t‖²  (MSE reconstruction loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import (AE_LATENT_DIM, AE_HIDDEN_DIM, AE_NUM_LAYERS,
                        AE_DROPOUT, NUM_PITCHES, SEQUENCE_LENGTH)


class LSTMEncoder(nn.Module):
    """
    Encodes a piano-roll sequence X ∈ ℝ^{T × P} into a latent vector z.

    Architecture:
        Input  →  Linear projection  →  Bi-LSTM ×2  →  Linear  →  z
    """
    def __init__(self, input_dim: int = NUM_PITCHES,
                 hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM,
                 num_layers: int = AE_NUM_LAYERS,
                 dropout: float = AE_DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            z: (batch, latent_dim)
        """
        h = F.relu(self.input_proj(x))            # (B, T, H)
        _, (hidden, _) = self.lstm(h)              # hidden: (layers*2, B, H)
        # Concatenate forward and backward last hidden states
        fwd = hidden[-2]                            # (B, H)
        bwd = hidden[-1]                            # (B, H)
        z = self.to_latent(torch.cat([fwd, bwd], dim=-1))  # (B, latent)
        return z


class LSTMDecoder(nn.Module):
    """
    Decodes a latent vector z into a reconstructed piano-roll sequence X̂.

    Architecture:
        z  →  Linear  →  repeat T times  →  LSTM ×2  →  Linear  →  Sigmoid  →  X̂
    """
    def __init__(self, input_dim: int = NUM_PITCHES,
                 hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM,
                 num_layers: int = AE_NUM_LAYERS,
                 dropout: float = AE_DROPOUT,
                 seq_len: int = SEQUENCE_LENGTH):
        super().__init__()
        self.seq_len = seq_len
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_hat: (batch, T, input_dim)
        """
        h = self.latent_to_hidden(z)               # (B, H)
        repeated = h.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, H)
        out, _ = self.lstm(repeated)                # (B, T, H)
        x_hat = self.out_proj(out)                  # (B, T, P)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder.
    Task 1 (Easy): single-genre reconstruction and generation.

    Loss: L_AE = (1/T) * Σ‖x_t − x̂_t‖²₂
    """
    def __init__(self, input_dim: int = NUM_PITCHES,
                 hidden_dim: int = AE_HIDDEN_DIM,
                 latent_dim: int = AE_LATENT_DIM,
                 num_layers: int = AE_NUM_LAYERS,
                 dropout: float = AE_DROPOUT,
                 seq_len: int = SEQUENCE_LENGTH):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(input_dim, hidden_dim, latent_dim, num_layers, dropout, seq_len)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, T, P)  piano-roll batch
        Returns:
            x_hat: (B, T, P) reconstruction
            z:     (B, latent_dim) latent codes
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: L_AE = MSE(X, X̂)
        = (1/(T*P)) * Σ_{t,p} (x_{t,p} - x̂_{t,p})²
        """
        return F.mse_loss(x_hat, x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time encoding."""
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Inference-time decoding."""
        with torch.no_grad():
            return self.decoder(z)

    def generate(self, n_samples: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        Generate music by sampling random latent vectors z ~ N(0, I).

        Returns:
            piano_rolls: (n_samples, T, P)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            return self.decode(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LSTMAutoencoder()
    print(f"LSTM Autoencoder — Total trainable parameters: {model.count_parameters():,}")
    dummy = torch.randn(4, SEQUENCE_LENGTH, NUM_PITCHES)
    x_hat, z = model(dummy)
    loss = model.compute_loss(dummy, x_hat)
    print(f"Input shape : {dummy.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {x_hat.shape}")
    print(f"Loss        : {loss.item():.6f}")
