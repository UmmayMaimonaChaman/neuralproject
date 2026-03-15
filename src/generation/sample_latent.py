"""
sample_latent.py  –  Latent Space Sampling and Interpolation
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Provides utilities for:
    1. Random latent sampling for generation
    2. Latent space interpolation between two pieces
    3. Genre-conditioned latent grid exploration
"""

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (VAE_LATENT_DIM, AE_LATENT_DIM,
                        SEQUENCE_LENGTH, NUM_PITCHES, NUM_GENRES, GENRES,
                        MIDI_OUT_DIR, PLOTS_DIR, RANDOM_SEED)
from src.models.vae import MusicVAE
from src.generation.midi_export import piano_roll_to_midi


def sample_and_generate_vae(model: MusicVAE, n_samples: int = 8,
                             device: str = 'cpu') -> None:
    """
    Generate music by sampling z ~ N(0, I) for each genre.
    Saves MIDI outputs and visualises piano-rolls.
    """
    model.eval()
    os.makedirs(MIDI_OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,    exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(min(n_samples, 8)):
        genre = i % NUM_GENRES
        roll  = model.generate(1, genre=genre, device=device)
        roll  = roll.squeeze(0).cpu().numpy()               # (T, P)
        path  = os.path.join(MIDI_OUT_DIR, f'latent_sample_{GENRES[genre]}_{i+1}.mid')
        piano_roll_to_midi(roll.T, fs=4.0, path=path)
        axes[i].imshow(roll.T, aspect='auto', origin='lower',
                       cmap='Blues', vmin=0, vmax=1)
        axes[i].set_title(f'{GENRES[genre].capitalize()} (z~N(0,I))', fontsize=9)
        axes[i].set_xlabel('Time steps')
        axes[i].set_ylabel('Pitch bins')

    fig.suptitle('VAE Latent Sampling – Generated Piano Rolls', fontsize=14)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'vae_latent_samples.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SampleLatent] Piano-roll visualisation saved → {path}")


def interpolate_latent(model: MusicVAE, device: str = 'cpu',
                        n_steps: int = 8) -> None:
    """
    Latent interpolation between two genres.
    z_interp = (1-α)·z₁ + α·z₂   α ∈ [0, 1]
    """
    torch.manual_seed(RANDOM_SEED)
    model.eval()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Sample two latent points from different genres
    z1 = torch.randn(1, VAE_LATENT_DIM, device=device)
    z2 = torch.randn(1, VAE_LATENT_DIM, device=device)
    g  = torch.zeros(1, dtype=torch.long, device=device)   # classical genre

    alphas  = torch.linspace(0.0, 1.0, n_steps, device=device)
    rolls   = []
    with torch.no_grad():
        for a in alphas:
            z_i = (1 - a) * z1 + a * z2
            r   = model.decoder(z_i, g).squeeze(0).cpu().numpy()  # (T, P)
            rolls.append(r)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, (roll, alpha) in enumerate(zip(rolls, alphas.tolist())):
        axes[i].imshow(roll.T, aspect='auto', origin='lower', cmap='Purples')
        axes[i].set_title(f'α = {alpha:.2f}', fontsize=10)
        axes[i].axis('off')

    fig.suptitle('Latent Space Interpolation  z=(1-α)z₁ + αz₂', fontsize=14)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'latent_interpolation.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SampleLatent] Interpolation plot saved → {path}")


if __name__ == "__main__":
    model = MusicVAE()
    sample_and_generate_vae(model, n_samples=8)
    interpolate_latent(model, n_steps=8)
