"""
train_vae.py  –  Training Script for Task 2: VAE Multi-Genre Generator
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Algorithm 2 from PDF:
    for epoch = 1 to E:
        for each batch (X, genre) in D:
            (μ, σ) = Encoder(X)
            z = μ + σ⊙ε,  ε~N(0,I)     # reparameterisation
            X̂ = Decoder(z)
            L_recon = ‖X − X̂‖²
            L_KL    = D_KL(q(z|X) ‖ p(z))
            L_VAE   = L_recon + β·L_KL
            (φ,θ) ← (φ,θ) − η∇L_VAE
"""

import os, sys, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-bright') # Colorful white background

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (VAE_EPOCHS, VAE_BATCH_SIZE, VAE_LR, VAE_BETA,
                        SEQUENCE_LENGTH, NUM_PITCHES, NUM_GENRES, RANDOM_SEED,
                        PLOTS_DIR, MIDI_OUT_DIR, GENRES)
from src.models.vae import MusicVAE
from src.preprocessing.midi_parser import MIDIParser
from src.preprocessing.piano_roll import note_events_to_piano_roll, segment_piano_roll


def build_multi_genre_dataset():
    """Build multi-genre dataset from preprocessed real data."""
    from src.config import TRAIN_TEST_DIR
    X_path = os.path.join(TRAIN_TEST_DIR, 'ae_train.npy')
    y_path = os.path.join(TRAIN_TEST_DIR, 'genres_train.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Training data or labels not found. Run preprocessing first.")
        
    X_data = np.load(X_path) # (N, T, P)
    y_data = np.load(y_path) # (N,)
    
    X = torch.tensor(X_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.long)
    return TensorDataset(X, y)


def train_vae(epochs: int = VAE_EPOCHS, device: str = 'cpu'):
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(PLOTS_DIR,    exist_ok=True)
    os.makedirs(MIDI_OUT_DIR, exist_ok=True)

    print("[VAE] Building dataset from REAL multi-genre data ...")
    dataset   = build_multi_genre_dataset()
    n_train   = int(0.85 * len(dataset))
    n_val     = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_dl = DataLoader(train_ds, batch_size=VAE_BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=VAE_BATCH_SIZE, shuffle=False)
    print(f"[VAE] Train={n_train} | Val={n_val}")

    model     = MusicVAE(beta=VAE_BETA).to(device)
    print(f"[VAE] Parameters: {model.count_parameters():,}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=VAE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=VAE_LR, steps_per_epoch=len(train_dl), epochs=epochs)

    # KL annealing: β_eff = β * min(1, epoch/warmup)
    WARMUP = 10

    hist = {"train_loss": [], "val_loss": [], "kl": [], "recon": []}
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        beta_eff = VAE_BETA * min(1.0, epoch / WARMUP)
        model.beta = beta_eff

        # ── Train ─────────────────────────────────────────────
        model.train()
        t_loss = r_loss = k_loss = 0.0
        for (xb, gb) in train_dl:
            xb, gb = xb.to(device), gb.to(device)
            optimiser.zero_grad()
            x_hat, mu, log_var = model(xb, gb)
            losses = model.compute_loss(xb, x_hat, mu, log_var)
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            t_loss += losses["loss"].item()
            r_loss += losses["recon"].item()
            k_loss += losses["kl"].item()

        n = len(train_dl)
        hist["train_loss"].append(t_loss / n)
        hist["recon"].append(r_loss / n)
        hist["kl"].append(k_loss / n)

        # ── Validation ────────────────────────────────────────
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for (xb, gb) in val_dl:
                xb, gb = xb.to(device), gb.to(device)
                x_hat, mu, lv = model(xb, gb)
                v_loss += model.compute_loss(xb, x_hat, mu, lv)["loss"].item()
        avg_val = v_loss / len(val_dl)
        hist["val_loss"].append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(PLOTS_DIR, '..', 'vae_best.pt'))

        if epoch % 10 == 0 or epoch == 1:
            print(f"[VAE] Ep {epoch:3d}/{epochs}  "
                  f"Loss={hist['train_loss'][-1]:.4f}  "
                  f"Recon={hist['recon'][-1]:.4f}  KL={hist['kl'][-1]:.4f}  "
                  f"Val={avg_val:.4f}  β={beta_eff:.3f}")

    # ── Plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist["train_loss"], label='Train', color='royalblue')
    axes[0].plot(hist["val_loss"],   label='Val',   color='tomato', ls='--')
    axes[0].set_title('VAE Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["recon"], color='green')
    axes[1].set_title('Reconstruction Loss'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(hist["kl"], color='purple')
    axes[2].set_title('KL Divergence'); axes[2].grid(True, alpha=0.3)

    for ax in axes: ax.set_xlabel('Epoch')
    fig.suptitle('Task 2 – Music VAE Training', fontsize=14)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'vae_loss_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[VAE] Plots saved → {path}")

    # ── Generate 8 multi-genre samples ────────────────────────
    from src.generation.midi_export import piano_roll_to_midi
    model.eval()
    for i, genre_idx in enumerate(range(NUM_GENRES)):
        roll = model.generate(1, genre=genre_idx, device=device)
        roll = roll.squeeze(0).cpu().numpy()   # (T, P)
        fname = os.path.join(MIDI_OUT_DIR, f'vae_{GENRES[genre_idx]}_gen_{i+1}.mid')
        piano_roll_to_midi(roll.T, fs=4.0, path=fname)
    # 3 extra random genre samples to reach 8
    for i in range(3):
        g = random.randint(0, NUM_GENRES - 1)
        roll = model.generate(1, genre=g, device=device).squeeze(0).cpu().numpy()
        fname = os.path.join(MIDI_OUT_DIR, f'vae_random_genre_{i+1}.mid')
        piano_roll_to_midi(roll.T, fs=4.0, path=fname)
    print("[VAE] Generated 8 multi-genre MIDI samples.")

    return model, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=VAE_EPOCHS)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()
    train_vae(args.epochs, args.device)
