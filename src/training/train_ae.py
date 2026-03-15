"""
train_ae.py  –  Training Script for Task 1: LSTM Autoencoder
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Algorithm 1 from PDF:
    for epoch = 1 to E:
        for each batch X in D:
            z = f_φ(X)            # encode
            X̂ = g_θ(z)            # decode
            L_AE = ‖X − X̂‖²     # reconstruction loss
            (φ,θ) ← (φ,θ) − η∇L_AE  # gradient descent
    Generate new music by sampling latent codes z
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper') # Professional white background
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (AE_EPOCHS, AE_BATCH_SIZE, AE_LR, AE_LATENT_DIM,
                        SEQUENCE_LENGTH, NUM_PITCHES, RANDOM_SEED,
                        PLOTS_DIR, MIDI_OUT_DIR)
from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.midi_parser import MIDIParser
from src.preprocessing.piano_roll import note_events_to_piano_roll, segment_piano_roll


def build_dataset():
    """Build piano-roll dataset from preprocessed real data."""
    from src.config import TRAIN_TEST_DIR
    X_path = os.path.join(TRAIN_TEST_DIR, 'ae_train.npy')
    if not os.path.exists(X_path):
        raise FileNotFoundError(f"Training data not found at {X_path}. Run preprocessing first.")
    
    data = np.load(X_path) # (N, T, P)
    tensor = torch.tensor(data, dtype=torch.float32)
    return TensorDataset(tensor)


def train_autoencoder(epochs: int = AE_EPOCHS, device: str = 'cpu'):
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    os.makedirs(MIDI_OUT_DIR, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────
    print("[AE] Building dataset from REAL data ...")
    dataset  = build_dataset()
    n_train  = int(0.85 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_dl = DataLoader(train_ds, batch_size=AE_BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=AE_BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"[AE] Train: {n_train} | Val: {n_val} samples")

    # ── Model ─────────────────────────────────────────────────
    model = LSTMAutoencoder().to(device)
    print(f"[AE] Parameters: {model.count_parameters():,}")
    optimiser = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for (xb,) in train_dl:
            xb = xb.to(device)
            optimiser.zero_grad()
            x_hat, z = model(xb)
            loss = model.compute_loss(xb, x_hat)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_train = epoch_loss / len(train_dl)

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_dl:
                xb = xb.to(device)
                x_hat, _ = model(xb)
                val_loss += model.compute_loss(xb, x_hat).item()
        avg_val = val_loss / len(val_dl)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(PLOTS_DIR, '..', 'ae_best.pt'))

        if epoch % 10 == 0 or epoch == 1:
            print(f"[AE] Epoch {epoch:3d}/{epochs}  "
                  f"Train={avg_train:.6f}  Val={avg_val:.6f}  LR={scheduler.get_last_lr()[0]:.2e}")

    # ── Loss Curve ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train Loss', color='royalblue')
    ax.plot(val_losses,   label='Val Loss',   color='tomato', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Task 1 – LSTM Autoencoder Reconstruction Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'ae_loss_curve.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[AE] Loss curve saved → {path}")
    print(f"[AE] Best validation loss: {best_val:.6f}")

    # ── Generate MIDI samples ─────────────────────────────────
    model.eval()
    from src.generation.midi_export import piano_roll_to_midi
    for i in range(5):
        roll = model.generate(1, device=device).squeeze(0).cpu().numpy()  # (T, P)
        piano_roll_to_midi(roll.T, fs=4.0,
                           path=os.path.join(MIDI_OUT_DIR, f'ae_generated_{i+1}.mid'))
    print("[AE] Generated 5 MIDI samples.")

    return model, train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=AE_EPOCHS)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train_autoencoder(args.epochs, args.device)
