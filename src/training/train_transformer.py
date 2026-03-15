"""
train_transformer.py – Training Script for Task 3: Transformer Music Generator
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Algorithm 3 from PDF:
    for epoch = 1 to E:
        for each sequence X in D:
            for t = 1 to T:
                predict p_θ(x_t | x_{<t})
            L_TR = -Σ log p_θ(x_t | x_{<t})
            θ ← θ − η∇L_TR
    Generate by: x_t ~ p_θ(x_t | x_{<t})

Also implements Task 4 (RLHF) via Policy-Gradient fine-tuning.
"""

import os, sys, argparse, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-bright') # Colorful white background

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import (TR_EPOCHS, TR_BATCH_SIZE, TR_LR,
                        RL_STEPS, RL_LR, RL_BATCH_SIZE,
                        SEQUENCE_LENGTH, NUM_GENRES, RANDOM_SEED,
                        PLOTS_DIR, MIDI_OUT_DIR, GENRES)
from src.models.transformer import MusicTransformer
from src.preprocessing.midi_parser import MIDIParser
from src.preprocessing.tokenizer import MIDITokenizer


# ─────────────────────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────────────────────

def build_token_dataset():
    """Build token-based dataset from preprocessed real data."""
    from src.config import TRAIN_TEST_DIR
    X_path = os.path.join(TRAIN_TEST_DIR, 'tr_train.npy')
    y_path = os.path.join(TRAIN_TEST_DIR, 'genres_train.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Training token data or labels not found. Run preprocessing first.")
        
    X_data = np.load(X_path) # (N, L)
    y_data = np.load(y_path) # (N,)
    
    X = torch.tensor(X_data, dtype=torch.long)
    y = torch.tensor(y_data, dtype=torch.long)
    return TensorDataset(X, y)


# ─────────────────────────────────────────────────────────────
# Task 3: Transformer Training
# ─────────────────────────────────────────────────────────────

def train_transformer(epochs: int = TR_EPOCHS, device: str = 'cpu'):
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(PLOTS_DIR,    exist_ok=True)
    os.makedirs(MIDI_OUT_DIR, exist_ok=True)

    print("[TR] Building tokenised dataset from REAL data ...")
    dataset  = build_token_dataset()
    n_train  = int(0.85 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_dl = DataLoader(train_ds, TR_BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   TR_BATCH_SIZE, shuffle=False)
    print(f"[TR] Train={n_train} | Val={n_val}")

    model     = MusicTransformer().to(device)
    print(f"[TR] Parameters: {model.count_parameters():,}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=TR_LR, weight_decay=1e-2)
    # Warmup + cosine decay
    def lr_lambda(step):
        warmup = 500
        if step < warmup: return step / warmup
        progress = (step - warmup) / max(1, epochs * len(train_dl) - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    train_losses, val_losses, perplexities = [], [], []
    best_ppl = float('inf')

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────
        model.train(); ep_loss = 0.0
        for (xb, gb) in train_dl:
            xb, gb = xb.to(device), gb.to(device)
            inp     = xb[:, :-1]   # input tokens (all but last)
            tgt     = xb[:, 1:]    # target tokens (all but first)
            optimiser.zero_grad()
            logits = model(inp, gb)
            loss   = model.compute_loss(logits, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            ep_loss += loss.item()
        avg_train = ep_loss / len(train_dl)
        train_losses.append(avg_train)

        # ── Validation ────────────────────────────────────────
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for (xb, gb) in val_dl:
                xb, gb = xb.to(device), gb.to(device)
                logits = model(xb[:, :-1], gb)
                v_loss += model.compute_loss(logits, xb[:, 1:]).item()
        avg_val = v_loss / len(val_dl)
        ppl     = MusicTransformer.compute_perplexity(avg_val)
        val_losses.append(avg_val)
        perplexities.append(ppl)

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), os.path.join(PLOTS_DIR, '..', 'tr_best.pt'))

        if epoch % 10 == 0 or epoch == 1:
            print(f"[TR] Ep {epoch:3d}/{epochs}  "
                  f"Train={avg_train:.4f}  Val={avg_val:.4f}  PPL={ppl:.2f}")

    # ── Plots ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train', color='royalblue')
    ax1.plot(val_losses,   label='Val',   color='tomato', ls='--')
    ax1.set_title('Transformer Cross-Entropy Loss')
    ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(perplexities, color='darkorange')
    ax2.set_title('Validation Perplexity'); ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Task 3 – Transformer Training', fontsize=14)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'transformer_loss.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[TR] Plots saved → {path}")
    print(f"[TR] Best Perplexity: {best_ppl:.2f}")

    # ── Generate 10 long compositions ─────────────────────────
    from src.generation.midi_export import tokens_to_midi
    tokenizer = MIDITokenizer()
    model.eval()
    for i in range(10):
        g = torch.tensor([i % NUM_GENRES], device=device)
        toks = model.generate(g, max_len=128, temperature=0.9, top_k=40, device=device)
        events = tokenizer.decode(toks[0].cpu().numpy())
        tokens_to_midi(events, path=os.path.join(MIDI_OUT_DIR,
                        f'transformer_{GENRES[i % NUM_GENRES]}_comp_{i+1}.mid'))
    print("[TR] Generated 10 long-sequence compositions.")

    return model, train_losses, val_losses, perplexities, best_ppl


# ─────────────────────────────────────────────────────────────
# Task 4: RLHF Policy-Gradient Fine-Tuning
# ─────────────────────────────────────────────────────────────

class MusicRewardModel(nn.Module):
    """
    Reward model: r = HumanScore(X_gen)
    Implemented as a trainable scoring function based on musicality heuristics
    combined with a small learned MLP that simulates human preference.

    Heuristics used:
        1. Pitch diversity: number of unique pitches / total pitches
        2. Rhythm regularity: low std of inter-onset intervals
        3. Note density: notes per bar (penalised if too dense/sparse)
    """
    def __init__(self, token_vocab: int = 391, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(token_vocab, 32)
        self.lstm  = nn.LSTM(32, hidden, batch_first=True)
        self.head  = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                   nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → reward: (B,)"""
        e = self.embed(tokens)
        _, (h, _) = self.lstm(e)
        reward = self.head(h.squeeze(0)).squeeze(-1)  # (B,) in [0,1]
        return reward


def simulate_human_reward(tokens: torch.Tensor) -> torch.Tensor:
    """
    Simulate human preference scores using musicality heuristics.
    Score ∈ [0, 1].  Combines:
        - Unique pitch ratio  (higher = better)
        - Balanced note density (penalised extremes)
    """
    rewards = []
    for seq in tokens.cpu().numpy():
        note_on  = seq[(seq >= 0) & (seq < 128)]
        n_unique = len(np.unique(note_on))
        n_total  = max(len(note_on), 1)
        diversity = n_unique / n_total                  # 0-1
        density   = min(n_total, 50) / 50              # reward up to 50 notes
        score     = 0.6 * diversity + 0.4 * density
        rewards.append(score)
    return torch.tensor(rewards, dtype=torch.float32)


def rlhf_finetune(model: MusicTransformer, steps: int = RL_STEPS,
                  device: str = 'cpu'):
    """
    RLHF Policy-Gradient fine-tuning.

    Objective:   J(θ) = E[r(X_gen)]
    Update rule: ∇_θ J(θ) = E[r · ∇_θ log p_θ(X)]

    Implements REINFORCE with baseline subtraction to reduce variance.
    """
    print("\n[RLHF] Starting Task 4: Policy-Gradient fine-tuning ...")
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=RL_LR)

    rewards_before = []
    rewards_after  = []
    policy_losses  = []

    for step in range(1, steps + 1):
        model.train()
        # Sample genre randomly
        genres = torch.randint(0, NUM_GENRES, (RL_BATCH_SIZE,), device=device)
        # Generate token sequences (autoregressive sampling)
        gen_tokens = model.generate(genres, max_len=64,
                                    temperature=1.0, top_k=50, device=device)  # (B, T)
        gen_tokens = gen_tokens[:, 1:]  # remove BOS for reward

        # Compute reward
        r = simulate_human_reward(gen_tokens).to(device)   # (B,)
        if step <= RL_STEPS // 2:
            rewards_before.append(r.mean().item())

        # Baseline (moving average)
        baseline = r.mean()

        # Policy: log p_θ(X_gen)
        model.train()
        logits = model(gen_tokens[:, :-1], genres)          # (B, T-1, V)
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = gen_tokens[:, 1:].clamp(0, model.vocab_size - 1)
        tok_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        # Mask padding tokens
        mask = (gen_tokens[:, 1:] != 388).float()
        seq_log_prob = (tok_log_probs * mask).sum(-1)       # (B,)

        # Policy gradient loss: L = -E[(r - baseline) * log p(X)]
        advantage  = (r - baseline).detach()
        policy_loss = -(advantage * seq_log_prob).mean()
        policy_losses.append(policy_loss.item())

        optimiser.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimiser.step()

        if step > RL_STEPS // 2:
            rewards_after.append(r.mean().item())

        if step % 50 == 0 or step == 1:
            print(f"[RLHF] Step {step:4d}/{steps}  "
                  f"Reward={r.mean().item():.4f}  "
                  f"PolicyLoss={policy_loss.item():.4f}")

    # Summary comparison
    mean_before = np.mean(rewards_before) if rewards_before else 0
    mean_after  = np.mean(rewards_after)  if rewards_after  else 0
    improvement = (mean_after - mean_before) / max(mean_before, 1e-8) * 100

    print(f"\n[RLHF] Reward before fine-tuning: {mean_before:.4f}")
    print(f"[RLHF] Reward after  fine-tuning: {mean_after:.4f}")
    print(f"[RLHF] Improvement:               {improvement:+.1f}%")

    # ── RLHF Plot ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(policy_losses, color='crimson')
    ax1.set_title('RLHF Policy Loss'); ax1.set_xlabel('Step'); ax1.grid(True, alpha=0.3)

    x = [f"Before\n({mean_before:.3f})", f"After\n({mean_after:.3f})"]
    ax2.bar(x, [mean_before, mean_after], color=['steelblue', 'seagreen'], width=0.4)
    ax2.set_title('Human Reward: Before vs After RLHF')
    ax2.set_ylabel('Score (0-1)')
    ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)

    fig.suptitle('Task 4 – RLHF Fine-Tuning Results', fontsize=14)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'rlhf_results.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[RLHF] Plot saved → {path}")

    # ── Generate 10 RLHF-tuned samples ────────────────────────
    from src.generation.midi_export import tokens_to_midi
    tokenizer = MIDITokenizer()
    model.eval()
    for i in range(10):
        g = torch.tensor([i % NUM_GENRES], device=device)
        toks = model.generate(g, max_len=128, temperature=0.85, top_k=35, device=device)
        events = tokenizer.decode(toks[0].cpu().numpy())
        tokens_to_midi(events, path=os.path.join(MIDI_OUT_DIR,
                        f'rlhf_tuned_{GENRES[i % NUM_GENRES]}_{i+1}.mid'))
    print("[RLHF] Generated 10 RLHF-tuned MIDI compositions.")

    return mean_before, mean_after, improvement


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--tr_epochs', type=int, default=TR_EPOCHS)
    ap.add_argument('--rl_steps',  type=int, default=RL_STEPS)
    ap.add_argument('--device',    type=str, default='cpu')
    args = ap.parse_args()

    model, tr_loss, val_loss, ppls, best_ppl = train_transformer(args.tr_epochs, args.device)
    mean_b, mean_a, impv = rlhf_finetune(model, args.rl_steps, args.device)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Transformer Best Perplexity : {best_ppl:.2f}")
    print(f"  RLHF Before Reward          : {mean_b:.4f}")
    print(f"  RLHF After Reward           : {mean_a:.4f}")
    print(f"  RLHF Improvement            : {impv:+.1f}%")
