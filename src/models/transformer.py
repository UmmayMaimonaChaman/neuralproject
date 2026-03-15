"""
transformer.py  –  Task 3: Transformer Music Generator
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Mathematical Formulation:
    Autoregressive probability: p(X) = Π_{t=1}^{T} p(x_t | x_{<t})
    Training loss:  L_TR = -Σ_{t=1}^{T} log p_θ(x_t | x_{<t})
    Perplexity:     PPL  = exp(L_TR / T)
    Genre embedding: h_t = Emb(x_t) + Emb(genre)  (token + genre conditioning)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import (TR_D_MODEL, TR_NHEAD, TR_NUM_LAYERS, TR_DIM_FF,
                        TR_DROPOUT, TR_MAX_SEQ_LEN, VOCAB_SIZE, NUM_GENRES)


# ─────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                    # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ─────────────────────────────────────────────────────────────
# Transformer Decoder-Only Music Generator
# ─────────────────────────────────────────────────────────────

class MusicTransformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive token generation.

    Architecture:
        Token Embedding  +  Genre Embedding  +  Positional Encoding
              ↓
        N × TransformerDecoderLayer  (causal / masked self-attention)
              ↓
        Linear Projection → Vocab Logits
    """
    def __init__(self, vocab_size: int = VOCAB_SIZE + 3,  # +3 for PAD/BOS/EOS
                 d_model: int = TR_D_MODEL,
                 nhead: int = TR_NHEAD,
                 num_layers: int = TR_NUM_LAYERS,
                 dim_feedforward: int = TR_DIM_FF,
                 dropout: float = TR_DROPOUT,
                 max_seq_len: int = TR_MAX_SEQ_LEN,
                 num_genres: int = NUM_GENRES):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=388)
        self.genre_embed = nn.Embedding(num_genres, d_model)
        self.pos_enc     = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer layers (decoder-only = encoder-free; use memory=None trick)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True   # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Output projection
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Weight tying (token embedding and output projection share weights)
        self.out_proj.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, T: int, device) -> torch.Tensor:
        """
        Upper-triangular mask ensuring position t only attends to x_{<t}.
        Shape: (T, T)  dtype=bool  (True = ignore)
        """
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, tokens: torch.Tensor, genre: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) integer token indices
            genre:  (B,)   integer genre labels
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tokens.shape
        device = tokens.device
        causal_mask = self._make_causal_mask(T, device)

        # h_t = Emb(x_t) + Emb(genre)   [as per PDF equation]
        tok_emb   = self.token_embed(tokens)                    # (B, T, D)
        genre_emb = self.genre_embed(genre).unsqueeze(1)        # (B, 1, D)
        h = self.pos_enc(tok_emb + genre_emb)                   # (B, T, D)

        out    = self.transformer(h, mask=causal_mask,
                                  src_key_padding_mask=src_key_padding_mask)
        logits = self.out_proj(out)                              # (B, T, V)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                     ignore_index: int = 388) -> torch.Tensor:
        """
        L_TR = -Σ log p_θ(x_t | x_{<t})   (cross-entropy, ignoring <PAD>)
        """
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=ignore_index
        )

    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """
        Perplexity = exp(L_TR / T)
        A lower perplexity indicates a better model.
        """
        return math.exp(loss)

    @torch.no_grad()
    def generate(self, genre: torch.Tensor, max_len: int = 128,
                 temperature: float = 1.0, top_k: int = 50,
                 device: str = 'cpu') -> torch.Tensor:
        """
        Autoregressive generation:  x_t ~ p_θ(x_t | x_{<t})
        with Top-k sampling + temperature scaling.

        Args:
            genre:       (1,) or (B,) genre labels
            max_len:     number of tokens to generate
            temperature: controls randomness (lower = more deterministic)
            top_k:       restrict sampling to top-k logits

        Returns:
            generated: (B, max_len) token sequence
        """
        self.eval()
        B = genre.size(0)
        BOS = 389
        generated = torch.full((B, 1), BOS, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(generated, genre)          # (B, t, V)
            next_logits = logits[:, -1, :] / temperature    # (B, V)
            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                threshold     = topk_vals[:, -1].unsqueeze(-1)
                next_logits   = next_logits.masked_fill(next_logits < threshold, -1e9)
            probs = F.softmax(next_logits, dim=-1)           # (B, V)
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok == 390).all():  # EOS
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MusicTransformer()
    print(f"MusicTransformer — Trainable parameters: {model.count_parameters():,}")
    B, T = 4, TR_MAX_SEQ_LEN
    V    = model.vocab_size
    tokens = torch.randint(0, V - 3, (B, T))
    genre  = torch.randint(0, NUM_GENRES, (B,))
    logits = model(tokens, genre)
    print(f"Logits shape: {logits.shape}")
    targets = torch.randint(0, V, (B, T))
    loss = model.compute_loss(logits, targets)
    ppl  = model.compute_perplexity(loss.item())
    print(f"Loss={loss.item():.4f}  Perplexity={ppl:.2f}")
