"""
pitch_histogram.py  –  Pitch Histogram Visualisation
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1
"""

import numpy as np
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.midi_parser import NoteEvent
from src.evaluation.metrics import pitch_histogram
from src.config import PLOTS_DIR

PITCH_CLASSES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def plot_pitch_histogram(samples_dict: Dict[str, List[List[NoteEvent]]],
                         save_path: str = None) -> None:
    """
    Plot overlaid pitch class histograms for multiple model outputs.

    Args:
        samples_dict: {'Model Name': [[NoteEvent, ...], ...], ...}
        save_path: where to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(12)
    colors = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'purple']
    width  = 0.15

    for idx, (model_name, sample_lists) in enumerate(samples_dict.items()):
        # Average histogram over all samples
        hists = np.array([pitch_histogram(evs) for evs in sample_lists if evs])
        avg   = hists.mean(axis=0) if len(hists) > 0 else np.zeros(12)
        offset = (idx - len(samples_dict) / 2) * width
        ax.bar(x + offset, avg, width, label=model_name,
               color=colors[idx % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(PITCH_CLASSES)
    ax.set_xlabel('Pitch Class')
    ax.set_ylabel('Relative Frequency')
    ax.set_title('Pitch Class Histogram Comparison Across Models\n'
                 'H(p,q) = Σ|p_i − q_i|  for i=1...12')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[PitchHist] Saved → {save_path}")
    else:
        fig.savefig(os.path.join(PLOTS_DIR, 'pitch_histogram.png'), dpi=150)
    plt.close(fig)


def compute_kl_divergence(p: np.ndarray, q: np.ndarray,
                           eps: float = 1e-9) -> float:
    """KL(P‖Q) = Σ p_i log(p_i/q_i)"""
    p = np.clip(p, eps, None); p /= p.sum()
    q = np.clip(q, eps, None); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


if __name__ == "__main__":
    from src.preprocessing.midi_parser import NoteEvent
    # Create simple test samples using NoteEvent
    samples = {
        "Random": [[NoteEvent(pitch=60+np.random.randint(0,12), start=j*0.5, end=(j+1)*0.5, velocity=80) for j in range(32)] for _ in range(5)],
        "Task1":  [[NoteEvent(pitch=65+np.random.randint(0,12), start=j*0.5, end=(j+1)*0.5, velocity=80) for j in range(48)] for _ in range(5)]
    }
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_pitch_histogram(samples, save_path=os.path.join(PLOTS_DIR, 'pitch_histogram_test.png'))
    print("Pitch histogram created (Test sequence).")
