"""
rhythm_score.py  –  Rhythm Analysis and Diversity Scoring
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
from src.config import PLOTS_DIR


def inter_onset_intervals(events: List[NoteEvent]) -> np.ndarray:
    """Compute inter-onset intervals (IOIs) in seconds."""
    if len(events) < 2:
        return np.array([])
    onsets = sorted(ev.start for ev in events)
    return np.diff(onsets)


def tempo_estimate_bpm(events: List[NoteEvent]) -> float:
    """
    Estimate BPM from mean IOI.
    BPM ≈ 60 / mean_IOI   (assumes quarter-note onsets)
    """
    iois = inter_onset_intervals(events)
    if len(iois) == 0:
        return 120.0
    mean_ioi = np.mean(iois)
    return 60.0 / max(mean_ioi, 1e-3)


def syncopation_score(events: List[NoteEvent], beat_dur: float = 0.5) -> float:
    """
    Measure syncopation as the fraction of notes landing off the beat.
    """
    if not events:
        return 0.0
    off_beat = sum(1 for ev in events if (ev.start % beat_dur) > (beat_dur * 0.15))
    return off_beat / len(events)


def plot_rhythm_diversity(samples_dict: Dict[str, List[List[NoteEvent]]],
                          save_path: str = None) -> None:
    """
    Bar chart comparing rhythm diversity scores across models.
    D_rhythm = #unique_durations / #total_notes
    """
    from src.evaluation.metrics import rhythm_diversity_score

    model_names, diversities, tempos = [], [], []
    for model_name, sample_lists in samples_dict.items():
        rd = np.mean([rhythm_diversity_score(ev) for ev in sample_lists if ev])
        tp = np.mean([tempo_estimate_bpm(ev)     for ev in sample_lists if ev])
        model_names.append(model_name)
        diversities.append(rd)
        tempos.append(tp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Diversity bar
    colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B3']
    axes[0].bar(model_names, diversities, color=colors[:len(model_names)], alpha=0.85)
    axes[0].set_title('Rhythm Diversity Score\nD_rhythm = #unique_durations / #total_notes')
    axes[0].set_ylabel('Diversity Score')
    axes[0].set_ylim(0, 1); axes[0].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(diversities):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)

    # Tempo bar
    axes[1].bar(model_names, tempos, color=colors[:len(model_names)], alpha=0.85)
    axes[1].set_title('Estimated Tempo (BPM)\nBPM ≈ 60 / mean(IOI)')
    axes[1].set_ylabel('BPM'); axes[1].grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(tempos):
        axes[1].text(i, v + 1, f"{v:.0f}", ha='center', fontsize=10)

    fig.tight_layout()
    out = save_path or os.path.join(PLOTS_DIR, 'rhythm_diversity.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[RhythmScore] Saved → {out}")


if __name__ == "__main__":
    from src.preprocessing.midi_parser import NoteEvent
    # Create simple test samples using NoteEvent
    samples = {
        "Random": [[NoteEvent(pitch=60, start=j*0.5, end=(j+1)*0.5, velocity=80) for j in range(32)] for _ in range(5)],
        "VAE":    [[NoteEvent(pitch=60, start=j*0.4, end=(j+1)*0.4, velocity=80) for j in range(48)] for _ in range(5)]
    }
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_rhythm_diversity(samples, save_path=os.path.join(PLOTS_DIR, 'rhythm_test.png'))
    print("Rhythm diversity plot created (Test sequence).")
