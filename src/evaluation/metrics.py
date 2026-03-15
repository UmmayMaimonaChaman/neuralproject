"""
metrics.py  –  Evaluation Metrics for Music Generation
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Implements all metrics from the PDF:
    1. Pitch Histogram Similarity: H(p,q) = Σ|p_i - q_i|  for i=1..12
    2. Rhythm Diversity Score:     D_rhythm = #unique_durations / #total_notes
    3. Repetition Ratio:           R = #repeated_patterns / #total_patterns
    4. Human Listening Score:      Simulated survey (1-5 scale)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import TRAIN_TEST_DIR
from src.preprocessing.midi_parser import NoteEvent


# ─────────────────────────────────────────────────────────────
# 1. Pitch Histogram Similarity
# ─────────────────────────────────────────────────────────────

def pitch_histogram(events: List[NoteEvent]) -> np.ndarray:
    """
    Compute 12-bin pitch class histogram.
    p_i = frequency of pitch class i / total notes
    """
    hist = np.zeros(12, dtype=np.float64)
    for ev in events:
        hist[ev.pitch % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def pitch_histogram_similarity(events_gen: List[NoteEvent],
                                events_ref: List[NoteEvent]) -> float:
    """
    H(p,q) = Σ_{i=1}^{12} |p_i - q_i|

    Lower is better (0 = identical distributions).
    Normalised by dividing by 2 so ∈ [0, 1].

    Args:
        events_gen: generated note events
        events_ref: reference (real) note events
    Returns:
        similarity score ∈ [0, 1]  (1 - normalised L1)
    """
    p = pitch_histogram(events_gen)
    q = pitch_histogram(events_ref)
    l1 = float(np.sum(np.abs(p - q)))
    return 1.0 - l1 / 2.0          # 1 = identical, 0 = completely different


# ─────────────────────────────────────────────────────────────
# 2. Rhythm Diversity Score
# ─────────────────────────────────────────────────────────────

def rhythm_diversity_score(events: List[NoteEvent], resolution: float = 0.05) -> float:
    """
    D_rhythm = #unique_quantised_durations / #total_notes

    Args:
        events: list of NoteEvents
        resolution: quantisation in seconds (0.05 = 50ms)
    Returns:
        D_rhythm ∈ [0, 1]
    """
    if not events:
        return 0.0
    durations  = [round(ev.duration / resolution) * resolution for ev in events]
    n_unique   = len(set(durations))
    n_total    = len(durations)
    return n_unique / n_total


# ─────────────────────────────────────────────────────────────
# 3. Repetition Ratio
# ─────────────────────────────────────────────────────────────

def repetition_ratio(events: List[NoteEvent], window: int = 4) -> float:
    """
    R = #repeated_n-gram_patterns / #total_patterns

    Uses pitch-n-gram (window=4 notes) repetition.
    Lower R = more diverse (better).

    Args:
        events: note events
        window: n-gram size
    Returns:
        R ∈ [0, 1]
    """
    if len(events) < window:
        return 0.0
    pitches  = [ev.pitch for ev in events]
    ngrams   = [tuple(pitches[i:i+window]) for i in range(len(pitches) - window + 1)]
    total    = len(ngrams)
    unique   = len(set(ngrams))
    repeated = total - unique
    return repeated / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# 4. Human Listening Score (Simulated)
# ─────────────────────────────────────────────────────────────

def simulate_human_score(events: List[NoteEvent]) -> float:
    """
    Simulates a human listening survey score ∈ [1, 5].

    Combines:
        - Pitch diversity (weight 0.3)
        - Rhythm diversity (weight 0.3)
        - Note count reasonableness (weight 0.2)
        - Low repetition reward (weight 0.2)

    Scaled to [1, 5] to match the PDF rubric.
    """
    if not events:
        return 1.0

    p_hist     = pitch_histogram(events)
    p_div      = float(np.sum(p_hist > 0)) / 12.0          # fraction of pitch classes used

    r_div      = rhythm_diversity_score(events)
    rep        = repetition_ratio(events)
    n_reasonable = min(len(events), 64) / 64.0

    composite = (0.30 * p_div + 0.30 * r_div +
                 0.20 * n_reasonable + 0.20 * (1 - rep))
    score = 1.0 + 4.0 * composite                          # map [0,1] → [1,5]
    return round(score, 2)


# ─────────────────────────────────────────────────────────────
# 5. Full Evaluation Suite
# ─────────────────────────────────────────────────────────────

def evaluate_model(generated_samples: List[List[NoteEvent]],
                   reference_samples: Optional[List[List[NoteEvent]]] = None
                   ) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a set of generated samples.

    Args:
        generated_samples: list of generated event sequences
        reference_samples: list of reference event sequences (optional)

    Returns:
        dict with mean metrics
    """
    if not generated_samples:
        return {}

    ph_sims, r_divs, rep_ratios, h_scores = [], [], [], []
    ref_pool = reference_samples or generated_samples  # self-evaluation if no ref

    for i, gen in enumerate(generated_samples):
        ref = ref_pool[i % len(ref_pool)]
        ph_sims.append(pitch_histogram_similarity(gen, ref))
        r_divs.append(rhythm_diversity_score(gen))
        rep_ratios.append(repetition_ratio(gen))
        h_scores.append(simulate_human_score(gen))

    return {
        "pitch_histogram_similarity": float(np.mean(ph_sims)),
        "rhythm_diversity":           float(np.mean(r_divs)),
        "repetition_ratio":           float(np.mean(rep_ratios)),
        "human_score":                float(np.mean(h_scores)),
        "n_samples":                  len(generated_samples),
    }


def print_evaluation_table(results: Dict[str, Dict]) -> None:
    """ Pretty-print comparison table matching project rubric (Screenshot 3). """
    print("\n" + "=" * 110)
    print(f"{'Model':<25} {'Loss':>8} {'Perp.':>10} {'Rhythm Div':>12} {'Hum. Score':>12} {'Genre Control':>15}")
    print("-" * 110)
    for model_name, metrics in results.items():
        loss = metrics.get('loss', '---')
        perp = metrics.get('perplexity', '---')
        r_div = metrics.get('rhythm_diversity', 0)
        h_score = metrics.get('human_score', 0)
        control = metrics.get('genre_control', '---')
        
        # Formatting
        loss_str = f"{loss:.4f}" if isinstance(loss, (float, int)) else str(loss)
        perp_str = f"{perp:.2f}" if isinstance(perp, (float, int)) else str(perp)
        
        print(f"{model_name:<25} "
              f"{loss_str:>8} "
              f"{perp_str:>10} "
              f"{r_div:>12.4f} "
              f"{h_score:>12.2f} "
              f"{control:>15}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    from src.preprocessing.midi_parser import NoteEvent
    # Create simple test samples using NoteEvent
    samples = [
        [NoteEvent(pitch=60+j, start=j*0.5, end=(j+1)*0.5, velocity=80) for j in range(8+i*4)]
        for i in range(5)
    ]
    metrics = evaluate_model(samples)
    print("Evaluation results (Test sequence):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
