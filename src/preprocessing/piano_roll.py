"""
piano_roll.py - Piano-Roll Representation
CSE425 - Neural Networks
Student: Ummay Maimona Chaman | ID: 22301719 | Section: 1

Converts parsed MIDI NoteEvents into a 2-D piano-roll matrix:
  rows  = pitch bins  (PIANO_ROLL_PITCH_LOW … PIANO_ROLL_PITCH_HIGH)
  cols  = time steps  (quantised at fs steps/second)
"""

import numpy as np
from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import (PIANO_ROLL_PITCH_LOW, PIANO_ROLL_PITCH_HIGH,
                        PIANO_ROLL_TIME_STEPS, STEPS_PER_BAR, MAX_BARS,
                        NUM_PITCHES, SEQUENCE_LENGTH)
from src.preprocessing.midi_parser import NoteEvent


# ─────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────

def note_events_to_piano_roll(
        events: List[NoteEvent],
        fs: float = 4.0,
        duration_seconds: Optional[float] = None
) -> np.ndarray:
    """
    Convert a list of NoteEvents into a binary piano-roll matrix.

    Args:
        events: sorted list of NoteEvent objects.
        fs: quantisation rate (steps per second).
        duration_seconds: total length; inferred from events if None.

    Returns:
        roll: np.ndarray  shape = (NUM_PITCHES, T)  dtype=float32
              roll[pitch_idx, t] = 1 if note is active, else 0
    """
    if not events:
        return np.zeros((NUM_PITCHES, PIANO_ROLL_TIME_STEPS), dtype=np.float32)

    total_dur = duration_seconds or (max(e.end for e in events))
    T = int(np.ceil(total_dur * fs))
    T = max(T, PIANO_ROLL_TIME_STEPS)

    roll = np.zeros((NUM_PITCHES, T), dtype=np.float32)

    for ev in events:
        pitch_idx = ev.pitch - PIANO_ROLL_PITCH_LOW
        if not (0 <= pitch_idx < NUM_PITCHES):
            continue
        t_start = int(ev.start * fs)
        t_end   = int(ev.end   * fs)
        t_end   = max(t_end, t_start + 1)          # at least 1 step
        roll[pitch_idx, t_start:t_end] = 1.0

    # Normalise velocity (optional: already binary here)
    return roll[:, :PIANO_ROLL_TIME_STEPS]


def segment_piano_roll(
        roll: np.ndarray,
        segment_len: int = SEQUENCE_LENGTH
) -> List[np.ndarray]:
    """
    Slice the piano-roll into overlapping fixed-length windows.

    Args:
        roll: shape (NUM_PITCHES, T)
        segment_len: number of time steps per window

    Returns:
        List of arrays each with shape (NUM_PITCHES, segment_len)
    """
    _, T = roll.shape
    segments = []
    step = segment_len // 2   # 50 % overlap
    for start in range(0, T - segment_len + 1, step):
        seg = roll[:, start:start + segment_len]
        # Only keep segments with some content
        if seg.sum() > 4:
            segments.append(seg.astype(np.float32))
    return segments


def piano_roll_to_flat_token(roll: np.ndarray) -> np.ndarray:
    """
    Flatten piano-roll columns into a 1-D token sequence.

    Each time-step becomes a pitch bin vector → vectorised as integer index
    of the active pitch (or 0 for silence). Useful for Transformer input.

    Args:
        roll: shape (NUM_PITCHES, T)

    Returns:
        tokens: shape (T,) int32, values in [0, NUM_PITCHES]
    """
    tokens = np.zeros(roll.shape[1], dtype=np.int32)
    for t in range(roll.shape[1]):
        active = np.where(roll[:, t] > 0)[0]
        if len(active) > 0:
            tokens[t] = int(active[0]) + 1   # 0 = silence, 1-NUM_PITCHES = pitch
    return tokens


def normalize_piano_roll(roll: np.ndarray) -> np.ndarray:
    """Min-max normalise piano-roll to [0, 1] (already binary, just cast)."""
    return roll.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Typing stub
# ─────────────────────────────────────────────────────────────
from typing import Optional   # noqa: E402 (needed above)

if __name__ == "__main__":
    # Quick sanity check with synthetic data
    from src.preprocessing.midi_parser import MIDIParser
    parser = MIDIParser()
    events = parser._generate_synthetic_notes(64)
    roll = note_events_to_piano_roll(events, fs=4.0)
    print(f"Piano-roll shape: {roll.shape}")
    segs = segment_piano_roll(roll)
    print(f"Number of segments: {len(segs)}")
    tokens = piano_roll_to_flat_token(roll)
    print(f"Token sequence shape: {tokens.shape}, unique tokens: {np.unique(tokens)}")
