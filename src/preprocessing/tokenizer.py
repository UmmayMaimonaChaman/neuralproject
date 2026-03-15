"""
tokenizer.py - MIDI-Like Event Tokenizer
CSE425 - Neural Networks
Student: Ummay Maimona Chaman | ID: 22301719 | Section: 1

Implements a MIDI-like event tokenization scheme with:
  Tokens 0-127   : note-ON  events  (pitch 0-127)
  Tokens 128-255 : note-OFF events  (pitch 0-127)
  Tokens 256-387 : time-shift events (1-132 ticks of 10 ms each)
  Token 388      : <PAD>
  Token 389      : <BOS>  (beginning of sequence)
  Token 390      : <EOS>  (end of sequence)
"""

import numpy as np
from typing import List, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import VOCAB_SIZE, SEQUENCE_LENGTH
from src.preprocessing.midi_parser import NoteEvent

# ─────────────────────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────────────────────
NOTE_ON_OFFSET   = 0
NOTE_OFF_OFFSET  = 128
TIME_SHIFT_OFFSET = 256
NUM_TIME_SHIFTS  = 132      # each step = 10 ms  → up to 1.32 s of silence
PAD_TOKEN        = 388
BOS_TOKEN        = 389
EOS_TOKEN        = 390
EXTENDED_VOCAB   = 391      # true vocab size including special tokens


class MIDITokenizer:
    """
    Encodes NoteEvent lists into integer token sequences and
    decodes token sequences back to NoteEvent lists.
    """

    def __init__(self, resolution_ms: int = 10):
        """
        Args:
            resolution_ms: milliseconds per time-shift quantum.
        """
        self.resolution_ms = resolution_ms
        self.vocab_size = EXTENDED_VOCAB

    # ── Encoding ─────────────────────────────────────────────
    def encode(self, events: List[NoteEvent], max_len: int = SEQUENCE_LENGTH) -> np.ndarray:
        """
        Convert NoteEvents → token sequence.

        Returns:
            tokens: np.ndarray shape (max_len,) int32, padded with PAD_TOKEN.
        """
        tokens: List[int] = [BOS_TOKEN]
        current_time_ms = 0.0

        for ev in events:
            # Time-shift to reach note onset
            onset_ms  = ev.start * 1000.0
            offset_ms = ev.end   * 1000.0
            dt        = onset_ms - current_time_ms
            tokens += self._encode_time_shift(dt)

            # Note-ON
            pitch = max(0, min(127, ev.pitch))
            tokens.append(NOTE_ON_OFFSET + pitch)
            current_time_ms = onset_ms

            # Note duration (NOTE-OFF)
            dur_ms = offset_ms - onset_ms
            tokens += self._encode_time_shift(dur_ms)
            tokens.append(NOTE_OFF_OFFSET + pitch)
            current_time_ms = offset_ms

        tokens.append(EOS_TOKEN)

        # Truncate / pad
        if len(tokens) >= max_len:
            tokens = tokens[:max_len - 1] + [EOS_TOKEN]
        else:
            tokens += [PAD_TOKEN] * (max_len - len(tokens))

        return np.array(tokens, dtype=np.int32)

    def _encode_time_shift(self, dt_ms: float) -> List[int]:
        """Encode a time-delta in ms as a sequence of TIME_SHIFT tokens."""
        shifts = []
        remaining = max(0.0, dt_ms)
        while remaining >= self.resolution_ms:
            steps = min(int(remaining // self.resolution_ms), NUM_TIME_SHIFTS)
            shifts.append(TIME_SHIFT_OFFSET + steps - 1)
            remaining -= steps * self.resolution_ms
        return shifts

    # ── Decoding ─────────────────────────────────────────────
    def decode(self, tokens: np.ndarray) -> List[NoteEvent]:
        """Convert token array → NoteEvent list."""
        events: List[NoteEvent] = []
        current_time_s = 0.0
        active_notes: dict = {}   # pitch → onset time

        for tok in tokens.tolist():
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            elif NOTE_ON_OFFSET <= tok < NOTE_ON_OFFSET + 128:
                pitch = tok - NOTE_ON_OFFSET
                active_notes[pitch] = current_time_s
            elif NOTE_OFF_OFFSET <= tok < NOTE_OFF_OFFSET + 128:
                pitch = tok - NOTE_OFF_OFFSET
                if pitch in active_notes:
                    onset = active_notes.pop(pitch)
                    events.append(NoteEvent(
                        pitch=pitch,
                        start=onset,
                        end=current_time_s,
                        velocity=80
                    ))
            elif TIME_SHIFT_OFFSET <= tok < TIME_SHIFT_OFFSET + NUM_TIME_SHIFTS:
                steps = tok - TIME_SHIFT_OFFSET + 1
                current_time_s += steps * self.resolution_ms / 1000.0

        return sorted(events, key=lambda e: e.start)

    def batch_encode(self, event_lists: List[List[NoteEvent]]) -> np.ndarray:
        """Encode multiple event sequences → 2-D token array."""
        return np.stack([self.encode(evs) for evs in event_lists], axis=0)


if __name__ == "__main__":
    from src.preprocessing.midi_parser import MIDIParser
    parser    = MIDIParser()
    events    = parser._generate_synthetic_notes(32)
    tokenizer = MIDITokenizer()
    tokens    = tokenizer.encode(events)
    print(f"Token array shape : {tokens.shape}")
    print(f"Unique tokens      : {np.unique(tokens)}")
    decoded   = tokenizer.decode(tokens)
    print(f"Decoded {len(decoded)} note events from {len(events)} original events")
