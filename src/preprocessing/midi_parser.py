"""
midi_parser.py - MIDI File Parsing Utilities
CSE425 - Neural Networks
Student: Ummay Maimona Chaman | ID: 22301719 | Section: 1

Parses raw MIDI files into structured note sequences suitable for
piano-roll and token-based representations.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

from src.config import (PIANO_ROLL_PITCH_LOW, PIANO_ROLL_PITCH_HIGH,
                        STEPS_PER_BAR, MAX_BARS, GENRES, SEQUENCE_LENGTH)


class NoteEvent:
    """Represents a single note event."""
    def __init__(self, pitch: int, start: float, end: float, velocity: int):
        self.pitch = pitch
        self.start = start          # seconds
        self.end = end
        self.velocity = velocity    # 0-127
        self.duration = end - start

    def __repr__(self):
        return (f"NoteEvent(pitch={self.pitch}, start={self.start:.3f}, "
                f"end={self.end:.3f}, vel={self.velocity})")


class MIDIParser:
    """
    Parses a MIDI file into a list of NoteEvent objects.

    Algorithm (from paper notation):
        Given MIDI file M, extract sequence X = {x1, x2, ..., xT}
        where each xt = (pitch, onset, duration, velocity)
    """

    def __init__(self, fs: int = 16, max_bars: int = MAX_BARS):
        """
        Args:
            fs: quantisation frequency (steps per second).
            max_bars: maximum number of bars to keep.
        """
        self.fs = fs
        self.max_bars = max_bars

    def parse_file(self, midi_path: str) -> List[NoteEvent]:
        """Parse a MIDI file and return list of NoteEvents."""
        if not HAS_PRETTY_MIDI:
            print("[MIDIParser] Error: pretty_midi not installed.")
            return []
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            events: List[NoteEvent] = []
            for instrument in pm.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    if (PIANO_ROLL_PITCH_LOW <= note.pitch <= PIANO_ROLL_PITCH_HIGH):
                        events.append(NoteEvent(
                            pitch=note.pitch,
                            start=note.start,
                            end=note.end,
                            velocity=note.velocity
                        ))
            events.sort(key=lambda e: e.start)
            return events
        except Exception as e:
            print(f"[MIDIParser] Error parsing {midi_path}: {e}")
            return []

    def batch_parse(self, midi_dir: str) -> Dict[str, List[NoteEvent]]:
        """Parse all MIDI files in a directory."""
        results: Dict[str, List[NoteEvent]] = {}
        if not os.path.exists(midi_dir):
            print(f"[MIDIParser] Directory not found: {midi_dir}")
            return results

        for fname in os.listdir(midi_dir):
            if fname.lower().endswith(('.mid', '.midi')):
                path = os.path.join(midi_dir, fname)
                parsed = self.parse_file(path)
                if parsed:
                    results[fname] = parsed
        return results


def assign_genre_label(filename: str) -> str:
    """
    Heuristically assign genre label from filename or parent directory.
    Falls back to 'classical' if no match found.
    """
    fname_lower = filename.lower()
    for genre in GENRES:
        if genre in fname_lower:
            return genre
    return "classical"


if __name__ == "__main__":
    parser = MIDIParser()
    print("MIDIParser class ready (Strictly real data).")
