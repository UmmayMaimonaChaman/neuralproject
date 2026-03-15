"""
midi_export.py  –  MIDI File Export Utilities
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

Exports generated piano-rolls and token sequences to MIDI files.
"""

import os, sys
import numpy as np
from typing import List, Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.midi_parser import NoteEvent
from src.config import PIANO_ROLL_PITCH_LOW, NUM_PITCHES, MIDI_OUT_DIR

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False

try:
    import mido
    from mido import MidiFile, MidiTrack, Message
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False


def note_events_to_midi(events: List[NoteEvent], path: str,
                         tempo_bpm: float = 120.0,
                         instrument_program: int = 0) -> None:
    """
    Export a list of NoteEvents to a MIDI file.

    Args:
        events:  list of NoteEvent objects
        path:    output .mid file path
        tempo_bpm: beats per minute
        instrument_program: GM instrument (0=Grand Piano)
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if HAS_PRETTY_MIDI:
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
        inst = pretty_midi.Instrument(program=instrument_program, name="Piano")
        for ev in events:
            note = pretty_midi.Note(
                velocity=max(1, min(127, ev.velocity)),
                pitch=max(0, min(127, ev.pitch)),
                start=ev.start, end=max(ev.start + 0.05, ev.end)
            )
            inst.notes.append(note)
        pm.instruments.append(inst)
        pm.write(path)

    elif HAS_MIDO:
        mid  = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        ticks_per_beat = mid.ticks_per_beat
        us_per_beat    = int(60_000_000 / tempo_bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
        track.append(Message('program_change', program=instrument_program, time=0))

        events_sorted = sorted(events, key=lambda e: e.start)
        abs_msgs = []
        for ev in events_sorted:
            t_on  = int(ev.start * ticks_per_beat * tempo_bpm / 60)
            t_off = int(ev.end   * ticks_per_beat * tempo_bpm / 60)
            abs_msgs.append((t_on,  'note_on',  ev.pitch, max(1, ev.velocity)))
            abs_msgs.append((t_off, 'note_off', ev.pitch, 0))
        abs_msgs.sort(key=lambda m: m[0])

        prev_tick = 0
        for (tick, msg_type, pitch, vel) in abs_msgs:
            delta = tick - prev_tick
            track.append(Message(msg_type, note=pitch, velocity=vel, time=delta))
            prev_tick = tick
        mid.save(path)
    else:
        # Minimal fallback: write a dummy MIDI-like text file
        with open(path.replace('.mid', '_events.txt'), 'w') as f:
            f.write(f"# MIDI export fallback (install pretty_midi or mido)\n")
            f.write(f"# Tempo: {tempo_bpm} BPM\n")
            for ev in events:
                f.write(f"NOTE pitch={ev.pitch} start={ev.start:.3f} end={ev.end:.3f} vel={ev.velocity}\n")


def piano_roll_to_midi(roll: np.ndarray, fs: float = 4.0,
                        path: str = None, tempo_bpm: float = 120.0) -> None:
    """
    Convert a piano-roll matrix to MIDI.

    Args:
        roll:  np.ndarray shape (NUM_PITCHES, T) – binary piano roll
        fs:    steps per second
        path:  output path
        tempo_bpm: tempo
    """
    path = path or os.path.join(MIDI_OUT_DIR, 'piano_roll_output.mid')
    events: List[NoteEvent] = []
    _, T = roll.shape

    for pitch_idx in range(roll.shape[0]):
        pitch_abs = pitch_idx + PIANO_ROLL_PITCH_LOW
        note_on   = None
        for t in range(T):
            active = roll[pitch_idx, t] > 0.5
            if active and note_on is None:
                note_on = t
            elif not active and note_on is not None:
                events.append(NoteEvent(
                    pitch=pitch_abs,
                    start=note_on / fs,
                    end=t / fs,
                    velocity=80
                ))
                note_on = None
        if note_on is not None:
            events.append(NoteEvent(
                pitch=pitch_abs,
                start=note_on / fs,
                end=T / fs,
                velocity=80
            ))

    note_events_to_midi(events, path, tempo_bpm=tempo_bpm)


def tokens_to_midi(events: List[NoteEvent], path: str,
                    tempo_bpm: float = 120.0) -> None:
    """Shorthand – directly export NoteEvents from tokenizer.decode()."""
    note_events_to_midi(events, path, tempo_bpm=tempo_bpm)


if __name__ == "__main__":
    from src.preprocessing.midi_parser import NoteEvent
    # Create a small realistic pitch sequence for testing
    events = [NoteEvent(pitch=60+i, start=i*0.5, end=(i+1)*0.5, velocity=80) for i in range(8)]
    os.makedirs(MIDI_OUT_DIR, exist_ok=True)
    path = os.path.join(MIDI_OUT_DIR, 'test_export.mid')
    note_events_to_midi(events, path)
    print(f"Exported test MIDI (Real data pattern) to {path}")
