"""
preprocess_data.py - Batch MIDI Preprocessing
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

This script processes raw MIDI files from MAESTRO and Lakh into piano-rolls
and token sequences, then saves them to disk for faster training.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import *
from src.preprocessing.midi_parser import MIDIParser, assign_genre_label
from src.preprocessing.piano_roll import note_events_to_piano_roll, segment_piano_roll
from src.preprocessing.tokenizer import MIDITokenizer

def batch_process():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TRAIN_TEST_DIR, exist_ok=True)
    
    parser = MIDIParser()
    tokenizer = MIDITokenizer()
    
    all_rolls = []   # (N, T, P) for Task 1/2
    all_tokens = []  # (N, L) for Task 3
    all_genres = []
    
    raw_dirs = {
        'maestro': RAW_MIDI_DIR + '/maestro',
        'lakh': RAW_MIDI_DIR + '/lakh'
    }
    
    print("[Preprocessing] Starting batch conversion...")
    
    for origin, path in raw_dirs.items():
        if not os.path.exists(path):
            print(f"  Warning: {path} not found. Skipping {origin}.")
            continue
            
        files = [f for f in os.listdir(path) if f.lower().endswith(('.mid', '.midi'))]
        print(f"  Processing {len(files[:200])} files from {origin}...") # Limit to first 200 for speed
        
        for fname in tqdm(files[:200]):
            genre = assign_genre_label(fname)
            genre_idx = GENRES.index(genre) if genre in GENRES else 0
            
            events = parser.parse_file(os.path.join(path, fname))
            if not events: continue
            
            # 1. Piano-roll processing
            roll = note_events_to_piano_roll(events, fs=4.0)
            segments = segment_piano_roll(roll, SEQUENCE_LENGTH)
            
            for seg in segments:
                if seg.shape == (NUM_PITCHES, SEQUENCE_LENGTH):
                    all_rolls.append(seg.T) # (T, P)
                    all_genres.append(genre_idx)
            
            # 2. Token processing (for Transformer)
            toks = tokenizer.encode(events, max_len=SEQUENCE_LENGTH)
            if len(toks) >= SEQUENCE_LENGTH:
                all_tokens.append(toks[:SEQUENCE_LENGTH])
                
    if not all_rolls:
        print("Error: No data processed. Check MIDI paths.")
        return

    # Convert to numpy arrays
    rolls_np = np.array(all_rolls, dtype=np.float32)
    tokens_np = np.array(all_tokens, dtype=np.int32)
    genres_np = np.array(all_genres[:len(rolls_np)], dtype=np.int32)
    
    print(f"[Preprocessing] Done! Total samples: {len(rolls_np)}")
    
    # Save processed files
    np.save(os.path.join(PROCESSED_DIR, 'piano_rolls.npy'), rolls_np)
    np.save(os.path.join(PROCESSED_DIR, 'tokens.npy'), tokens_np)
    np.save(os.path.join(PROCESSED_DIR, 'genres.npy'), genres_np)
    
    # Train/Test Split
    print("[Preprocessing] Saving train/test splits...")
    X_train, X_test, g_train, g_test = train_test_split(rolls_np, genres_np, test_size=0.15, random_state=42)
    
    np.save(os.path.join(TRAIN_TEST_DIR, 'ae_train.npy'), X_train)
    np.save(os.path.join(TRAIN_TEST_DIR, 'ae_test.npy'),  X_test)
    np.save(os.path.join(TRAIN_TEST_DIR, 'genres_train.npy'), g_train)
    np.save(os.path.join(TRAIN_TEST_DIR, 'genres_test.npy'),  g_test)
    
    if len(tokens_np) > 0:
        T_train, T_test = train_test_split(tokens_np, test_size=0.15, random_state=42)
        np.save(os.path.join(TRAIN_TEST_DIR, 'tr_train.npy'), T_train)
        np.save(os.path.join(TRAIN_TEST_DIR, 'tr_test.npy'),  T_test)

    print(f"Split results: {len(X_train)} train, {len(X_test)} test.")

if __name__ == "__main__":
    batch_process()
