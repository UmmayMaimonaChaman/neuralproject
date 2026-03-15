"""
config.py - Global Configuration for Music Generation Project
CSE425 - Neural Networks
Student: Ummay Maimona Chaman | ID: 22301719 | Section: 1
"""

import os

# =============================================================
# Paths
# =============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_MIDI_DIR = os.path.join(DATA_DIR, "raw_midi")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_TEST_DIR = os.path.join(DATA_DIR, "train_test_split")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MIDI_OUT_DIR = os.path.join(OUTPUTS_DIR, "generated_midis")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
SURVEY_DIR = os.path.join(OUTPUTS_DIR, "survey_results")

# =============================================================
# MIDI / Tokenisation
# =============================================================
SEQUENCE_LENGTH = 128      # number of time steps per segment
VOCAB_SIZE = 388           # MIDI-Like token vocabulary (note-on, note-off, velocity, time-shift)
NUM_PITCH_CLASSES = 128    # MIDI pitches 0-127
STEPS_PER_BAR = 16         # 16 steps per bar (16th-note resolution)
MAX_BARS = 8               # maximum bars per sample

# Piano-roll dimensions
PIANO_ROLL_TIME_STEPS = 128
PIANO_ROLL_PITCH_LOW = 21   # lowest piano key (A0)
PIANO_ROLL_PITCH_HIGH = 108 # highest piano key (C8)
NUM_PITCHES = PIANO_ROLL_PITCH_HIGH - PIANO_ROLL_PITCH_LOW  # 87

# =============================================================
# Genres
# =============================================================
GENRES = ["classical", "jazz", "rock", "pop", "electronic"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
NUM_GENRES = len(GENRES)

# =============================================================
# Task 1 – LSTM Autoencoder
# =============================================================
AE_LATENT_DIM = 128
AE_HIDDEN_DIM = 256
AE_NUM_LAYERS = 2
AE_DROPOUT = 0.3
AE_EPOCHS = 50
AE_BATCH_SIZE = 32
AE_LR = 1e-3

# =============================================================
# Task 2 – Variational Autoencoder (VAE)
# =============================================================
VAE_LATENT_DIM = 128
VAE_HIDDEN_DIM = 256
VAE_NUM_LAYERS = 2
VAE_DROPOUT = 0.3
VAE_BETA = 0.5          # KL weight (beta-VAE)
VAE_EPOCHS = 60
VAE_BATCH_SIZE = 32
VAE_LR = 1e-3

# =============================================================
# Task 3 – Transformer
# =============================================================
TR_D_MODEL = 256
TR_NHEAD = 8
TR_NUM_LAYERS = 6
TR_DIM_FF = 512
TR_DROPOUT = 0.1
TR_MAX_SEQ_LEN = 128
TR_EPOCHS = 50
TR_BATCH_SIZE = 32
TR_LR = 1e-4

# =============================================================
# Task 4 – RLHF
# =============================================================
RL_STEPS = 200
RL_LR = 1e-5
RL_BATCH_SIZE = 8

# =============================================================
# Evaluation
# =============================================================
NUM_GENERATED_SAMPLES_TASK1 = 5
NUM_GENERATED_SAMPLES_TASK2 = 8
NUM_GENERATED_SAMPLES_TASK3 = 10
NUM_GENERATED_SAMPLES_TASK4 = 10

RANDOM_SEED = 42
