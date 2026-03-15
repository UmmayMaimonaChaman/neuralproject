import sys
import os
sys.path.insert(0, os.getcwd())
try:
    from src.generation.generate_music import run_full_generation
    from src.evaluation.metrics import print_evaluation_table
    results = run_full_generation('cpu')
    print("\nCSV_RESULTS")
    print("Model,PHSim,RhythmDiv,RepRatio,HumanScore")
    for k, v in results.items():
        ph = v.get('pitch_histogram_similarity', 0)
        rd = v.get('rhythm_diversity', 0)
        rr = v.get('repetition_ratio', 0)
        hs = v.get('human_score', 1.0)
        print(f"{k},{ph:.4f},{rd:.4f},{rr:.4f},{hs:.2f}")
except Exception as e:
    print(f"FAILED: {e}")
