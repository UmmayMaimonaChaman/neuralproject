import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.generation.generate_music import run_full_generation

def main():
    print("============================================================")
    print("  MUSIC GENERATION – RESULTS EXPORT")
    print("============================================================")
    
    # Run core generation
    results = run_full_generation('cpu')
    
    print('\nCSV_RESULTS-------------------------------------------------')
    print('Model,PHSim,RhythmDiv,RepRatio,HumanScore')
    for model, metrics in results.items():
        ph = metrics.get('pitch_histogram_similarity', 0)
        rd = metrics.get('rhythm_diversity', 0)
        rr = metrics.get('repetition_ratio', 0)
        hs = metrics.get('human_score', 0)
        print(f"{model},{ph:.4f},{rd:.4f},{rr:.4f},{hs:.2f}")
    print("============================================================")

if __name__ == "__main__":
    main()
