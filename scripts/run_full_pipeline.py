"""Run full IDS Healthcare CIP pipeline."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: str):
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=PROJECT_ROOT)


def main():
    steps = [
        "python src/phase1_preprocessing/run_phase1.py",
        "python src/phase2_feature_selection/run_phase2.py",
        "python src/phase3_autoencoder/run_phase3.py",
        "python src/phase4_clustering/run_phase4.py",
        "python src/phase5_classification/run_phase5.py",
    ]
    for step in steps:
        run(step)


if __name__ == "__main__":
    main()
