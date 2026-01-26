"""Run full IDS Healthcare CIP pipeline."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: str):
    """
    Execute a command safely without shell injection vulnerabilities.
    
    Args:
        cmd: Command string to execute (will be parsed into arguments)
    
    Raises:
        subprocess.CalledProcessError: If command execution fails
    """
    print(f"\n>>> {cmd}")
    # Parse command string into argument list safely
    args = cmd.split()
    try:
        result = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            shell=False,
            check=True,
            capture_output=False
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError as e:
        print(f"\n❌ Error: Command not found - {e}")
        sys.exit(1)


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
