"""Allow ``python -m src.phase1_preprocessing.phase1`` execution."""

import sys
from pathlib import Path

# When invoked directly, ensure the project root is on sys.path
# so that absolute imports resolve correctly.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.phase1_preprocessing.phase1.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
