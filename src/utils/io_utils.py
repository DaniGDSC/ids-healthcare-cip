"""IO utilities."""

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    path = Path(path)
    with open(path, 'r') as f:
        return json.load(f)