from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.paths import DATA_RAW_DIR, ensure_directory


def save_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_directory(target.parent)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return target


def save_dataset_raw(payload: Any, filename: str = "dataset_proprio.json") -> Path:
    return save_json(DATA_RAW_DIR / filename, payload)
