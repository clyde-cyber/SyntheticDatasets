"""Model bundle save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return out


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
