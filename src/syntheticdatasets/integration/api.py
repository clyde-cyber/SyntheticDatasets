"""FastAPI-style plain function stubs for online inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from syntheticdatasets.mlops.inference import Predictor

_predictor: Predictor | None = None


def health() -> dict[str, str]:
    return {"status": "ok", "service": "syntheticdatasets"}


def get_predictor(model_dir: str | Path = "artifacts/models/latest") -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(model_dir)
    return _predictor


def predict_one(
    features: dict[str, Any],
    model_dir: str | Path = "artifacts/models/latest",
) -> dict[str, Any]:
    """Score a single feature dict — wrap with FastAPI later if desired."""
    predictor = get_predictor(model_dir)
    results = predictor.predict_records([features])
    return results[0]
