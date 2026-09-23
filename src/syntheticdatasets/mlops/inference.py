"""Simple offline inference stub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from syntheticdatasets.mlops.packaging import load_bundle


class Predictor:
    """Load a trained bundle and run predict / predict_proba."""

    def __init__(self, model_dir: str | Path) -> None:
        model_path = Path(model_dir)
        bundle_file = model_path / "model.joblib" if model_path.is_dir() else model_path
        self.bundle = load_bundle(bundle_file)
        self.model = self.bundle["model"]
        self.engineer = self.bundle["engineer"]
        self.selector = self.bundle["selector"]

    def _prepare(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.selector.transform(self.engineer.transform(x))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(x))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Underlying model has no predict_proba")
        return self.model.predict_proba(self._prepare(x))

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = pd.DataFrame(records)
        preds = self.predict(frame)
        return [{"prediction": int(p)} for p in preds]
