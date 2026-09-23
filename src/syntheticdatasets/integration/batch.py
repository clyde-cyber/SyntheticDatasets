"""Batch inference over CSV inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from syntheticdatasets.mlops.inference import Predictor


def run_batch_inference(
    input_csv: str | Path,
    output_csv: str | Path,
    model_dir: str | Path = "artifacts/models/latest",
    target_column: str | None = "target",
) -> Path:
    frame = pd.read_csv(input_csv)
    features = frame.drop(columns=[target_column]) if target_column and target_column in frame.columns else frame
    predictor = Predictor(model_dir)
    preds = predictor.predict(features)
    out = features.copy()
    out["prediction"] = preds
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
