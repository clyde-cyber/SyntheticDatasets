"""Evaluation metrics and offline evaluate entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from syntheticdatasets.data.loaders import load_csv_dataset


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def evaluate_model(model_dir: str | Path, data_path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(model_dir) / "model.joblib")
    dataset = load_csv_dataset(data_path)
    x, y = dataset.to_xy()
    x = bundle["engineer"].transform(x)
    x = bundle["selector"].transform(x)
    model = bundle["model"]
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x) if hasattr(model, "predict_proba") else None
    metrics = classification_metrics(y, y_pred, y_proba)
    metrics["model_dir"] = str(model_dir)
    metrics["data_path"] = str(data_path)
    out = Path(model_dir) / "eval_metrics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("--model-dir", default="artifacts/models/latest")
    parser.add_argument("--data", default="artifacts/data/synthetic.csv")
    args = parser.parse_args()
    print(json.dumps(evaluate_model(args.model_dir, args.data), indent=2))


if __name__ == "__main__":
    main()
