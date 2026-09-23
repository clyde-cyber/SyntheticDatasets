"""Train a baseline model and persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.model_selection import train_test_split

from syntheticdatasets.data.pipelines import build_dataset_from_config, load_yaml
from syntheticdatasets.features.engineering import FeatureEngineer
from syntheticdatasets.features.selection import FeatureSelector
from syntheticdatasets.models.registry import get_model
from syntheticdatasets.training.evaluate import classification_metrics
from syntheticdatasets.training.tuning import maybe_tune


def train_model(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        config = load_yaml(config)

    data_cfg_path = config.get("data_config", "configs/data.yaml")
    data_cfg = load_yaml(data_cfg_path) if isinstance(data_cfg_path, str) else data_cfg_path
    dataset = build_dataset_from_config(data_cfg)
    x, y = dataset.to_xy()

    split = data_cfg.get("split", {})
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=split.get("test_size", 0.2),
        random_state=split.get("random_state", 42),
        stratify=y if y.nunique() > 1 else None,
    )

    feat_cfg = config.get("features", {})
    engineer = FeatureEngineer(scale=feat_cfg.get("scale", True))
    x_train_e = engineer.fit_transform(x_train)
    x_test_e = engineer.transform(x_test)

    selector = FeatureSelector(k=feat_cfg.get("select_k"))
    x_train_s = selector.fit_transform(x_train_e, y_train)
    x_test_s = selector.transform(x_test_e)

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "logistic_regression")
    model_params = dict(model_cfg.get("params", {}))
    estimator = get_model(model_name, **model_params)

    tuning_cfg = config.get("tuning", {})
    estimator = maybe_tune(estimator, x_train_s, y_train, tuning_cfg)

    estimator.fit(x_train_s, y_train)
    y_pred = estimator.predict(x_test_s)
    y_proba = None
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(x_test_s)

    metrics = classification_metrics(y_test, y_pred, y_proba)
    metrics["model"] = model_name
    metrics["n_train"] = int(len(x_train))
    metrics["n_test"] = int(len(x_test))
    metrics["n_features"] = int(x_train_s.shape[1])

    art = config.get("artifacts", {})
    model_dir = Path(art.get("model_dir", "artifacts/models/latest"))
    metrics_path = Path(art.get("metrics_path", "artifacts/metrics/train_metrics.json"))
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": estimator,
            "engineer": engineer,
            "selector": selector,
            "feature_names": selector.selected_features_,
            "model_name": model_name,
        },
        model_dir / "model.joblib",
    )
    with open(model_dir / "config_snapshot.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    metrics["model_dir"] = str(model_dir)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    metrics = train_model(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
