"""Smoke test: generate -> train -> metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from syntheticdatasets.data.synthetic import SyntheticDataGenerator
from syntheticdatasets.training.train import train_model


@pytest.fixture()
def tmp_artifacts(tmp_path: Path) -> Path:
    return tmp_path


def test_generate_synthetic_shape():
    ds = SyntheticDataGenerator(n_samples=120, n_features=6, random_state=0).generate()
    assert ds.n_samples == 120
    assert ds.n_features == 6
    assert set(ds.target.unique()).issubset({0, 1})
    assert "source" in ds.metadata


def test_generate_train_metrics(tmp_artifacts: Path):
    data_path = tmp_artifacts / "synthetic.csv"
    SyntheticDataGenerator(n_samples=200, n_features=6, random_state=1).save_csv(data_path)

    data_cfg = tmp_artifacts / "data.yaml"
    train_cfg = tmp_artifacts / "train.yaml"
    model_dir = tmp_artifacts / "models"
    metrics_path = tmp_artifacts / "metrics.json"

    data_cfg.write_text(
        "\n".join(
            [
                "source: synthetic",
                "synthetic:",
                "  n_samples: 200",
                "  n_features: 6",
                "  n_informative: 4",
                "  n_classes: 2",
                "  random_state: 1",
                f"  output_path: {data_path.as_posix()}",
                "split:",
                "  test_size: 0.25",
                "  random_state: 1",
            ]
        ),
        encoding="utf-8",
    )
    train_cfg.write_text(
        "\n".join(
            [
                f"data_config: {data_cfg.as_posix()}",
                "model:",
                "  name: logistic_regression",
                "  params:",
                "    max_iter: 300",
                "    random_state: 1",
                "features:",
                "  scale: true",
                "  select_k: null",
                "artifacts:",
                f"  model_dir: {model_dir.as_posix()}",
                f"  metrics_path: {metrics_path.as_posix()}",
                "tuning:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    metrics = train_model(train_cfg)
    assert metrics_path.exists()
    assert (model_dir / "model.joblib").exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in saved
    assert saved["accuracy"] >= 0.5
    assert metrics["n_train"] + metrics["n_test"] == 200
