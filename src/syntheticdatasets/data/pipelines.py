"""Pipeline construction helpers for train/eval flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from syntheticdatasets.data.loaders import RealDataLoader, load_csv_dataset
from syntheticdatasets.data.schemas import TabularDataset
from syntheticdatasets.data.synthetic import SyntheticDataGenerator


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_dataset_from_config(config: dict[str, Any] | str | Path) -> TabularDataset:
    """Construct a TabularDataset from data.yaml (or an already-loaded dict)."""
    if not isinstance(config, dict):
        config = load_yaml(config)

    source = config.get("source", "synthetic")
    if source == "synthetic":
        params = dict(config.get("synthetic", {}))
        output_path = params.pop("output_path", None)
        generator = SyntheticDataGenerator(**{
            k: v for k, v in params.items() if k in {
                "n_samples", "n_features", "n_informative", "n_classes",
                "class_sep", "random_state",
            }
        })
        dataset = generator.generate()
        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            frame = dataset.features.copy()
            frame["target"] = dataset.target.values
            frame.to_csv(out, index=False)
            dataset.metadata["saved_path"] = str(out)
        return dataset

    if source == "real":
        real = config.get("real", {})
        return RealDataLoader(
            path=real.get("path", "data/raw/real.csv"),
            target_column=real.get("target_column", "target"),
        ).load()

    raise ValueError(f"Unknown data source: {source!r}")


def load_or_build(path: str | Path | None, config: dict[str, Any] | None = None) -> TabularDataset:
    """Prefer an existing CSV; otherwise build from config."""
    if path and Path(path).exists():
        return load_csv_dataset(path)
    if config is None:
        raise FileNotFoundError(f"No dataset at {path} and no config provided")
    return build_dataset_from_config(config)
