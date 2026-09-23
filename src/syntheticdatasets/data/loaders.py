"""Data loader interfaces for real and synthetic sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from syntheticdatasets.data.schemas import FeatureSpec, TabularDataset


class BaseDataLoader(ABC):
    """Common loader contract — real and synthetic both return TabularDataset."""

    @abstractmethod
    def load(self) -> TabularDataset:
        raise NotImplementedError


class RealDataLoader(BaseDataLoader):
    """Stub for swapping in production / real tabular data later.

    Expected CSV layout: feature columns + a target column (default ``target``).
    """

    def __init__(self, path: str | Path, target_column: str = "target") -> None:
        self.path = Path(path)
        self.target_column = target_column

    def load(self) -> TabularDataset:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Real data not found at {self.path}. "
                "Point configs/data.yaml real.path at your CSV, or use source: synthetic."
            )
        frame = pd.read_csv(self.path)
        if self.target_column not in frame.columns:
            raise ValueError(f"Missing target column '{self.target_column}' in {self.path}")
        target = frame[self.target_column]
        features = frame.drop(columns=[self.target_column])
        specs = [
            FeatureSpec(name=col, dtype=str(features[col].dtype), role="feature")
            for col in features.columns
        ]
        specs.append(FeatureSpec(name=self.target_column, dtype=str(target.dtype), role="target"))
        return TabularDataset(
            features=features,
            target=target,
            feature_specs=specs,
            metadata={"source": "real", "path": str(self.path)},
        )


def load_csv_dataset(path: str | Path, target_column: str = "target") -> TabularDataset:
    """Convenience loader for any CSV already matching the shared schema."""
    return RealDataLoader(path=path, target_column=target_column).load()


def dataset_from_frames(
    features: pd.DataFrame,
    target: pd.Series,
    metadata: dict[str, Any] | None = None,
) -> TabularDataset:
    specs = [
        FeatureSpec(name=col, dtype=str(features[col].dtype), role="feature")
        for col in features.columns
    ]
    specs.append(FeatureSpec(name=str(target.name or "target"), dtype=str(target.dtype), role="target"))
    return TabularDataset(
        features=features,
        target=target,
        feature_specs=specs,
        metadata=metadata or {},
    )
