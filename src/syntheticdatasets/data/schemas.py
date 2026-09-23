"""Shared schema for real and synthetic tabular datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class FeatureSpec(BaseModel):
    """Description of a single feature column."""

    name: str
    dtype: str = "float"
    role: str = Field(default="feature", description="feature | target | id")


class TabularDataset(BaseModel):
    """In-memory tabular classification dataset shared by real + synthetic paths."""

    model_config = {"arbitrary_types_allowed": True}

    features: pd.DataFrame
    target: pd.Series
    feature_specs: list[FeatureSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target")
    @classmethod
    def _target_aligned(cls, v: pd.Series, info):  # type: ignore[no-untyped-def]
        return v

    def to_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.features, self.target

    @property
    def n_samples(self) -> int:
        return len(self.features)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    def head(self, n: int = 5) -> pd.DataFrame:
        frame = self.features.copy()
        frame["target"] = self.target.values
        return frame.head(n)
