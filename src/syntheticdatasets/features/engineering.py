"""Lightweight feature engineering transforms."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Optional scaling and passthrough transforms for tabular features."""

    def __init__(self, scale: bool = True) -> None:
        self.scale = scale
        self._scaler: StandardScaler | None = StandardScaler() if scale else None
        self.feature_names_: list[str] | None = None

    def fit(self, x: pd.DataFrame) -> "FeatureEngineer":
        self.feature_names_ = list(x.columns)
        if self._scaler is not None:
            self._scaler.fit(x)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            return x.copy()
        scaled = self._scaler.transform(x)
        cols = self.feature_names_ or list(x.columns)
        return pd.DataFrame(scaled, columns=cols, index=x.index)

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.fit(x).transform(x)
