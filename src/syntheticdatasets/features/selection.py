"""Feature selection helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


class FeatureSelector:
    """Select top-k features via ANOVA F-value; no-op when k is None."""

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self._selector: SelectKBest | None = None
        self.selected_features_: list[str] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        if self.k is None or self.k >= x.shape[1]:
            self.selected_features_ = list(x.columns)
            return self
        self._selector = SelectKBest(score_func=f_classif, k=self.k)
        self._selector.fit(x, y)
        mask = self._selector.get_support()
        self.selected_features_ = [c for c, keep in zip(x.columns, mask) if keep]
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("FeatureSelector must be fit before transform")
        return x[self.selected_features_].copy()

    def fit_transform(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(x, y).transform(x)
