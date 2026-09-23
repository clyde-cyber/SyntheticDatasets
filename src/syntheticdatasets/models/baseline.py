"""Sklearn baseline classifiers."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_baseline(name: str = "logistic_regression", **params: Any):
    """Return a sklearn estimator by name."""
    key = name.lower().replace("-", "_")
    if key in {"logistic_regression", "logreg", "lr"}:
        defaults = {"max_iter": 500, "random_state": 42}
        defaults.update(params)
        return LogisticRegression(**defaults)
    if key in {"random_forest", "rf"}:
        defaults = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    raise ValueError(f"Unknown baseline model: {name!r}")
