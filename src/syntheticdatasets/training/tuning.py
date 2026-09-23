"""Hyperparameter tuning hooks (opt-in via config)."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV


DEFAULT_SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "LogisticRegression": {
        "C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "penalty": ["l2"],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
}


def maybe_tune(estimator: Any, x, y, tuning_cfg: dict[str, Any] | None = None):
    """Run a light RandomizedSearchCV when tuning.enabled is true; else return estimator."""
    tuning_cfg = tuning_cfg or {}
    if not tuning_cfg.get("enabled", False):
        return estimator

    name = type(estimator).__name__
    space = tuning_cfg.get("param_distributions") or DEFAULT_SEARCH_SPACES.get(name)
    if not space:
        return estimator

    search = RandomizedSearchCV(
        estimator,
        param_distributions=space,
        n_iter=int(tuning_cfg.get("n_iter", 10)),
        cv=int(tuning_cfg.get("cv", 3)),
        scoring=tuning_cfg.get("scoring", "f1_weighted"),
        random_state=int(tuning_cfg.get("random_state", 42)),
        n_jobs=tuning_cfg.get("n_jobs", -1),
        refit=True,
    )
    search.fit(x, y)
    return search.best_estimator_
