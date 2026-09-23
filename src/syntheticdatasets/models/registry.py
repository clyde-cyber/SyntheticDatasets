"""Simple in-process model registry."""

from __future__ import annotations

from typing import Any, Callable

from syntheticdatasets.models.baseline import build_baseline

_REGISTRY: dict[str, Callable[..., Any]] = {
    "logistic_regression": lambda **p: build_baseline("logistic_regression", **p),
    "random_forest": lambda **p: build_baseline("random_forest", **p),
}


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def get_model(name: str, **params: Any):
    if name not in _REGISTRY:
        raise KeyError(f"Model {name!r} not registered. Available: {list_models()}")
    return _REGISTRY[name](**params)


def register_model(name: str, factory: Callable[..., Any]) -> None:
    _REGISTRY[name] = factory
