"""Model baselines and registry."""

from syntheticdatasets.models.baseline import build_baseline
from syntheticdatasets.models.registry import get_model, list_models

__all__ = ["build_baseline", "get_model", "list_models"]
