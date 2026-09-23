"""Integration surfaces (API-style + batch)."""

from syntheticdatasets.integration.api import predict_one, health
from syntheticdatasets.integration.batch import run_batch_inference

__all__ = ["predict_one", "health", "run_batch_inference"]
