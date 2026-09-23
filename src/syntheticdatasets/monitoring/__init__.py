"""Monitoring metrics and drift stubs."""

from syntheticdatasets.monitoring.metrics import summarize_predictions
from syntheticdatasets.monitoring.drift import simple_drift_report

__all__ = ["summarize_predictions", "simple_drift_report"]
