"""Training, evaluation, and tuning."""

from syntheticdatasets.training.train import train_model
from syntheticdatasets.training.evaluate import evaluate_model

__all__ = ["train_model", "evaluate_model"]
