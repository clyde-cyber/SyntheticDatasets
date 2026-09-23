"""Lightweight prediction / ops metric summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_predictions(y_pred: np.ndarray | pd.Series) -> dict[str, Any]:
    series = pd.Series(y_pred)
    counts = series.value_counts(normalize=True).to_dict()
    return {
        "n": int(len(series)),
        "class_proportions": {str(k): float(v) for k, v in counts.items()},
    }
