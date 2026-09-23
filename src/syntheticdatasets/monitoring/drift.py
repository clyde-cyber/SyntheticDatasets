"""Simple distribution-drift stub for tabular features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def simple_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare column means; flag absolute relative shift above threshold.

    This is a placeholder — swap in PSI / KS tests for production monitoring.
    """
    shared = [c for c in reference.columns if c in current.columns]
    drifts: dict[str, float] = {}
    flagged: list[str] = []
    for col in shared:
        ref_mean = float(np.mean(reference[col]))
        cur_mean = float(np.mean(current[col]))
        denom = abs(ref_mean) if abs(ref_mean) > 1e-9 else 1.0
        shift = abs(cur_mean - ref_mean) / denom
        drifts[col] = shift
        if shift > threshold:
            flagged.append(col)
    return {
        "n_features_checked": len(shared),
        "threshold": threshold,
        "drifts": drifts,
        "flagged": flagged,
    }
