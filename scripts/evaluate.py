#!/usr/bin/env python3
"""CLI: evaluate a saved model bundle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from syntheticdatasets.training.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
