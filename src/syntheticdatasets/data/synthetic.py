"""Synthetic tabular classification dataset generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import make_classification

from syntheticdatasets.data.schemas import FeatureSpec, TabularDataset


class SyntheticDataGenerator:
    """Emit a small sklearn-friendly tabular classification set."""

    def __init__(
        self,
        n_samples: int = 500,
        n_features: int = 8,
        n_informative: int = 5,
        n_classes: int = 2,
        class_sep: float = 1.2,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_classes = n_classes
        self.class_sep = class_sep
        self.random_state = random_state
        self.kwargs = kwargs

    def generate(self) -> TabularDataset:
        x, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=max(0, self.n_features - self.n_informative - 1),
            n_classes=self.n_classes,
            class_sep=self.class_sep,
            random_state=self.random_state,
            **self.kwargs,
        )
        feature_names = [f"feature_{i}" for i in range(self.n_features)]
        features = pd.DataFrame(x, columns=feature_names)
        target = pd.Series(y, name="target")
        specs = [FeatureSpec(name=n, dtype="float", role="feature") for n in feature_names]
        specs.append(FeatureSpec(name="target", dtype="int", role="target"))
        return TabularDataset(
            features=features,
            target=target,
            feature_specs=specs,
            metadata={
                "source": "synthetic",
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "random_state": self.random_state,
            },
        )

    def save_csv(self, path: str | Path) -> Path:
        dataset = self.generate()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = dataset.features.copy()
        frame["target"] = dataset.target.values
        frame.to_csv(out, index=False)
        return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic tabular data")
    parser.add_argument("--output", default="artifacts/data/synthetic.csv")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    path = SyntheticDataGenerator(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        random_state=args.random_state,
    ).save_csv(args.output)
    print(f"Wrote synthetic dataset to {path}")


if __name__ == "__main__":
    main()
