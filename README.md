# SyntheticDatasets

Machine learning workspace for training on real and generated synthetic datasets.

Covers data pipelines, feature engineering, model training and tuning, MLOps deployment, monitoring, and system integration.

## Layout

```
configs/                  # YAML configs for data + training
src/syntheticdatasets/
  data/                   # schemas, loaders, synthetic generator, pipelines
  features/               # engineering + selection
  models/                 # baselines + registry
  training/               # train / evaluate / tuning
  mlops/                  # packaging + inference
  monitoring/             # metrics + drift stubs
  integration/            # API + batch entrypoints
scripts/                  # CLI entrypoints
tests/                    # smoke tests
docs/ARCHITECTURE.md      # component map
.github/workflows/ci.yml  # pytest on PR
```

## Install

```bash
python -m pip install -e ".[dev]"
# or
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quickstart

Generate a small synthetic classification dataset:

```bash
python scripts/generate_synthetic.py --output artifacts/data/synthetic.csv --n-samples 500
```

Train a baseline model (writes metrics/artifacts under `artifacts/`):

```bash
python scripts/train.py --config configs/train.yaml
```

Evaluate a saved model:

```bash
python scripts/evaluate.py --model-dir artifacts/models/latest --data artifacts/data/synthetic.csv
```

Run tests:

```bash
pytest -q
```

## Swapping in real data

Implement `RealDataLoader.load()` in `src/syntheticdatasets/data/loaders.py` to return a
`TabularDataset` (same schema as the synthetic generator). Point `configs/data.yaml`
at your source; training and evaluation paths stay unchanged.
