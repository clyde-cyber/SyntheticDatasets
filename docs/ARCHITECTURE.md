# Architecture

SyntheticDatasets is organized so **real** and **synthetic** tabular data share one schema
and one training path.

## Component map

| Area | Package | Role |
|------|---------|------|
| Data pipelines | `syntheticdatasets.data` | Schemas, loaders, synthetic generator, YAML-driven pipeline |
| Features | `syntheticdatasets.features` | Scaling / engineering and optional SelectKBest |
| Models | `syntheticdatasets.models` | Sklearn baselines + registry |
| Training / tuning | `syntheticdatasets.training` | Fit, metrics, optional RandomizedSearchCV |
| MLOps | `syntheticdatasets.mlops` | Bundle packaging and Predictor |
| Monitoring | `syntheticdatasets.monitoring` | Prediction summaries + mean-shift drift stub |
| Integration | `syntheticdatasets.integration` | Online-style `predict_one` + batch CSV scoring |

## Data flow

```
configs/data.yaml
        │
        ▼
 build_dataset_from_config() ──► TabularDataset (features, target, specs)
        │
        ▼
 FeatureEngineer → FeatureSelector → Model (registry)
        │
        ▼
 artifacts/models/latest/model.joblib
 artifacts/metrics/train_metrics.json
        │
        ▼
 Predictor / evaluate / batch / api stubs
```

## Swapping real data

1. Set `source: real` in `configs/data.yaml`.
2. Point `real.path` at a CSV with a `target` column (or configure `target_column`).
3. Keep train/eval scripts unchanged — they consume `TabularDataset` only.
