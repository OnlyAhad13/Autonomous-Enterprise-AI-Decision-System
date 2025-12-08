# ML Platform Service

Experiment tracking and model registry powered by MLflow.

## Overview

This service manages:
- Experiment tracking and comparison
- Model versioning and registry
- Model packaging and deployment
- Artifact storage

## Structure

```
ml-platform/
├── experiments/         # Experiment definitions
│   ├── __init__.py
│   └── config.py
├── models/              # Model training code
│   ├── __init__.py
│   ├── base_model.py
│   ├── classifier.py
│   └── regressor.py
├── pipelines/           # Training pipelines
│   ├── __init__.py
│   └── training_pipeline.py
├── registry/            # Model registry configs
│   └── models.yaml
├── config.py
├── Dockerfile
└── README.md
```

## Models

| Model | Type | Stage | Metrics |
|-------|------|-------|---------|
| `decision_model` | Classification | Production | AUC: 0.92 |
| `scoring_model` | Regression | Staging | RMSE: 0.15 |
| `ranking_model` | LTR | Production | NDCG: 0.87 |

## Usage

```python
import mlflow

# Start experiment run
with mlflow.start_run(experiment_id="decision-system"):
    mlflow.log_params({"learning_rate": 0.01})
    mlflow.log_metrics({"auc": 0.92, "f1": 0.88})
    mlflow.sklearn.log_model(model, "model")

# Load model from registry
model = mlflow.pyfunc.load_model("models:/decision_model/Production")
predictions = model.predict(features)
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server | `http://localhost:5000` |
| `MLFLOW_ARTIFACT_ROOT` | Artifact storage location | `s3://bucket/mlflow` |
| `MLFLOW_REGISTRY_URI` | Model registry URI | Same as tracking URI |

## Development

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Run training pipeline
poetry run python pipelines/training_pipeline.py

# Register model
mlflow models serve -m "models:/decision_model/Production" -p 8000
```
