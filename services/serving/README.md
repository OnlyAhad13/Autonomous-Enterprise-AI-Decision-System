# Model Serving Service

Scalable model inference powered by FastAPI and Ray Serve.

## Overview

This service handles:
- Low-latency model inference
- Auto-scaling based on load
- A/B testing and canary deployments
- API gateway and request routing

## Structure

```
serving/
├── api/                 # FastAPI application
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── health.py
│   └── middleware/
│       └── auth.py
├── deployments/         # Ray Serve deployments
│   ├── __init__.py
│   ├── model_deployment.py
│   └── ensemble.py
├── middleware/          # Request middleware
│   └── logging.py
├── monitoring/          # Serving metrics
│   └── metrics.py
├── config.py
├── Dockerfile
└── README.md
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Model prediction |
| `/batch-predict` | POST | Batch predictions |
| `/explain` | POST | Model explanations |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Usage

```python
import httpx

response = httpx.post(
    "http://localhost:8000/predict",
    json={
        "features": {"user_id": "123", "context": {...}},
        "model_name": "decision_model"
    }
)
result = response.json()
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RAY_ADDRESS` | Ray cluster address | `auto` |
| `MODEL_NAME` | Default model to serve | `decision_model` |
| `NUM_REPLICAS` | Number of model replicas | `2` |
| `MAX_BATCH_SIZE` | Maximum batch size | `32` |

## Development

```bash
# Start FastAPI server
poetry run uvicorn api.main:app --reload

# Start Ray Serve
poetry run serve run deployments.model_deployment:deployment

# Run with Docker
docker build -t serving .
docker run -p 8000:8000 serving
```

## Performance

- **Latency P50**: < 10ms
- **Latency P99**: < 50ms
- **Throughput**: 10,000 RPS per replica
