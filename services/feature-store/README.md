# Feature Store Service

Centralized feature management powered by Feast.

## Overview

This service handles:
- Feature definitions and registry
- Online/offline feature serving
- Point-in-time correct retrieval
- Feature monitoring and statistics

## Structure

```
feature-store/
├── features/            # Feature view definitions
│   ├── __init__.py
│   ├── user_features.py
│   └── event_features.py
├── entities/            # Entity definitions
│   ├── __init__.py
│   └── entities.py
├── data_sources/        # Data source configurations
│   └── sources.py
├── feature_services/    # Feature service definitions
│   └── services.py
├── feature_store.yaml   # Feast configuration
├── config.py
└── README.md
```

## Feature Views

| Feature View | Entity | Features | Refresh |
|--------------|--------|----------|---------|
| `user_features` | user_id | profile, preferences | Daily |
| `event_features` | event_id | aggregations, stats | Hourly |
| `realtime_features` | entity_id | streaming features | Real-time |

## Usage

```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Get online features
features = store.get_online_features(
    features=["user_features:total_spend", "user_features:last_active"],
    entity_rows=[{"user_id": "123"}]
).to_dict()

# Get historical features for training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:total_spend"]
).to_df()
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `FEAST_OFFLINE_STORE` | Offline store type | `file` |
| `FEAST_ONLINE_STORE` | Online store type | `redis` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |

## Development

```bash
# Apply feature definitions
feast apply

# Materialize features to online store
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)

# Start feature server
feast serve
```
