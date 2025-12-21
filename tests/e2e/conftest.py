"""
E2E Test Fixtures.

Provides fixtures for end-to-end pipeline testing.
"""

import os
import time
import pytest
import requests
from pathlib import Path

# Configuration
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:15000")
PREDICT_URL = os.environ.get("PREDICT_API_URL", "http://localhost:18000")
E2E_MODE = os.environ.get("E2E_MODE", "false").lower() == "true"


def is_service_ready(url: str, timeout: int = 60) -> bool:
    """Wait for a service to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope="session")
def kafka_producer():
    """Create Kafka producer for testing."""
    try:
        from kafka import KafkaProducer
        import json
        
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        yield producer
        producer.close()
    except Exception as e:
        pytest.skip(f"Kafka not available: {e}")


@pytest.fixture(scope="session")
def kafka_consumer():
    """Create Kafka consumer for testing."""
    try:
        from kafka import KafkaConsumer
        import json
        
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            consumer_timeout_ms=30000,
            group_id="e2e-test-group",
        )
        yield consumer
        consumer.close()
    except Exception as e:
        pytest.skip(f"Kafka not available: {e}")


@pytest.fixture(scope="session")
def mlflow_client():
    """Create MLflow client for testing."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        yield mlflow.MlflowClient()
    except Exception as e:
        pytest.skip(f"MLflow not available: {e}")


@pytest.fixture(scope="session")
def predict_api():
    """Return predict API URL if available."""
    if is_service_ready(PREDICT_URL, timeout=30):
        return PREDICT_URL
    pytest.skip("Predict API not available")


@pytest.fixture(scope="session")
def sample_events():
    """Generate sample events for testing."""
    import random
    from datetime import datetime
    
    events = []
    for i in range(100):
        events.append({
            "event_id": f"test-{i}",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": f"user-{random.randint(1, 10)}",
            "event_type": random.choice(["view", "click", "purchase"]),
            "value": round(random.uniform(10, 1000), 2),
        })
    return events


@pytest.fixture(scope="session")
def sample_features():
    """Sample features for prediction."""
    return {
        "age": 35,
        "income": 75000,
        "tenure_months": 24,
        "subscription_tier": "premium",
    }


# KPI Thresholds
KPI_THRESHOLDS = {
    "ingest_latency_p99_ms": 500,
    "prediction_latency_p99_ms": 200,
    "error_rate_percent": 1.0,
    "throughput_events_per_sec": 100,
    "model_accuracy": 0.85,
}


@pytest.fixture(scope="session")
def kpi_thresholds():
    """Return KPI thresholds for assertions."""
    return KPI_THRESHOLDS
