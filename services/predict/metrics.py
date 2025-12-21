"""
FastAPI Metrics Instrumentation.

Provides Prometheus metrics for FastAPI applications.

Usage:
    from services.predict.metrics import setup_metrics, track_request
    
    app = FastAPI()
    setup_metrics(app)
"""

import time
import logging
from typing import Callable
from functools import wraps

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger(__name__)


# ==============================================================================
# Metrics Definitions
# ==============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "handler", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "handler"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUEST_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently in progress",
    ["method", "handler"],
)

# Model metrics
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["model_version", "status"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_duration_seconds",
    "Model prediction latency in seconds",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Model drift metrics
MODEL_DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Current model drift score",
    ["model_name"],
)

FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "Feature drift score",
    ["feature"],
)

# Data freshness metrics
LAST_TRAINING_DATA_TIMESTAMP = Gauge(
    "last_training_data_timestamp",
    "Timestamp of last training data update",
)

FEATURE_STORE_SYNC_LAG = Gauge(
    "feature_store_sync_lag_seconds",
    "Feature store synchronization lag in seconds",
)

# Agent metrics
AGENT_ACTION_COUNT = Counter(
    "agent_action_total",
    "Total agent actions",
    ["action", "status"],
)

AGENT_EXECUTION_DURATION = Histogram(
    "agent_execution_duration_seconds",
    "Agent execution duration in seconds",
    ["objective"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# Service info
SERVICE_INFO = Info(
    "service",
    "Service information",
)


# ==============================================================================
# Middleware
# ==============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP request metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path
        
        # Normalize path for metrics (remove IDs, etc.)
        handler = self._normalize_path(path)
        
        # Track in-progress requests
        REQUEST_IN_PROGRESS.labels(method=method, handler=handler).inc()
        
        start_time = time.time()
        response = None
        status = "500"
        
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        except Exception as e:
            status = "500"
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=method,
                handler=handler,
                status=status,
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                handler=handler,
            ).observe(duration)
            
            REQUEST_IN_PROGRESS.labels(
                method=method,
                handler=handler,
            ).dec()
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics labels."""
        # Remove query params
        path = path.split("?")[0]
        
        # Common patterns to normalize
        import re
        
        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
        )
        
        # Replace numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)
        
        return path


# ==============================================================================
# Setup Functions
# ==============================================================================

def setup_metrics(app: FastAPI, service_name: str = "predict") -> None:
    """
    Set up Prometheus metrics for FastAPI app.
    
    Args:
        app: FastAPI application instance.
        service_name: Name of the service.
    """
    # Add middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Set service info
    SERVICE_INFO.info({
        "name": service_name,
        "version": "1.0.0",
    })
    
    # Add metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return StarletteResponse(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )
    
    logger.info(f"Prometheus metrics enabled for {service_name}")


# ==============================================================================
# Decorators for Manual Instrumentation
# ==============================================================================

def track_prediction(model_version: str = "unknown"):
    """
    Decorator to track model prediction metrics.
    
    Usage:
        @track_prediction(model_version="1.0")
        def predict(features):
            return model.predict(features)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                
                PREDICTION_COUNT.labels(
                    model_version=model_version,
                    status=status,
                ).inc()
                
                PREDICTION_LATENCY.labels(
                    model_version=model_version,
                ).observe(duration)
        
        return wrapper
    return decorator


def track_agent_action(action: str):
    """
    Decorator to track agent action metrics.
    
    Usage:
        @track_agent_action("trigger_retrain")
        def trigger_retrain():
            # ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                AGENT_ACTION_COUNT.labels(
                    action=action,
                    status=status,
                ).inc()
        
        return wrapper
    return decorator


# ==============================================================================
# Metric Update Functions
# ==============================================================================

def update_drift_score(model_name: str, score: float) -> None:
    """Update model drift score metric."""
    MODEL_DRIFT_SCORE.labels(model_name=model_name).set(score)


def update_feature_drift(feature: str, score: float) -> None:
    """Update feature drift score metric."""
    FEATURE_DRIFT_SCORE.labels(feature=feature).set(score)


def update_training_data_timestamp(timestamp: float) -> None:
    """Update last training data timestamp."""
    LAST_TRAINING_DATA_TIMESTAMP.set(timestamp)


def update_feature_store_lag(lag_seconds: float) -> None:
    """Update feature store sync lag."""
    FEATURE_STORE_SYNC_LAG.set(lag_seconds)


# ==============================================================================
# Example Usage
# ==============================================================================

"""
# Integration Example

from fastapi import FastAPI
from services.predict.metrics import (
    setup_metrics,
    track_prediction,
    update_drift_score,
)

app = FastAPI()
setup_metrics(app, service_name="predict")

class ModelManager:
    @track_prediction(model_version="1.0.0")
    def predict(self, features):
        # Model prediction logic
        return prediction

# Update drift score (e.g., from a scheduled job)
update_drift_score("forecasting-model", 0.08)
"""
