"""
Tests for Prediction Service.

Tests cover:
- Status codes for all endpoints
- Response schema validation
- Batch prediction functionality
- Error handling
"""

import pytest
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def client():
    """Create test client."""
    from services.predict.app import app, model_manager
    
    # Ensure model is loaded
    if not model_manager.is_loaded:
        model_manager.load()
    
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample feature dictionary."""
    return {
        "age": 35,
        "income": 75000,
        "tenure_months": 24,
        "subscription_tier": "premium",
    }


@pytest.fixture
def sample_csv():
    """Sample CSV content."""
    csv_content = """age,income,tenure_months,subscription_tier
35,75000,24,premium
42,90000,36,basic
28,55000,12,premium
"""
    return csv_content


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint returns status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_health_response_schema(self, client):
        """Test health response matches schema."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["model_version"], str)


# ============================================================================
# Single Prediction Tests
# ============================================================================


class TestSinglePrediction:
    """Tests for /predict endpoint."""
    
    def test_predict_returns_200(self, client, sample_features):
        """Test successful prediction returns 200."""
        response = client.post(
            "/predict",
            json={"features": sample_features}
        )
        
        assert response.status_code == 200
    
    def test_predict_response_schema(self, client, sample_features):
        """Test prediction response matches schema."""
        response = client.post(
            "/predict",
            json={"features": sample_features}
        )
        
        data = response.json()
        
        assert "prediction" in data
        assert "model_version" in data
        assert "inference_time_ms" in data
        assert "timestamp" in data
    
    def test_predict_with_probabilities(self, client, sample_features):
        """Test prediction with probabilities."""
        response = client.post(
            "/predict",
            json={
                "features": sample_features,
                "return_probabilities": True,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Probabilities may or may not be present depending on model type
        assert "prediction" in data
    
    def test_predict_inference_time(self, client, sample_features):
        """Test that inference time is reported."""
        response = client.post(
            "/predict",
            json={"features": sample_features}
        )
        
        data = response.json()
        
        assert data["inference_time_ms"] >= 0
        assert data["inference_time_ms"] < 10000  # Less than 10 seconds
    
    def test_predict_empty_features_error(self, client):
        """Test prediction with empty features."""
        response = client.post(
            "/predict",
            json={"features": {}}
        )
        
        # Should either return 200 or 400/422 depending on model requirements
        assert response.status_code in [200, 400, 422]
    
    def test_predict_missing_body_error(self, client):
        """Test prediction without request body."""
        response = client.post("/predict")
        
        assert response.status_code == 422


# ============================================================================
# Batch Prediction Tests
# ============================================================================


class TestBatchPrediction:
    """Tests for /batch_predict endpoint."""
    
    def test_batch_predict_returns_200(self, client, sample_csv):
        """Test successful batch prediction returns 200."""
        csv_file = io.BytesIO(sample_csv.encode())
        
        response = client.post(
            "/batch_predict",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        
        assert response.status_code == 200
    
    def test_batch_predict_response_schema(self, client, sample_csv):
        """Test batch prediction response matches schema."""
        csv_file = io.BytesIO(sample_csv.encode())
        
        response = client.post(
            "/batch_predict",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        
        data = response.json()
        
        assert "predictions" in data
        assert "row_count" in data
        assert "model_version" in data
        assert "inference_time_ms" in data
        assert "failed_rows" in data
    
    def test_batch_predict_row_count(self, client, sample_csv):
        """Test batch prediction returns correct row count."""
        csv_file = io.BytesIO(sample_csv.encode())
        
        response = client.post(
            "/batch_predict",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        
        data = response.json()
        
        assert data["row_count"] == 3
        assert len(data["predictions"]) == 3
    
    def test_batch_predict_non_csv_error(self, client):
        """Test batch prediction with non-CSV file."""
        json_file = io.BytesIO(b'{"key": "value"}')
        
        response = client.post(
            "/batch_predict",
            files={"file": ("test.json", json_file, "application/json")},
        )
        
        assert response.status_code == 400
    
    def test_batch_predict_no_file_error(self, client):
        """Test batch prediction without file."""
        response = client.post("/batch_predict")
        
        assert response.status_code == 422
    
    def test_batch_predict_download(self, client, sample_csv):
        """Test batch prediction download returns CSV."""
        csv_file = io.BytesIO(sample_csv.encode())
        
        response = client.post(
            "/batch_predict/download",
            files={"file": ("test.csv", csv_file, "text/csv")},
        )
        
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")


# ============================================================================
# Explain Tests
# ============================================================================


class TestExplain:
    """Tests for /explain endpoint."""
    
    def test_explain_returns_200(self, client, sample_features):
        """Test successful explain returns 200."""
        response = client.post(
            "/explain",
            json={
                "features": sample_features,
                "method": "shap",
                "top_k": 5,
            }
        )
        
        assert response.status_code == 200
    
    def test_explain_response_schema(self, client, sample_features):
        """Test explain response matches schema."""
        response = client.post(
            "/explain",
            json={
                "features": sample_features,
                "method": "shap",
                "top_k": 5,
            }
        )
        
        data = response.json()
        
        assert "prediction" in data
        assert "feature_importance" in data
        assert "method" in data
        assert "model_version" in data
    
    def test_explain_feature_importance_format(self, client, sample_features):
        """Test feature importance is in correct format."""
        response = client.post(
            "/explain",
            json={
                "features": sample_features,
                "method": "shap",
                "top_k": 3,
            }
        )
        
        data = response.json()
        
        for importance in data["feature_importance"]:
            assert "feature" in importance
            assert "importance" in importance
    
    def test_explain_respects_top_k(self, client, sample_features):
        """Test explain respects top_k parameter."""
        response = client.post(
            "/explain",
            json={
                "features": sample_features,
                "method": "shap",
                "top_k": 2,
            }
        )
        
        data = response.json()
        
        assert len(data["feature_importance"]) <= 2


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    """Tests for /metrics endpoint."""
    
    def test_metrics_returns_200(self, client):
        """Test metrics endpoint returns 200."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
    
    def test_metrics_response_format(self, client):
        """Test metrics response format."""
        response = client.get("/metrics")
        data = response.json()
        
        assert "model_loaded" in data
        assert "model_version" in data


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_json_returns_422(self, client):
        """Test invalid JSON returns 422."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        
        assert response.status_code == 422
    
    def test_404_for_unknown_endpoint(self, client):
        """Test 404 for unknown endpoint."""
        response = client.get("/unknown_endpoint")
        
        assert response.status_code == 404


# ============================================================================
# OpenAPI Documentation Tests
# ============================================================================


class TestDocumentation:
    """Tests for API documentation."""
    
    def test_docs_endpoint_available(self, client):
        """Test /docs endpoint is available."""
        response = client.get("/docs")
        
        assert response.status_code == 200
    
    def test_openapi_json_available(self, client):
        """Test OpenAPI JSON schema is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
    
    def test_openapi_has_all_endpoints(self, client):
        """Test OpenAPI includes all endpoints."""
        response = client.get("/openapi.json")
        data = response.json()
        
        paths = data["paths"]
        
        assert "/predict" in paths
        assert "/batch_predict" in paths
        assert "/explain" in paths
        assert "/health" in paths


# ============================================================================
# Sample Request Documentation
# ============================================================================


"""
# Sample Requests for Testing

## Curl Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {
      "age": 35,
      "income": 75000,
      "tenure_months": 24,
      "subscription_tier": "premium"
    },
    "return_probabilities": true
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch_predict \\
  -F "file=@test_data.csv"
```

### Explain
```bash
curl -X POST http://localhost:8000/explain \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {"age": 35, "income": 75000},
    "method": "shap",
    "top_k": 5
  }'
```

## Python Examples

### Using requests
```python
import requests

# Single prediction
resp = requests.post(
    "http://localhost:8000/predict",
    json={"features": {"age": 35, "income": 75000}}
)
print(resp.json())

# Batch prediction
with open("data.csv", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/batch_predict",
        files={"file": f}
    )
print(resp.json())
```

### Using httpx (async)
```python
import httpx
import asyncio

async def predict():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8000/predict",
            json={"features": {"age": 35}}
        )
        return resp.json()

result = asyncio.run(predict())
```
"""
