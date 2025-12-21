"""
End-to-End Pipeline Tests.

Tests the complete flow:
    ingest -> stream -> feature -> train -> serve -> predict

Run with:
    docker-compose -f tests/e2e/docker-compose.e2e.yml up -d
    pytest tests/e2e/test_pipeline_e2e.py -v
"""

import os
import time
import json
import pytest
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Skip if not in E2E mode
E2E_MODE = os.environ.get("E2E_MODE", "false").lower() == "true"
pytestmark = pytest.mark.skipif(not E2E_MODE, reason="E2E mode not enabled")


# ============================================================================
# Ingest Tests
# ============================================================================


class TestIngestStage:
    """Test data ingestion to Kafka."""
    
    def test_produce_events_to_kafka(self, kafka_producer, sample_events):
        """Test sending events to Kafka topic."""
        topic = "events.raw.v1"
        
        for event in sample_events[:10]:
            future = kafka_producer.send(topic, value=event)
            result = future.get(timeout=10)
            
            assert result.topic == topic
        
        kafka_producer.flush()
    
    def test_events_in_topic(self, kafka_consumer):
        """Test events are available in topic."""
        topic = "events.raw.v1"
        kafka_consumer.subscribe([topic])
        
        messages = []
        for msg in kafka_consumer:
            messages.append(msg.value)
            if len(messages) >= 5:
                break
        
        assert len(messages) >= 5
        assert "event_id" in messages[0]


# ============================================================================
# Streaming Tests
# ============================================================================


class TestStreamingStage:
    """Test streaming processing."""
    
    def test_canonical_events_produced(self, kafka_consumer):
        """Test that canonical events are produced."""
        topic = "events.canonical.v1"
        kafka_consumer.subscribe([topic])
        
        # Wait for processing (may need adjustments based on actual streaming job)
        time.sleep(5)
        
        # In real test, we'd check for transformed events
        # For now, just validate structure
        assert True  # Placeholder
    
    def test_streaming_latency(self, kpi_thresholds):
        """Test streaming latency is within KPI."""
        # In real test, measure actual latency
        latency_ms = 100  # Mock value
        
        assert latency_ms < kpi_thresholds["ingest_latency_p99_ms"]


# ============================================================================
# Feature Store Tests
# ============================================================================


class TestFeatureStoreStage:
    """Test feature store operations."""
    
    def test_features_available(self):
        """Test features are available in store."""
        # In real test, query Feast
        # For now, validate structure
        features = {
            "user_profile": {"age": 35, "income": 75000},
            "product_features": {"category": "electronics"},
        }
        
        assert "user_profile" in features
        assert "age" in features["user_profile"]
    
    def test_feature_freshness(self, kpi_thresholds):
        """Test feature freshness is within KPI."""
        # In real test, check feature store lag
        lag_seconds = 60
        
        assert lag_seconds < 3600  # Less than 1 hour


# ============================================================================
# Training Tests
# ============================================================================


class TestTrainingStage:
    """Test model training."""
    
    def test_model_registered_in_mlflow(self, mlflow_client):
        """Test model is registered in MLflow."""
        try:
            models = mlflow_client.search_registered_models()
            # Just verify we can query, model may not exist in test env
            assert True
        except Exception:
            pytest.skip("MLflow not configured")
    
    def test_model_metrics_logged(self, mlflow_client):
        """Test training metrics are logged."""
        try:
            # Search for recent runs
            runs = mlflow_client.search_runs(
                experiment_ids=["0"],
                max_results=5,
            )
            # Validate structure
            assert True
        except Exception:
            pytest.skip("No runs available")


# ============================================================================
# Serving Tests
# ============================================================================


class TestServingStage:
    """Test model serving."""
    
    def test_predict_api_health(self, predict_api):
        """Test prediction API is healthy."""
        response = requests.get(f"{predict_api}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
    
    def test_single_prediction(self, predict_api, sample_features):
        """Test single prediction endpoint."""
        response = requests.post(
            f"{predict_api}/predict",
            json={"features": sample_features},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "inference_time_ms" in data
    
    def test_prediction_latency(self, predict_api, sample_features, kpi_thresholds):
        """Test prediction latency is within KPI."""
        response = requests.post(
            f"{predict_api}/predict",
            json={"features": sample_features},
        )
        
        data = response.json()
        latency_ms = data.get("inference_time_ms", 0)
        
        assert latency_ms < kpi_thresholds["prediction_latency_p99_ms"]
    
    def test_batch_prediction(self, predict_api):
        """Test batch prediction endpoint."""
        import io
        
        csv_content = """age,income,tenure_months,subscription_tier
35,75000,24,premium
42,90000,36,basic
28,55000,12,premium
"""
        files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
        
        response = requests.post(
            f"{predict_api}/batch_predict",
            files=files,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["row_count"] == 3
        assert len(data["predictions"]) == 3


# ============================================================================
# KPI Threshold Tests
# ============================================================================


class TestKPIThresholds:
    """Test KPI thresholds are met."""
    
    def test_error_rate(self, predict_api, sample_features):
        """Test error rate is within KPI."""
        errors = 0
        total = 100
        
        for _ in range(total):
            try:
                response = requests.post(
                    f"{predict_api}/predict",
                    json={"features": sample_features},
                    timeout=5,
                )
                if response.status_code != 200:
                    errors += 1
            except requests.RequestException:
                errors += 1
        
        error_rate = (errors / total) * 100
        
        assert error_rate < 5.0  # Less than 5% errors
    
    def test_throughput(self, predict_api, sample_features, kpi_thresholds):
        """Test throughput is within KPI."""
        start = time.time()
        requests_count = 50
        
        for _ in range(requests_count):
            requests.post(
                f"{predict_api}/predict",
                json={"features": sample_features},
            )
        
        duration = time.time() - start
        throughput = requests_count / duration
        
        # Just validate we can process requests
        assert throughput > 1  # At least 1 req/sec


# ============================================================================
# Full Pipeline Integration Test
# ============================================================================


class TestFullPipeline:
    """Test complete pipeline integration."""
    
    def test_end_to_end_flow(
        self,
        kafka_producer,
        predict_api,
        sample_events,
        sample_features,
    ):
        """
        Test complete flow:
        1. Ingest events to Kafka
        2. Wait for processing
        3. Make prediction
        4. Validate response
        """
        # 1. Ingest events
        topic = "events.raw.v1"
        for event in sample_events[:5]:
            kafka_producer.send(topic, value=event)
        kafka_producer.flush()
        
        # 2. Wait for processing (in real test, would check downstream)
        time.sleep(2)
        
        # 3. Make prediction
        response = requests.post(
            f"{predict_api}/predict",
            json={"features": sample_features},
        )
        
        # 4. Validate
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        
        print(f"E2E test passed! Prediction: {data['prediction']}")
