"""
Locust Load Test for Prediction Service.

Tests /predict endpoint with realistic payload distributions.

Run with:
    locust -f perf/locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10

Web UI:
    locust -f perf/locustfile.py --host=http://localhost:8000
    # Open http://localhost:8089
"""

import random
import json
import time
from typing import Dict, Any, List

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner


# ============================================================================
# Payload Generators
# ============================================================================


class PayloadGenerator:
    """Generate realistic prediction payloads."""
    
    # Feature distributions
    AGE_DISTRIBUTION = (18, 80, 35, 12)  # min, max, mean, stddev
    INCOME_DISTRIBUTION = (20000, 500000, 75000, 40000)
    TENURE_DISTRIBUTION = (0, 120, 24, 18)
    
    SUBSCRIPTION_TIERS = ["basic", "standard", "premium", "enterprise"]
    TIER_WEIGHTS = [0.4, 0.3, 0.2, 0.1]
    
    REGIONS = ["US-East", "US-West", "EU-West", "EU-Central", "APAC"]
    
    @classmethod
    def gaussian_int(cls, min_val: int, max_val: int, mean: float, stddev: float) -> int:
        """Generate gaussian distributed integer."""
        value = random.gauss(mean, stddev)
        return max(min_val, min(max_val, int(value)))
    
    @classmethod
    def generate_user_features(cls) -> Dict[str, Any]:
        """Generate realistic user features."""
        return {
            "age": cls.gaussian_int(*cls.AGE_DISTRIBUTION),
            "income": cls.gaussian_int(*cls.INCOME_DISTRIBUTION),
            "tenure_months": cls.gaussian_int(*cls.TENURE_DISTRIBUTION),
            "subscription_tier": random.choices(
                cls.SUBSCRIPTION_TIERS, 
                weights=cls.TIER_WEIGHTS
            )[0],
            "region": random.choice(cls.REGIONS),
            "num_products": random.randint(1, 10),
            "has_support_ticket": random.random() < 0.15,
            "last_login_days": random.randint(0, 30),
        }
    
    @classmethod
    def generate_batch_payload(cls, size: int) -> List[Dict[str, Any]]:
        """Generate batch of features."""
        return [cls.generate_user_features() for _ in range(size)]


# ============================================================================
# Metrics Collection
# ============================================================================


class MetricsCollector:
    """Collect and analyze performance metrics."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.errors: int = 0
        self.successes: int = 0
    
    def record(self, latency_ms: float, success: bool):
        self.latencies.append(latency_ms)
        if success:
            self.successes += 1
        else:
            self.errors += 1
    
    @property
    def p50(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        return sorted_lat[len(sorted_lat) // 2]
    
    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[idx]
    
    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[idx]


metrics = MetricsCollector()


# ============================================================================
# Locust Users
# ============================================================================


class PredictionUser(HttpUser):
    """
    Simulates user making prediction requests.
    
    Behavior:
    - 80% single predictions
    - 15% batch predictions
    - 5% explain requests
    """
    
    wait_time = between(0.1, 1.0)  # Think time between requests
    
    @task(80)
    def single_predict(self):
        """Single prediction request."""
        payload = {
            "features": PayloadGenerator.generate_user_features(),
            "return_probabilities": random.random() < 0.3,
        }
        
        start = time.time()
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
        ) as response:
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data:
                    response.success()
                    metrics.record(latency_ms, True)
                else:
                    response.failure("Missing prediction in response")
                    metrics.record(latency_ms, False)
            else:
                response.failure(f"Status {response.status_code}")
                metrics.record(latency_ms, False)
    
    @task(15)
    def batch_predict(self):
        """Batch prediction request."""
        import io
        import csv
        
        # Generate CSV
        batch_size = random.choice([10, 50, 100, 200])
        features_list = PayloadGenerator.generate_batch_payload(batch_size)
        
        output = io.StringIO()
        if features_list:
            writer = csv.DictWriter(output, fieldnames=features_list[0].keys())
            writer.writeheader()
            writer.writerows(features_list)
        
        csv_content = output.getvalue()
        
        start = time.time()
        
        with self.client.post(
            "/batch_predict",
            files={"file": ("batch.csv", csv_content, "text/csv")},
            catch_response=True,
        ) as response:
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data.get("row_count") == batch_size:
                    response.success()
                    metrics.record(latency_ms, True)
                else:
                    response.failure("Row count mismatch")
                    metrics.record(latency_ms, False)
            else:
                response.failure(f"Status {response.status_code}")
                metrics.record(latency_ms, False)
    
    @task(5)
    def explain_predict(self):
        """Explain prediction request."""
        payload = {
            "features": PayloadGenerator.generate_user_features(),
            "method": random.choice(["shap", "lime"]),
            "top_k": random.choice([3, 5, 10]),
        }
        
        start = time.time()
        
        with self.client.post(
            "/explain",
            json=payload,
            catch_response=True,
        ) as response:
            latency_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if "feature_importance" in data:
                    response.success()
                    metrics.record(latency_ms, True)
                else:
                    response.failure("Missing feature_importance")
                    metrics.record(latency_ms, False)
            else:
                response.failure(f"Status {response.status_code}")
                metrics.record(latency_ms, False)


class HighThroughputUser(HttpUser):
    """High-frequency user for stress testing."""
    
    wait_time = between(0.01, 0.1)  # Minimal wait
    
    @task
    def rapid_predict(self):
        """Rapid single predictions."""
        payload = {
            "features": PayloadGenerator.generate_user_features(),
        }
        
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# ============================================================================
# Event Hooks
# ============================================================================


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print metrics summary when test stops."""
    if not isinstance(environment.runner, WorkerRunner):
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Requests: {metrics.successes + metrics.errors}")
        print(f"Successful: {metrics.successes}")
        print(f"Failed: {metrics.errors}")
        print(f"P50 Latency: {metrics.p50:.2f} ms")
        print(f"P95 Latency: {metrics.p95:.2f} ms")
        print(f"P99 Latency: {metrics.p99:.2f} ms")
        print("=" * 60)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize test environment."""
    if isinstance(environment.runner, MasterRunner):
        print("Running as Master node")
    elif isinstance(environment.runner, WorkerRunner):
        print("Running as Worker node")


# ============================================================================
# CLI Usage Examples
# ============================================================================

"""
# Basic load test (100 users)
locust -f perf/locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --run-time=5m

# Stress test (500 users)
locust -f perf/locustfile.py --host=http://localhost:8000 --users=500 --spawn-rate=50 --run-time=10m

# Headless mode with CSV output
locust -f perf/locustfile.py --host=http://localhost:8000 --users=200 --spawn-rate=20 --run-time=5m --headless --csv=results

# Distributed mode (1 master + 4 workers)
# Terminal 1 (Master):
locust -f perf/locustfile.py --master --host=http://localhost:8000

# Terminal 2-5 (Workers):
locust -f perf/locustfile.py --worker --master-host=127.0.0.1

# With specific user class
locust -f perf/locustfile.py --host=http://localhost:8000 HighThroughputUser --users=100
"""
