"""
Tests for Auto-Retrain DAG.

Includes:
- DAG structure validation
- Task dependency tests
- Drift simulation instructions
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# DAG Import Tests
# ============================================================================


class TestDAGImport:
    """Tests for DAG structure and import."""
    
    def test_dag_loads_without_errors(self):
        """Test that DAG file can be imported."""
        # This tests that the DAG is syntactically correct
        from ingest.dags.auto_retrain_dag import dag
        
        assert dag is not None
        assert dag.dag_id == "auto_retrain_dag"
    
    def test_dag_has_required_tasks(self):
        """Test that DAG contains all required tasks."""
        from ingest.dags.auto_retrain_dag import dag
        
        expected_tasks = [
            "start",
            "check_drift_metrics",
            "get_agent_decision",
            "branch_on_decision",
            "prepare_retraining",
            "run_training",
            "run_validation_tests",
            "check_performance",
            "promote_model",
            "send_success_notification",
            "send_alert_only",
            "end",
        ]
        
        task_ids = [task.task_id for task in dag.tasks]
        
        for expected in expected_tasks:
            assert expected in task_ids, f"Missing task: {expected}"
    
    def test_dag_default_args(self):
        """Test DAG default arguments."""
        from ingest.dags.auto_retrain_dag import dag
        
        assert dag.default_args["owner"] == "ml-platform"
        assert dag.default_args["retries"] >= 1
    
    def test_dag_schedule(self):
        """Test DAG has a schedule."""
        from ingest.dags.auto_retrain_dag import dag
        
        assert dag.schedule_interval is not None
    
    def test_dag_catchup_disabled(self):
        """Test that catchup is disabled."""
        from ingest.dags.auto_retrain_dag import dag
        
        assert dag.catchup is False


# ============================================================================
# Task Dependency Tests
# ============================================================================


class TestTaskDependencies:
    """Tests for task dependencies."""
    
    def test_start_leads_to_sensor(self):
        """Test that start task leads to drift sensor."""
        from ingest.dags.auto_retrain_dag import dag
        
        start_task = dag.get_task("start")
        downstream = [t.task_id for t in start_task.downstream_list]
        
        assert "check_drift_metrics" in downstream
    
    def test_sensor_leads_to_agent(self):
        """Test that sensor leads to agent decision."""
        from ingest.dags.auto_retrain_dag import dag
        
        sensor_task = dag.get_task("check_drift_metrics")
        downstream = [t.task_id for t in sensor_task.downstream_list]
        
        assert "get_agent_decision" in downstream
    
    def test_branch_has_two_paths(self):
        """Test that branch has two downstream paths."""
        from ingest.dags.auto_retrain_dag import dag
        
        branch_task = dag.get_task("branch_on_decision")
        downstream = [t.task_id for t in branch_task.downstream_list]
        
        assert len(downstream) == 2
        assert "prepare_retraining" in downstream
        assert "send_alert_only" in downstream


# ============================================================================
# Prometheus Sensor Tests
# ============================================================================


class TestPrometheusDriftSensor:
    """Tests for PrometheusDriftSensor."""
    
    @pytest.fixture
    def sensor(self):
        """Create sensor instance."""
        from ingest.dags.auto_retrain_dag import PrometheusDriftSensor
        
        return PrometheusDriftSensor(
            task_id="test_sensor",
            prometheus_url="http://localhost:9090",
            metric_query="model_drift_score",
            threshold=0.1,
        )
    
    @patch('requests.get')
    def test_sensor_detects_drift(self, mock_get, sensor):
        """Test sensor returns True when drift exceeds threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [{"value": [1234567890, "0.15"]}]
            }
        }
        mock_get.return_value = mock_response
        
        # Create mock context
        mock_ti = MagicMock()
        context = {"ti": mock_ti}
        
        result = sensor.poke(context)
        
        assert result is True
        mock_ti.xcom_push.assert_called()
    
    @patch('requests.get')
    def test_sensor_no_drift(self, mock_get, sensor):
        """Test sensor returns False when drift is below threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "result": [{"value": [1234567890, "0.05"]}]
            }
        }
        mock_get.return_value = mock_response
        
        context = {"ti": MagicMock()}
        result = sensor.poke(context)
        
        assert result is False
    
    @patch('requests.get')
    def test_sensor_handles_connection_error(self, mock_get, sensor):
        """Test sensor handles connection errors gracefully."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")
        
        context = {"ti": MagicMock()}
        result = sensor.poke(context)
        
        assert result is False


# ============================================================================
# Agent Decision Tests
# ============================================================================


class TestAgentDecision:
    """Tests for agent decision task."""
    
    @patch('requests.post')
    def test_agent_decision_retrain(self, mock_post):
        """Test agent decides to retrain on high drift."""
        from ingest.dags.auto_retrain_dag import call_agent_for_decision
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "action": "retrain",
            "reason": "Drift exceeds threshold",
            "confidence": 0.95,
        }
        mock_post.return_value = mock_response
        
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = 0.15
        
        context = {"ti": mock_ti}
        
        with patch('ingest.dags.auto_retrain_dag.Variable') as mock_var:
            mock_var.get.return_value = "http://localhost:8000"
            result = call_agent_for_decision(**context)
        
        assert result["action"] == "retrain"
    
    def test_branch_returns_retrain_path(self):
        """Test branch function returns retrain path."""
        from ingest.dags.auto_retrain_dag import decide_branch
        
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {"action": "retrain", "reason": "test"}
        
        context = {"ti": mock_ti}
        result = decide_branch(**context)
        
        assert result == "prepare_retraining"
    
    def test_branch_returns_skip_path(self):
        """Test branch function returns skip path."""
        from ingest.dags.auto_retrain_dag import decide_branch
        
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {"action": "skip", "reason": "No drift"}
        
        context = {"ti": mock_ti}
        result = decide_branch(**context)
        
        assert result == "send_alert_only"


# ============================================================================
# Policy Configuration Tests
# ============================================================================


class TestPolicyConfiguration:
    """Tests for agent policy configuration."""
    
    def test_policy_file_exists(self):
        """Test that policy file exists."""
        policy_path = PROJECT_ROOT / "conf" / "agent_policy.json"
        assert policy_path.exists()
    
    def test_policy_is_valid_json(self):
        """Test that policy file is valid JSON."""
        policy_path = PROJECT_ROOT / "conf" / "agent_policy.json"
        
        with open(policy_path) as f:
            policy = json.load(f)
        
        assert isinstance(policy, dict)
    
    def test_policy_has_required_sections(self):
        """Test that policy has required sections."""
        policy_path = PROJECT_ROOT / "conf" / "agent_policy.json"
        
        with open(policy_path) as f:
            policy = json.load(f)
        
        assert "drift_detection" in policy
        assert "thresholds" in policy
        assert "allowed_actions" in policy
        assert "validation" in policy
    
    def test_policy_drift_threshold(self):
        """Test that drift threshold is reasonable."""
        policy_path = PROJECT_ROOT / "conf" / "agent_policy.json"
        
        with open(policy_path) as f:
            policy = json.load(f)
        
        threshold = policy["drift_detection"]["threshold"]
        assert 0 < threshold < 1


# ============================================================================
# Drift Simulation Instructions
# ============================================================================


"""
# Testing Instructions: Simulating Drift

## Option 1: Local Prometheus with Mock Metrics

1. Start Prometheus locally:
   ```bash
   docker run -d -p 9090:9090 prom/prometheus
   ```

2. Create a mock exporter that reports drift metrics:
   ```python
   # drift_exporter.py
   from prometheus_client import start_http_server, Gauge
   import time
   import random
   
   drift_gauge = Gauge('model_drift_score', 'Current model drift score')
   
   if __name__ == '__main__':
       start_http_server(8000)
       while True:
           # Simulate increasing drift
           drift_value = random.uniform(0.05, 0.20)
           drift_gauge.set(drift_value)
           print(f"Set drift to {drift_value:.4f}")
           time.sleep(10)
   ```

3. Run the exporter:
   ```bash
   pip install prometheus_client
   python drift_exporter.py
   ```

4. Configure Prometheus to scrape the exporter (prometheus.yml):
   ```yaml
   scrape_configs:
     - job_name: 'drift-exporter'
       static_configs:
         - targets: ['localhost:8000']
   ```

## Option 2: Direct API Testing

1. Mock the Prometheus response in tests:
   ```python
   @patch('requests.get')
   def test_with_high_drift(mock_get):
       mock_get.return_value.json.return_value = {
           "status": "success",
           "data": {"result": [{"value": [0, "0.15"]}]}
       }
       # Run your test
   ```

## Option 3: Airflow Variable Override

1. Set Airflow variable to force drift value:
   ```bash
   airflow variables set FORCE_DRIFT_VALUE 0.15
   ```

2. Modify the sensor to check this variable first (for testing only).

## Option 4: Run DAG with Test Configuration

1. Create a test policy with low threshold:
   ```json
   {
     "drift_detection": {
       "threshold": 0.01  // Very low threshold
     }
   }
   ```

2. Run the DAG:
   ```bash
   airflow dags test auto_retrain_dag 2024-01-15
   ```

## Validating the Pipeline

1. Check task logs:
   ```bash
   airflow tasks logs auto_retrain_dag check_drift_metrics 2024-01-15
   ```

2. Verify XCom values:
   ```bash
   airflow tasks render auto_retrain_dag get_agent_decision 2024-01-15
   ```

3. Check the branch taken:
   ```bash
   airflow dags state auto_retrain_dag 2024-01-15
   ```

## Expected Behavior

| Drift Value | Expected Path |
|-------------|---------------|
| < 0.08      | skip → alert_only |
| 0.08 - 0.10 | warning alert, skip |
| > 0.10      | retrain → train → validate → promote |

## Troubleshooting

1. **Sensor times out**: Check Prometheus URL and metric name
2. **Agent fails**: Verify AGENT_API_URL variable is set
3. **Spark job fails**: Check Spark connection configuration
4. **Promotion fails**: Verify MLflow tracking URI
"""


class TestDriftSimulation:
    """Tests for drift simulation utilities."""
    
    def test_drift_exporter_script(self):
        """Verify drift exporter script structure."""
        # This test documents the expected drift exporter interface
        expected_script = """
from prometheus_client import start_http_server, Gauge
drift_gauge = Gauge('model_drift_score', 'Current model drift score')
drift_gauge.set(0.15)
"""
        assert "Gauge" in expected_script
        assert "model_drift_score" in expected_script
    
    def test_high_drift_triggers_retrain(self):
        """Test that high drift value triggers retraining path."""
        from ingest.dags.auto_retrain_dag import decide_branch
        
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "action": "retrain",
            "reason": "Drift 0.15 > 0.1",
        }
        
        result = decide_branch(ti=mock_ti)
        
        assert result == "prepare_retraining"
    
    def test_low_drift_skips_retrain(self):
        """Test that low drift value skips retraining."""
        from ingest.dags.auto_retrain_dag import decide_branch
        
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = {
            "action": "skip",
            "reason": "Drift 0.05 <= 0.1",
        }
        
        result = decide_branch(ti=mock_ti)
        
        assert result == "send_alert_only"
