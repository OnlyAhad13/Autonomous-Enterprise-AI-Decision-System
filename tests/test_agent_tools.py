"""
Unit tests for Agent Tool Wrappers.

Tests cover:
- Airflow tool functions
- MLflow tool functions  
- Kafka tool functions
- Prometheus tool functions
All with mocked HTTP/API clients.
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock
import json

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Import Tool Classes
# ============================================================================


from agents.tools.tool_airflow import AirflowTool, ToolResult
from agents.tools.tool_prometheus import PrometheusTool


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    def _mock_response(status_code=200, json_data=None, text=""):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.text = text
        return response
    return _mock_response


# ============================================================================
# Airflow Tool Tests
# ============================================================================


class TestAirflowTool:
    """Tests for Airflow tool wrapper."""
    
    @pytest.fixture
    def tool(self):
        """Create Airflow tool instance."""
        return AirflowTool(
            base_url="http://localhost:8080",
            username="airflow",
            password="airflow",
        )
    
    @patch('agents.tools.tool_airflow.requests.request')
    def test_trigger_dag_success(self, mock_request, tool, mock_response):
        """Test successful DAG trigger."""
        mock_request.return_value = mock_response(200, {
            "dag_run_id": "manual__2024-01-15",
            "state": "running",
            "logical_date": "2024-01-15T00:00:00Z",
        })
        
        result = tool.trigger_dag("etl_pipeline", conf={"date": "2024-01-15"})
        
        assert result.success is True
        assert result.data["run_id"] == "manual__2024-01-15"
        assert result.data["state"] == "running"
        assert result.requires_confirmation is True
    
    @patch('agents.tools.tool_airflow.requests.request')
    def test_trigger_dag_failure(self, mock_request, tool, mock_response):
        """Test failed DAG trigger."""
        mock_request.return_value = mock_response(404, text="DAG not found")
        
        result = tool.trigger_dag("nonexistent_dag")
        
        assert result.success is False
        assert "404" in result.error
    
    @patch('agents.tools.tool_airflow.requests.request')
    def test_get_dag_status_success(self, mock_request, tool, mock_response):
        """Test getting DAG status."""
        mock_request.return_value = mock_response(200, {
            "dag_runs": [{
                "dag_run_id": "scheduled__2024-01-15",
                "state": "success",
                "logical_date": "2024-01-15T00:00:00Z",
                "execution_date": "2024-01-15T00:00:00Z",
                "start_date": "2024-01-15T00:05:00Z",
                "end_date": "2024-01-15T00:45:00Z",
            }]
        })
        
        result = tool.get_dag_status("etl_pipeline")
        
        assert result.success is True
        assert result.data["state"] == "success"
        assert result.requires_confirmation is False
    
    @patch('agents.tools.tool_airflow.requests.request')
    def test_list_dags_success(self, mock_request, tool, mock_response):
        """Test listing DAGs."""
        mock_request.return_value = mock_response(200, {
            "dags": [
                {"dag_id": "etl_pipeline", "is_paused": False, "is_active": True},
                {"dag_id": "ml_training", "is_paused": True, "is_active": True},
            ]
        })
        
        result = tool.list_dags()
        
        assert result.success is True
        assert result.data["total"] == 2
        assert len(result.data["dags"]) == 2
    
    @patch('agents.tools.tool_airflow.requests.request')
    def test_connection_error(self, mock_request, tool):
        """Test handling connection errors."""
        import requests
        mock_request.side_effect = requests.RequestException("Connection refused")
        
        result = tool.trigger_dag("test_dag")
        
        assert result.success is False
        assert "Connection refused" in result.error


# ============================================================================
# MLflow Tool Tests
# ============================================================================


class TestMLflowTool:
    """Tests for MLflow tool wrapper."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock MLflow client."""
        return MagicMock()
    
    @patch('agents.tools.tool_mlflow.mlflow')
    @patch('agents.tools.tool_mlflow.MlflowClient')
    def test_get_latest_run_metrics(self, mock_client_class, mock_mlflow):
        """Test fetching latest run metrics."""
        from agents.tools.tool_mlflow import MLflowTool
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock run
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1705312800000
        mock_run.data.metrics = {"mape": 8.5, "rmse": 1234.56}
        mock_run.data.params = {"epochs": "100"}
        mock_client.search_runs.return_value = [mock_run]
        
        tool = MLflowTool()
        result = tool.get_latest_run_metrics("forecasting-pipeline")
        
        assert result.success is True
        assert result.data["run_id"] == "abc123"
        assert result.data["metrics"]["mape"] == 8.5
    
    @patch('agents.tools.tool_mlflow.mlflow')
    @patch('agents.tools.tool_mlflow.MlflowClient')
    def test_get_latest_run_no_experiment(self, mock_client_class, mock_mlflow):
        """Test when experiment doesn't exist."""
        from agents.tools.tool_mlflow import MLflowTool
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_experiment_by_name.return_value = None
        
        tool = MLflowTool()
        result = tool.get_latest_run_metrics("nonexistent")
        
        assert result.success is False
        assert "not found" in result.error
    
    @patch('agents.tools.tool_mlflow.mlflow')
    @patch('agents.tools.tool_mlflow.MlflowClient')
    def test_register_model(self, mock_client_class, mock_mlflow):
        """Test model registration."""
        from agents.tools.tool_mlflow import MLflowTool
        
        mock_result = MagicMock()
        mock_result.version = "1"
        mock_mlflow.register_model.return_value = mock_result
        
        tool = MLflowTool()
        result = tool.register_model("abc123", "churn-model")
        
        assert result.success is True
        assert result.data["version"] == "1"
        assert result.requires_confirmation is True
    
    @patch('agents.tools.tool_mlflow.mlflow')
    @patch('agents.tools.tool_mlflow.MlflowClient')
    def test_get_model_versions(self, mock_client_class, mock_mlflow):
        """Test listing model versions."""
        from agents.tools.tool_mlflow import MLflowTool
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.status = "READY"
        mock_version.run_id = "abc123"
        mock_version.creation_timestamp = 1705312800000
        mock_client.search_model_versions.return_value = [mock_version]
        
        tool = MLflowTool()
        result = tool.get_model_versions("churn-model")
        
        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["versions"][0]["stage"] == "Production"


# ============================================================================
# Kafka Tool Tests
# ============================================================================


class TestKafkaTool:
    """Tests for Kafka tool wrapper."""
    
    def test_kafka_tool_no_package(self):
        """Test handling when kafka-python not installed."""
        with patch.dict('sys.modules', {'kafka': None}):
            # The import check happens at module level, so we test the result handling
            from agents.tools.tool_kafka import KafkaTool
            
            tool = KafkaTool()
            # Methods should return error when package not available
            # This is handled internally
    
    @patch('agents.tools.tool_kafka.HAS_KAFKA', False)
    def test_get_consumer_lag_no_kafka(self):
        """Test get_consumer_lag when kafka not installed."""
        from agents.tools.tool_kafka import KafkaTool
        
        tool = KafkaTool()
        result = tool.get_consumer_lag("test-group")
        
        assert result.success is False
        assert "not installed" in result.error
    
    @patch('agents.tools.tool_kafka.HAS_KAFKA', False)
    def test_list_topics_no_kafka(self):
        """Test list_topics when kafka not installed."""
        from agents.tools.tool_kafka import KafkaTool
        
        tool = KafkaTool()
        result = tool.list_topics()
        
        assert result.success is False


# ============================================================================
# Prometheus Tool Tests
# ============================================================================


class TestPrometheusTool:
    """Tests for Prometheus tool wrapper."""
    
    @pytest.fixture
    def tool(self):
        """Create Prometheus tool instance."""
        return PrometheusTool(base_url="http://localhost:9090")
    
    @patch('agents.tools.tool_prometheus.requests.get')
    def test_query_success(self, mock_get, tool, mock_response):
        """Test successful PromQL query."""
        mock_get.return_value = mock_response(200, {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {"metric": {"job": "api"}, "value": [1705312800, "245.67"]}
                ]
            }
        })
        
        result = tool.query("rate(http_requests_total[5m])")
        
        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["results"][0]["value"] == "245.67"
    
    @patch('agents.tools.tool_prometheus.requests.get')
    def test_query_error(self, mock_get, tool, mock_response):
        """Test PromQL query error."""
        mock_get.return_value = mock_response(200, {
            "status": "error",
            "error": "invalid query syntax"
        })
        
        result = tool.query("invalid[query")
        
        assert result.success is False
        assert "invalid query" in result.error
    
    @patch('agents.tools.tool_prometheus.requests.get')
    def test_query_range_success(self, mock_get, tool, mock_response):
        """Test successful range query."""
        mock_get.return_value = mock_response(200, {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"instance": "localhost:9090"},
                        "values": [
                            [1705312800, "1"],
                            [1705312860, "1"],
                        ]
                    }
                ]
            }
        })
        
        result = tool.query_range(
            "up",
            start="2024-01-15T00:00:00Z",
            end="2024-01-15T01:00:00Z",
            step="1m"
        )
        
        assert result.success is True
        assert result.data["series_count"] == 1
        assert result.data["results"][0]["point_count"] == 2
    
    @patch('agents.tools.tool_prometheus.requests.get')
    def test_get_alerts_success(self, mock_get, tool, mock_response):
        """Test getting alerts."""
        mock_get.return_value = mock_response(200, {
            "status": "success",
            "data": {
                "alerts": [
                    {
                        "labels": {"alertname": "HighCPU", "severity": "critical"},
                        "annotations": {"summary": "CPU > 90%"},
                        "state": "firing",
                        "activeAt": "2024-01-15T10:00:00Z",
                    }
                ]
            }
        })
        
        result = tool.get_alerts()
        
        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["firing"] == 1
        assert result.data["alerts"][0]["name"] == "HighCPU"
    
    @patch('agents.tools.tool_prometheus.requests.get')
    def test_get_targets_success(self, mock_get, tool, mock_response):
        """Test getting scrape targets."""
        mock_get.return_value = mock_response(200, {
            "status": "success",
            "data": {
                "activeTargets": [
                    {
                        "labels": {"job": "prometheus", "instance": "localhost:9090"},
                        "health": "up",
                        "lastScrape": "2024-01-15T10:00:00Z",
                        "lastScrapeDuration": 0.05,
                    }
                ],
                "droppedTargets": []
            }
        })
        
        result = tool.get_targets()
        
        assert result.success is True
        assert result.data["active_count"] == 1
    
    @patch('agents.tools.tool_prometheus.requests.get')
    def test_connection_error(self, mock_get, tool):
        """Test handling connection errors."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")
        
        result = tool.query("up")
        
        assert result.success is False
        assert "Connection refused" in result.error


# ============================================================================
# ToolResult Tests
# ============================================================================


class TestToolResult:
    """Tests for ToolResult dataclass."""
    
    def test_success_result(self):
        """Test creating success result."""
        result = ToolResult(success=True, data={"key": "value"})
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.requires_confirmation is False
    
    def test_error_result(self):
        """Test creating error result."""
        result = ToolResult(success=False, data=None, error="Something went wrong")
        
        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"
    
    def test_confirmation_required(self):
        """Test result with confirmation required."""
        result = ToolResult(
            success=True,
            data={"dag_id": "test"},
            requires_confirmation=True,
            action_description="Trigger DAG 'test'",
        )
        
        assert result.requires_confirmation is True
        assert result.action_description == "Trigger DAG 'test'"


# ============================================================================
# Prompt Template Tests
# ============================================================================


class TestPromptTemplates:
    """Tests for prompt templates."""
    
    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        from agents.prompts.system_prompt import SYSTEM_PROMPT
        
        assert SYSTEM_PROMPT is not None
        assert "confirmation" in SYSTEM_PROMPT.lower()
        assert "airflow" in SYSTEM_PROMPT.lower()
    
    def test_tool_descriptions(self):
        """Test tool descriptions."""
        from agents.prompts.system_prompt import TOOL_DESCRIPTIONS, get_tool_descriptions
        
        assert "airflow" in TOOL_DESCRIPTIONS
        assert "mlflow" in TOOL_DESCRIPTIONS
        assert "kafka" in TOOL_DESCRIPTIONS
        assert "prometheus" in TOOL_DESCRIPTIONS
        
        desc = get_tool_descriptions(["airflow"])
        assert "trigger_dag" in desc
    
    def test_few_shot_examples(self):
        """Test few-shot examples."""
        from agents.prompts.few_shot_examples import (
            FEW_SHOT_EXAMPLES,
            get_examples_for_tool,
        )
        
        assert len(FEW_SHOT_EXAMPLES) > 0
        
        airflow_examples = get_examples_for_tool("airflow")
        assert len(airflow_examples) > 0
        assert "user" in airflow_examples[0]
        assert "assistant" in airflow_examples[0]
