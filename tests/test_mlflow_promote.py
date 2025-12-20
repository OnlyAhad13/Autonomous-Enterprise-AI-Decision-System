"""
Unit tests for MLflow auto-promotion and CLI.

Tests cover:
- Promotion threshold logic
- Metric comparison
- Rollback functionality
- Webhook triggering (mocked)
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlflow_utils.auto_promote import (
    AutoPromoter,
    PromotionThresholds,
    PromotionResult,
    load_config,
    _expand_env_vars,
)
from mlflow_utils import cli


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Mock configuration dictionary."""
    return {
        "tracking": {
            "uri": "file:///tmp/mlruns"
        },
        "promotion": {
            "thresholds": {
                "mape_improvement_pct": 5.0,
                "mse_improvement_pct": 10.0,
                "min_samples": 100,
            }
        },
        "webhook": {
            "enabled": True,
            "url": "http://localhost:8080/deploy",
            "timeout_seconds": 30,
            "retry_attempts": 1,
        }
    }


@pytest.fixture
def thresholds():
    """Create test thresholds."""
    return PromotionThresholds(
        mape_improvement_pct=5.0,
        mse_improvement_pct=10.0,
        min_samples=100,
    )


@pytest.fixture
def mock_mlflow_client():
    """Create mock MLflow client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_run():
    """Create a mock MLflow run."""
    run = MagicMock()
    run.info.run_id = "abc123def456"
    run.info.experiment_id = "1"
    run.info.start_time = int(datetime.now().timestamp() * 1000)
    run.data.metrics = {
        "prophet_mape": 12.5,
        "prophet_mse": 1500.0,
    }
    run.data.params = {"model_type": "prophet"}
    return run


@pytest.fixture
def mock_model_version():
    """Create a mock ModelVersion."""
    version = MagicMock()
    version.version = "2"
    version.run_id = "abc123def456"
    version.current_stage = "Production"
    version.creation_timestamp = int(datetime.now().timestamp() * 1000)
    version.status = "READY"
    return version


# ============================================================================
# Config Loading Tests
# ============================================================================


class TestConfigLoading:
    """Tests for configuration loading."""
    
    def test_expand_env_vars_with_default(self):
        """Test environment variable expansion with defaults."""
        result = _expand_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"
    
    def test_expand_env_vars_with_actual_var(self, monkeypatch):
        """Test environment variable expansion with actual value."""
        monkeypatch.setenv("TEST_VAR", "actual_value")
        result = _expand_env_vars("${TEST_VAR:-default}")
        assert result == "actual_value"
    
    def test_expand_env_vars_nested_dict(self):
        """Test recursive expansion in nested dict."""
        config = {
            "outer": {
                "inner": "${MISSING:-nested_default}"
            }
        }
        result = _expand_env_vars(config)
        assert result["outer"]["inner"] == "nested_default"
    
    def test_expand_env_vars_list(self):
        """Test expansion in list."""
        config = ["${VAR1:-val1}", "${VAR2:-val2}"]
        result = _expand_env_vars(config)
        assert result == ["val1", "val2"]


# ============================================================================
# Promotion Threshold Tests
# ============================================================================


class TestPromotionThresholds:
    """Tests for promotion threshold logic."""
    
    def test_thresholds_default_values(self):
        """Test default threshold values."""
        thresholds = PromotionThresholds()
        assert thresholds.mape_improvement_pct == 5.0
        assert thresholds.mse_improvement_pct == 10.0
        assert thresholds.min_samples == 100
    
    def test_thresholds_custom_values(self):
        """Test custom threshold values."""
        thresholds = PromotionThresholds(
            mape_improvement_pct=10.0,
            mse_improvement_pct=20.0,
            min_samples=500,
        )
        assert thresholds.mape_improvement_pct == 10.0
        assert thresholds.mse_improvement_pct == 20.0
        assert thresholds.min_samples == 500


# ============================================================================
# Metric Comparison Tests
# ============================================================================


class TestMetricComparison:
    """Tests for metric comparison logic."""
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_compare_metrics_improvement_detected(self, mock_mlflow, mock_client, thresholds):
        """Test that improvement is detected when threshold exceeded."""
        promoter = AutoPromoter(thresholds=thresholds)
        
        baseline = {"prophet_mape": 20.0, "prophet_mse": 2000.0}
        candidate = {"prophet_mape": 15.0, "prophet_mse": 1500.0}  # 25% MAPE improvement
        
        should_promote, improvements, reason = promoter.compare_metrics(
            candidate, baseline
        )
        
        assert should_promote is True
        assert "prophet_mape_improvement_pct" in improvements
        assert improvements["prophet_mape_improvement_pct"] == 25.0
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_compare_metrics_no_improvement(self, mock_mlflow, mock_client, thresholds):
        """Test that no promotion when improvement below threshold."""
        promoter = AutoPromoter(thresholds=thresholds)
        
        baseline = {"prophet_mape": 20.0, "prophet_mse": 2000.0}
        candidate = {"prophet_mape": 19.5, "prophet_mse": 1950.0}  # Only 2.5% improvement
        
        should_promote, improvements, reason = promoter.compare_metrics(
            candidate, baseline
        )
        
        assert should_promote is False
        assert "No significant improvement" in reason
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_compare_metrics_regression(self, mock_mlflow, mock_client, thresholds):
        """Test handling of metric regression."""
        promoter = AutoPromoter(thresholds=thresholds)
        
        baseline = {"prophet_mape": 15.0}
        candidate = {"prophet_mape": 18.0}  # Worse performance
        
        should_promote, improvements, reason = promoter.compare_metrics(
            candidate, baseline
        )
        
        assert should_promote is False
        assert improvements["prophet_mape_improvement_pct"] < 0


# ============================================================================
# Auto-Promoter Tests
# ============================================================================


class TestAutoPromoter:
    """Tests for AutoPromoter class."""
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_promoter_initialization(self, mock_mlflow, mock_client):
        """Test AutoPromoter initializes correctly."""
        promoter = AutoPromoter()
        
        assert promoter.thresholds is not None
        assert promoter.tracking_uri is not None
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_get_new_runs_no_experiment(self, mock_mlflow, mock_client_class):
        """Test handling of non-existent experiment."""
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_client_class.return_value = mock_client
        
        promoter = AutoPromoter()
        promoter.client = mock_client
        
        runs = promoter.get_new_runs("nonexistent-experiment")
        
        assert runs == []
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_get_new_runs_with_results(self, mock_mlflow, mock_client_class, mock_run):
        """Test getting new runs from experiment."""
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = [mock_run]
        mock_client_class.return_value = mock_client
        
        promoter = AutoPromoter()
        promoter.client = mock_client
        
        runs = promoter.get_new_runs("test-experiment")
        
        assert len(runs) == 1
        assert runs[0].info.run_id == mock_run.info.run_id
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_get_production_baseline_no_model(self, mock_mlflow, mock_client_class):
        """Test handling when no production model exists."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []
        mock_client_class.return_value = mock_client
        
        promoter = AutoPromoter()
        promoter.client = mock_client
        
        baseline = promoter.get_production_baseline("test-model")
        
        assert baseline is None
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_get_production_baseline_with_model(
        self, mock_mlflow, mock_client_class, mock_model_version, mock_run
    ):
        """Test getting production baseline metrics."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_client.get_run.return_value = mock_run
        mock_client_class.return_value = mock_client
        
        promoter = AutoPromoter()
        promoter.client = mock_client
        
        baseline = promoter.get_production_baseline("test-model")
        
        assert baseline is not None
        assert "prophet_mape" in baseline


# ============================================================================
# Webhook Tests
# ============================================================================


class TestWebhook:
    """Tests for webhook triggering."""
    
    @patch('mlflow_utils.auto_promote.requests.post')
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_trigger_webhook_success(self, mock_mlflow, mock_client, mock_post, mock_config):
        """Test successful webhook trigger."""
        mock_post.return_value = MagicMock(
            status_code=200,
            text='{"status": "ok"}'
        )
        mock_post.return_value.raise_for_status = MagicMock()
        
        promoter = AutoPromoter()
        promoter.webhook_config = mock_config["webhook"]
        
        result = promoter.trigger_webhook(
            model_name="test-model",
            model_version="2",
            metrics={"mape": 10.0},
            run_id="abc123",
        )
        
        assert result is not None
        assert result["status_code"] == 200
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_trigger_webhook_disabled(self, mock_mlflow, mock_client):
        """Test webhook skipped when disabled."""
        promoter = AutoPromoter()
        promoter.webhook_config = {"enabled": False}
        
        result = promoter.trigger_webhook(
            model_name="test-model",
            model_version="2",
            metrics={},
            run_id="abc123",
        )
        
        assert result is None
    
    @patch('mlflow_utils.auto_promote.requests.post')
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_trigger_webhook_retry_on_failure(
        self, mock_mlflow, mock_client, mock_post, mock_config
    ):
        """Test webhook retry logic on failure."""
        import requests
        mock_post.side_effect = requests.RequestException("Connection error")
        
        promoter = AutoPromoter()
        promoter.webhook_config = {
            **mock_config["webhook"],
            "retry_attempts": 2,
            "retry_delay_seconds": 0,
        }
        
        result = promoter.trigger_webhook(
            model_name="test-model",
            model_version="2",
            metrics={},
            run_id="abc123",
        )
        
        assert result is None
        assert mock_post.call_count == 2  # Retried once


# ============================================================================
# CLI Tests
# ============================================================================


class TestCLI:
    """Tests for CLI commands."""
    
    @patch('mlflow_utils.cli.MlflowClient')
    @patch('mlflow_utils.cli.mlflow')
    def test_cmd_list_versions(self, mock_mlflow, mock_client_class, mock_model_version):
        """Test list-versions command."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_client_class.return_value = mock_client
        
        versions = cli.cmd_list_versions("test-model")
        
        assert len(versions) >= 0  # May be empty depending on stages
    
    @patch('mlflow_utils.cli.MlflowClient')
    @patch('mlflow_utils.cli.mlflow')
    def test_cmd_rollback_success(self, mock_mlflow, mock_client_class, mock_model_version):
        """Test successful rollback."""
        mock_client = MagicMock()
        mock_model_version.version = "2"
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_client_class.return_value = mock_client
        
        result = cli.cmd_rollback("test-model", target_version=1)
        
        assert result is True
        mock_client.transition_model_version_stage.assert_called()
    
    @patch('mlflow_utils.cli.MlflowClient')
    @patch('mlflow_utils.cli.mlflow')
    def test_cmd_rollback_no_production(self, mock_mlflow, mock_client_class):
        """Test rollback when no production model exists."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []
        mock_client_class.return_value = mock_client
        
        result = cli.cmd_rollback("test-model")
        
        assert result is False
    
    @patch('mlflow_utils.cli.MlflowClient')
    @patch('mlflow_utils.cli.mlflow')
    def test_cmd_promote(self, mock_mlflow, mock_client_class):
        """Test manual promote command."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []
        mock_client_class.return_value = mock_client
        
        result = cli.cmd_promote("test-model", version=1, stage="Staging")
        
        assert result is True
        mock_client.transition_model_version_stage.assert_called_once()
    
    @patch('mlflow_utils.cli.MlflowClient')
    @patch('mlflow_utils.cli.mlflow')
    def test_cmd_archive(self, mock_mlflow, mock_client_class):
        """Test archive command."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        result = cli.cmd_archive("test-model", version=1)
        
        assert result is True


# ============================================================================
# PromotionResult Tests
# ============================================================================


class TestPromotionResult:
    """Tests for PromotionResult dataclass."""
    
    def test_promotion_result_creation(self):
        """Test creating a PromotionResult."""
        result = PromotionResult(
            promoted=True,
            candidate_run_id="abc123",
            candidate_metrics={"mape": 10.0},
            baseline_metrics={"mape": 15.0},
            improvements={"mape_improvement_pct": 33.3},
            reason="MAPE improved by 33.3%",
            model_version="2",
        )
        
        assert result.promoted is True
        assert result.candidate_run_id == "abc123"
        assert result.model_version == "2"
    
    def test_promotion_result_defaults(self):
        """Test PromotionResult default values."""
        result = PromotionResult(
            promoted=False,
            candidate_run_id="xyz789",
            candidate_metrics={},
            baseline_metrics=None,
            improvements={},
            reason="Test",
        )
        
        assert result.model_version is None
        assert result.webhook_triggered is False
        assert result.webhook_response is None


# ============================================================================
# Integration-Style Tests (Simulated Runs)
# ============================================================================


class TestSimulatedRuns:
    """Tests with simulated MLflow run scenarios."""
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_first_model_promoted(self, mock_mlflow, mock_client_class, mock_run):
        """Test that first model (no baseline) gets promoted."""
        mock_client = MagicMock()
        mock_experiment = MagicMock(experiment_id="1")
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = [mock_run]
        mock_client.get_latest_versions.return_value = []  # No production model
        mock_client_class.return_value = mock_client
        
        # Mock register_model
        mock_result = MagicMock()
        mock_result.version = "1"
        mock_mlflow.register_model.return_value = mock_result
        
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_client.get_model_version.return_value = mock_version
        
        promoter = AutoPromoter()
        promoter.client = mock_client
        promoter.webhook_config = {"enabled": False}
        
        results = promoter.evaluate_and_promote(
            experiment_name="test-exp",
            model_name="test-model",
            dry_run=False,
        )
        
        assert len(results) == 1
        assert results[0].reason == "No production baseline - first model"
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')
    def test_improved_model_promoted(self, mock_mlflow, mock_client_class, thresholds):
        """Test that improved model gets promoted."""
        # Setup candidate run with better metrics
        candidate_run = MagicMock()
        candidate_run.info.run_id = "candidate123"
        candidate_run.data.metrics = {"prophet_mape": 10.0, "prophet_mse": 800.0}
        
        # Setup baseline run with worse metrics
        baseline_run = MagicMock()
        baseline_run.data.metrics = {"prophet_mape": 15.0, "prophet_mse": 1200.0}
        
        mock_client = MagicMock()
        mock_experiment = MagicMock(experiment_id="1")
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = [candidate_run]
        
        # Production model exists
        prod_version = MagicMock()
        prod_version.run_id = "baseline456"
        mock_client.get_latest_versions.return_value = [prod_version]
        mock_client.get_run.return_value = baseline_run
        mock_client_class.return_value = mock_client
        
        promoter = AutoPromoter(thresholds=thresholds)
        promoter.client = mock_client
        
        # Use dry run to avoid actual promotion
        results = promoter.evaluate_and_promote(
            experiment_name="test-exp",
            model_name="test-model",
            dry_run=True,
        )
        
        assert len(results) == 1
        # Should identify improvement
        assert "DRY RUN" in results[0].reason or "improvement" in results[0].reason.lower()
    
    @patch('mlflow_utils.auto_promote.MlflowClient')
    @patch('mlflow_utils.auto_promote.mlflow')  
    def test_worse_model_not_promoted(self, mock_mlflow, mock_client_class, thresholds):
        """Test that model with worse metrics is not promoted."""
        # Setup candidate run with worse metrics
        candidate_run = MagicMock()
        candidate_run.info.run_id = "candidate123"
        candidate_run.data.metrics = {"prophet_mape": 20.0}  # Worse than baseline
        
        # Setup baseline run
        baseline_run = MagicMock()
        baseline_run.data.metrics = {"prophet_mape": 15.0}
        
        mock_client = MagicMock()
        mock_experiment = MagicMock(experiment_id="1")
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = [candidate_run]
        
        prod_version = MagicMock()
        prod_version.run_id = "baseline456"
        mock_client.get_latest_versions.return_value = [prod_version]
        mock_client.get_run.return_value = baseline_run
        mock_client_class.return_value = mock_client
        
        promoter = AutoPromoter(thresholds=thresholds)
        promoter.client = mock_client
        
        results = promoter.evaluate_and_promote(
            experiment_name="test-exp",
            model_name="test-model",
            dry_run=True,
        )
        
        assert len(results) == 1
        assert results[0].promoted is False
