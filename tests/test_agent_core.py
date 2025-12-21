"""
Unit Tests for Agent Core.

Tests cover:
- ReAct loop execution
- Tool calling and observation
- Confirmation protocol
- Retry policy
- Drift check and retrain workflow
All with mocked LLM and tools.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from agents.agent_core import (
    AgentCore,
    AgentState,
    AgentStep,
    AgentResponse,
    ToolResult,
    MockLLM,
    AutoConfirmHandler,
    CallbackConfirmHandler,
    RetryPolicy,
    create_agent,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_responses():
    """Predefined LLM responses for testing."""
    return [
        """Thought: I need to check the model drift metrics first.
Action: check_drift
Action Input: {"metric_name": "model_drift_score"}""",
        
        """Thought: The drift score is 0.15 which exceeds the threshold of 0.1. I need to trigger retraining.
Action: trigger_retrain
Action Input: {"dag_id": "retrain"}""",
        
        """Thought: Retraining has been triggered. Now I should send a report.
Action: post_report
Action Input: {"title": "Drift Report", "sections": [{"title": "Drift", "value": "0.15"}]}""",
        
        """Thought: All tasks completed.
FINAL_ANSWER: Detected model drift (0.15 > 0.1). Triggered retraining and sent report to Slack.""",
    ]


@pytest.fixture
def mock_prometheus_response():
    """Mock Prometheus query result."""
    return ToolResult(
        success=True,
        data={
            "results": [{"labels": {}, "value": "0.15"}],
            "count": 1,
        },
    )


@pytest.fixture
def mock_airflow_response():
    """Mock Airflow trigger result."""
    return ToolResult(
        success=True,
        data={
            "run_id": "manual__2024-01-15",
            "state": "running",
            "dag_id": "retrain",
        },
        requires_confirmation=True,
    )


@pytest.fixture
def mock_slack_response():
    """Mock Slack post result."""
    return ToolResult(
        success=True,
        data={"mock": True, "channel": "#alerts"},
    )


# ============================================================================
# MockLLM Tests
# ============================================================================


class TestMockLLM:
    """Tests for MockLLM class."""
    
    def test_mock_llm_returns_responses_in_order(self):
        """Test that MockLLM returns responses in order."""
        responses = ["First response", "Second response"]
        llm = MockLLM(responses=responses)
        
        assert llm.generate("prompt 1") == "First response"
        assert llm.generate("prompt 2") == "Second response"
        assert llm.call_count == 2
    
    def test_mock_llm_default_response(self):
        """Test default response when no more responses."""
        llm = MockLLM(responses=["One"])
        
        llm.generate("prompt")
        result = llm.generate("prompt 2")
        
        assert "FINAL_ANSWER" in result
    
    def test_mock_llm_stores_prompts(self):
        """Test that prompts are stored."""
        llm = MockLLM()
        
        llm.generate("Test prompt")
        
        assert "Test prompt" in llm.prompts


# ============================================================================
# Confirmation Handler Tests
# ============================================================================


class TestConfirmationHandlers:
    """Tests for confirmation handlers."""
    
    def test_auto_confirm_always_returns_true(self):
        """Test AutoConfirmHandler always confirms."""
        handler = AutoConfirmHandler()
        
        result = handler.request_confirmation(
            action="trigger_retrain",
            description="Test action",
            details={"key": "value"},
        )
        
        assert result is True
    
    def test_callback_handler_uses_callback(self):
        """Test CallbackConfirmHandler uses the callback."""
        callback_results = []
        
        def my_callback(action, description, details):
            callback_results.append((action, description))
            return action == "approved_action"
        
        handler = CallbackConfirmHandler(my_callback)
        
        result1 = handler.request_confirmation("approved_action", "desc", {})
        result2 = handler.request_confirmation("denied_action", "desc", {})
        
        assert result1 is True
        assert result2 is False
        assert len(callback_results) == 2


# ============================================================================
# Retry Policy Tests
# ============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy class."""
    
    def test_should_retry_with_retryable_error(self):
        """Test retry on retryable errors."""
        policy = RetryPolicy(max_attempts=3)
        
        assert policy.should_retry("Connection timeout", 0) is True
        assert policy.should_retry("Connection timeout", 2) is True
        assert policy.should_retry("Connection timeout", 3) is False
    
    def test_should_not_retry_non_retryable_error(self):
        """Test no retry on non-retryable errors."""
        policy = RetryPolicy()
        
        assert policy.should_retry("Invalid API key", 0) is False
    
    def test_get_delay_exponential_backoff(self):
        """Test exponential backoff calculation."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0)
        
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
    
    def test_get_delay_max_cap(self):
        """Test delay doesn't exceed max."""
        policy = RetryPolicy(base_delay=10.0, max_delay=15.0)
        
        assert policy.get_delay(5) == 15.0
    
    def test_execute_with_retry_success(self):
        """Test successful execution without retry."""
        policy = RetryPolicy()
        
        result = policy.execute_with_retry(lambda: "success")
        
        assert result == "success"
    
    def test_execute_with_retry_recovers(self):
        """Test retry recovers from transient error."""
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        
        call_count = 0
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return ToolResult(success=False, data=None, error="Connection timeout")
            return ToolResult(success=True, data={"value": 1})
        
        result = policy.execute_with_retry(flaky_func)
        
        assert result.success is True
        assert call_count == 2


# ============================================================================
# AgentCore Tests
# ============================================================================


class TestAgentCore:
    """Tests for AgentCore class."""
    
    @pytest.fixture
    def agent(self, mock_llm_responses):
        """Create agent with mocked components."""
        return AgentCore(
            llm=MockLLM(responses=mock_llm_responses),
            confirmation_handler=AutoConfirmHandler(),
            drift_threshold=0.1,
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.state == AgentState.IDLE
        assert len(agent.tools) > 0
        assert "check_drift" in agent.tools
        assert "trigger_retrain" in agent.tools
    
    def test_agent_tools_registered(self, agent):
        """Test all expected tools are registered."""
        expected_tools = [
            "query_metrics",
            "get_alerts",
            "check_drift",
            "trigger_retrain",
            "get_dag_status",
            "send_slack_message",
            "post_report",
        ]
        
        for tool in expected_tools:
            assert tool in agent.tools
    
    def test_trigger_retrain_requires_confirmation(self, agent):
        """Test that trigger_retrain requires confirmation."""
        assert agent.tools["trigger_retrain"].requires_confirmation is True
        assert agent.tools["check_drift"].requires_confirmation is False


class TestAgentExecution:
    """Tests for agent execution with mocked tools."""
    
    @pytest.fixture
    def mock_tools(self, mock_prometheus_response, mock_airflow_response, mock_slack_response):
        """Create mocked tool instances."""
        prometheus = MagicMock()
        prometheus.query.return_value = mock_prometheus_response
        
        airflow = MagicMock()
        airflow.trigger_dag.return_value = mock_airflow_response
        airflow.get_dag_status.return_value = ToolResult(
            success=True,
            data={"state": "running"},
        )
        
        slack = MagicMock()
        slack.send_message.return_value = mock_slack_response
        slack.post_report.return_value = mock_slack_response
        
        return {"prometheus": prometheus, "airflow": airflow, "slack": slack}
    
    @patch('agents.agent_core.PrometheusTool')
    @patch('agents.agent_core.AirflowTool')
    @patch('agents.agent_core.SlackTool')
    def test_drift_check_workflow(
        self, 
        mock_slack_cls, 
        mock_airflow_cls, 
        mock_prometheus_cls,
        mock_tools,
    ):
        """Test complete drift check and retrain workflow."""
        mock_prometheus_cls.return_value = mock_tools["prometheus"]
        mock_airflow_cls.return_value = mock_tools["airflow"]
        mock_slack_cls.return_value = mock_tools["slack"]
        
        responses = [
            """Thought: Checking drift.
Action: check_drift
Action Input: {}""",
            """Thought: Drift detected, triggering retrain.
Action: trigger_retrain
Action Input: {"dag_id": "retrain"}""",
            """Thought: Sending report.
Action: post_report
Action Input: {"title": "Report", "sections": []}""",
            """FINAL_ANSWER: Complete.""",
        ]
        
        agent = AgentCore(
            llm=MockLLM(responses=responses),
            confirmation_handler=AutoConfirmHandler(),
        )
        
        # Mock the internal tools
        agent.prometheus = mock_tools["prometheus"]
        agent.airflow = mock_tools["airflow"]
        agent.slack = mock_tools["slack"]
        
        result = agent.run("Check drift and retrain")
        
        assert result.success is True
        assert len(result.steps) > 0
    
    def test_agent_respects_max_iterations(self):
        """Test agent stops at max iterations."""
        # LLM that never finishes
        infinite_llm = MockLLM(responses=[
            "Thought: Thinking...\nAction: check_drift\nAction Input: {}"
        ] * 20)
        
        agent = AgentCore(
            llm=infinite_llm,
            confirmation_handler=AutoConfirmHandler(),
        )
        
        result = agent.run("Test objective")
        
        assert result.success is False
        assert "Max iterations" in result.error


class TestCheckDriftAndRetrain:
    """Tests for the convenience check_drift_and_retrain method."""
    
    @patch('agents.agent_core.PrometheusTool')
    @patch('agents.agent_core.AirflowTool')
    @patch('agents.agent_core.SlackTool')
    def test_no_drift_no_retrain(
        self,
        mock_slack_cls,
        mock_airflow_cls,
        mock_prometheus_cls,
    ):
        """Test no retraining when drift is below threshold."""
        mock_prometheus = MagicMock()
        mock_prometheus.query.return_value = ToolResult(
            success=True,
            data={"results": [{"value": "0.05"}], "count": 1},
        )
        mock_prometheus_cls.return_value = mock_prometheus
        
        mock_slack = MagicMock()
        mock_slack.post_report.return_value = ToolResult(success=True, data={})
        mock_slack_cls.return_value = mock_slack
        
        agent = AgentCore(
            llm=MockLLM(),
            confirmation_handler=AutoConfirmHandler(),
            drift_threshold=0.1,
        )
        agent.prometheus = mock_prometheus
        agent.slack = mock_slack
        
        result = agent.check_drift_and_retrain()
        
        assert result.success is True
        assert "No significant drift" in result.final_answer
    
    @patch('agents.agent_core.PrometheusTool')
    @patch('agents.agent_core.AirflowTool')
    @patch('agents.agent_core.SlackTool')
    def test_drift_triggers_retrain_with_confirmation(
        self,
        mock_slack_cls,
        mock_airflow_cls,
        mock_prometheus_cls,
    ):
        """Test retraining triggered when drift exceeds threshold."""
        # Setup mocks
        mock_prometheus = MagicMock()
        mock_prometheus.query.return_value = ToolResult(
            success=True,
            data={"results": [{"value": "0.15"}], "count": 1},
        )
        mock_prometheus_cls.return_value = mock_prometheus
        
        mock_airflow = MagicMock()
        mock_airflow.trigger_dag.return_value = ToolResult(
            success=True,
            data={"run_id": "test_run", "state": "running"},
        )
        mock_airflow_cls.return_value = mock_airflow
        
        mock_slack = MagicMock()
        mock_slack.post_report.return_value = ToolResult(success=True, data={})
        mock_slack_cls.return_value = mock_slack
        
        agent = AgentCore(
            llm=MockLLM(),
            confirmation_handler=AutoConfirmHandler(),
            drift_threshold=0.1,
        )
        agent.prometheus = mock_prometheus
        agent.airflow = mock_airflow
        agent.slack = mock_slack
        
        result = agent.check_drift_and_retrain()
        
        assert result.success is True
        assert "Retraining triggered" in result.final_answer
        mock_airflow.trigger_dag.assert_called()
    
    @patch('agents.agent_core.PrometheusTool')
    @patch('agents.agent_core.AirflowTool')
    @patch('agents.agent_core.SlackTool')
    def test_drift_retrain_cancelled_when_not_confirmed(
        self,
        mock_slack_cls,
        mock_airflow_cls,
        mock_prometheus_cls,
    ):
        """Test retraining cancelled when user denies confirmation."""
        mock_prometheus = MagicMock()
        mock_prometheus.query.return_value = ToolResult(
            success=True,
            data={"results": [{"value": "0.15"}], "count": 1},
        )
        mock_prometheus_cls.return_value = mock_prometheus
        
        mock_airflow = MagicMock()
        mock_airflow_cls.return_value = mock_airflow
        
        mock_slack = MagicMock()
        mock_slack.post_report.return_value = ToolResult(success=True, data={})
        mock_slack_cls.return_value = mock_slack
        
        # Use callback handler that denies
        deny_handler = CallbackConfirmHandler(lambda *args: False)
        
        agent = AgentCore(
            llm=MockLLM(),
            confirmation_handler=deny_handler,
            drift_threshold=0.1,
        )
        agent.prometheus = mock_prometheus
        agent.airflow = mock_airflow
        agent.slack = mock_slack
        
        result = agent.check_drift_and_retrain()
        
        assert result.success is True
        assert "cancelled by user" in result.final_answer
        mock_airflow.trigger_dag.assert_not_called()


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateAgent:
    """Tests for create_agent factory function."""
    
    def test_create_mock_agent(self):
        """Test creating agent with mock LLM."""
        agent = create_agent(llm_type="mock", confirmation_mode="auto")
        
        assert isinstance(agent.llm, MockLLM)
        assert isinstance(agent.confirmation_handler, AutoConfirmHandler)
    
    def test_create_agent_with_callback(self):
        """Test creating agent with callback confirmation."""
        callback = lambda *args: True
        
        agent = create_agent(
            llm_type="mock",
            confirmation_mode="callback",
            confirm_callback=callback,
        )
        
        assert isinstance(agent.confirmation_handler, CallbackConfirmHandler)
    
    def test_create_agent_with_custom_threshold(self):
        """Test creating agent with custom drift threshold."""
        agent = create_agent(drift_threshold=0.05)
        
        assert agent.drift_threshold == 0.05


# ============================================================================
# Response Parsing Tests
# ============================================================================


class TestResponseParsing:
    """Tests for LLM response parsing."""
    
    @pytest.fixture
    def agent(self):
        return AgentCore(llm=MockLLM())
    
    def test_parse_thought_action(self, agent):
        """Test parsing thought and action."""
        response = """Thought: I need to check metrics.
Action: check_drift
Action Input: {"metric_name": "drift"}"""
        
        parsed = agent._parse_response(response)
        
        assert "check metrics" in parsed["thought"]
        assert parsed["action"] == "check_drift"
        assert parsed["action_input"]["metric_name"] == "drift"
    
    def test_parse_final_answer(self, agent):
        """Test parsing final answer."""
        response = """Thought: All done.
FINAL_ANSWER: Task completed successfully."""
        
        parsed = agent._parse_response(response)
        
        assert parsed["final_answer"] == "Task completed successfully."
    
    def test_parse_empty_action_input(self, agent):
        """Test parsing with empty action input."""
        response = """Thought: Checking.
Action: get_alerts
Action Input: {}"""
        
        parsed = agent._parse_response(response)
        
        assert parsed["action"] == "get_alerts"
        assert parsed["action_input"] == {}


# ============================================================================
# Slack Tool Tests
# ============================================================================


class TestSlackTool:
    """Tests for SlackTool."""
    
    def test_mock_mode_logs_message(self):
        """Test Slack tool in mock mode."""
        from agents.tools.tool_slack import SlackTool
        
        tool = SlackTool()  # No webhook = mock mode
        result = tool.send_message("Test message")
        
        assert result.success is True
        assert result.data["mock"] is True
    
    @patch('agents.tools.tool_slack.requests.post')
    def test_webhook_mode(self, mock_post):
        """Test Slack tool with webhook."""
        from agents.tools.tool_slack import SlackTool
        
        mock_post.return_value = MagicMock(status_code=200)
        
        tool = SlackTool(webhook_url="https://hooks.slack.com/test")
        result = tool.send_message("Test message")
        
        assert result.success is True
        mock_post.assert_called_once()
    
    def test_post_report_format(self):
        """Test report formatting."""
        from agents.tools.tool_slack import SlackTool
        
        tool = SlackTool()
        result = tool.post_report(
            title="Test Report",
            sections=[
                {"title": "Metric", "value": "0.15"},
                {"title": "Status", "value": "OK"},
            ],
        )
        
        assert result.success is True
        assert "fields" in str(result.data["payload"]["attachments"])
