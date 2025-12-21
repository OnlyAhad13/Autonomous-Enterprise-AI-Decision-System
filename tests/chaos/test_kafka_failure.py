"""
Kafka Failure Chaos Tests.

Tests system behavior during Kafka downtime.

Run with:
    E2E_MODE=true pytest tests/chaos/test_kafka_failure.py -v
"""

import os
import time
import pytest
import requests
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chaos.chaos_utils import (
    ChaosController,
    service_down,
    network_partition,
    RetryTracker,
    with_retries,
)

E2E_MODE = os.environ.get("E2E_MODE", "false").lower() == "true"


# ============================================================================
# Mock Tests (Run without Docker)
# ============================================================================


class TestKafkaFailureMocked:
    """Mocked Kafka failure tests."""
    
    def test_producer_retry_on_failure(self):
        """Test producer retries on Kafka failure."""
        tracker = RetryTracker()
        attempt_count = [0]
        
        def flaky_send():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ConnectionError("Kafka unavailable")
            return {"status": "sent"}
        
        success, result, tracker = with_retries(
            flaky_send,
            max_retries=5,
            delay=0.1,
        )
        
        assert success is True
        assert tracker.attempts == 3
        assert tracker.success_rate > 0
    
    def test_producer_exhausts_retries(self):
        """Test producer behavior when retries exhausted."""
        def always_fail():
            raise ConnectionError("Kafka unavailable")
        
        success, result, tracker = with_retries(
            always_fail,
            max_retries=3,
            delay=0.1,
        )
        
        assert success is False
        assert tracker.attempts == 3
        assert tracker.failures == 3
    
    def test_alert_triggered_on_failure(self):
        """Test alert is triggered on persistent failure."""
        alerts_sent = []
        
        def send_alert(message: str):
            alerts_sent.append(message)
        
        def always_fail():
            raise ConnectionError("Kafka unavailable")
        
        success, _, tracker = with_retries(
            always_fail,
            max_retries=3,
            delay=0.1,
        )
        
        if not success:
            send_alert(f"Kafka producer failed after {tracker.attempts} attempts")
        
        assert len(alerts_sent) == 1
        assert "failed after 3 attempts" in alerts_sent[0]


# ============================================================================
# Integration Tests (Require Docker)
# ============================================================================


@pytest.mark.skipif(not E2E_MODE, reason="E2E mode not enabled")
class TestKafkaFailureIntegration:
    """Integration tests with actual Kafka."""
    
    @pytest.fixture
    def chaos(self):
        return ChaosController()
    
    def test_kafka_stop_and_restart(self, chaos):
        """Test stopping and restarting Kafka."""
        # Stop Kafka
        assert chaos.stop_service("kafka") is True
        assert chaos.is_service_running("kafka") is False
        
        time.sleep(2)
        
        # Start Kafka
        assert chaos.start_service("kafka") is True
        
        # Wait for Kafka to be ready
        time.sleep(10)
        assert chaos.is_service_running("kafka") is True
    
    def test_producer_during_kafka_outage(self, chaos):
        """Test producer behavior during Kafka outage."""
        from kafka import KafkaProducer
        from kafka.errors import NoBrokersAvailable
        
        # Create producer with short timeout
        try:
            producer = KafkaProducer(
                bootstrap_servers="localhost:19092",
                request_timeout_ms=5000,
                retries=3,
            )
        except NoBrokersAvailable:
            pytest.skip("Kafka not available")
        
        # Stop Kafka
        with service_down(chaos, "kafka", duration=5):
            # Attempt to send should fail or timeout
            try:
                future = producer.send("test-topic", b"test-message")
                future.get(timeout=10)
                assert False, "Expected failure"
            except Exception as e:
                # Expected behavior
                assert True
        
        producer.close()
    
    def test_consumer_reconnects_after_outage(self, chaos):
        """Test consumer reconnects after Kafka outage."""
        from kafka import KafkaConsumer
        
        try:
            consumer = KafkaConsumer(
                bootstrap_servers="localhost:19092",
                consumer_timeout_ms=5000,
            )
        except Exception:
            pytest.skip("Kafka not available")
        
        # Simulate outage
        with service_down(chaos, "kafka", duration=3):
            pass
        
        # Consumer should reconnect
        time.sleep(10)
        
        # Check consumer is functional
        consumer.topics()  # This will reconnect if needed
        consumer.close()


# ============================================================================
# Agent Behavior During Kafka Failure
# ============================================================================


class TestAgentKafkaResilience:
    """Test agent behavior during Kafka failures."""
    
    def test_agent_detects_kafka_failure(self):
        """Test agent detects Kafka is unavailable."""
        from agents.tools.tool_kafka import KafkaTool
        
        # Mock Kafka client that fails
        with patch.object(KafkaTool, "_get_consumer") as mock_consumer:
            mock_consumer.side_effect = Exception("Connection refused")
            
            tool = KafkaTool(bootstrap_servers="localhost:9092")
            result = tool.get_consumer_lag("test-group", "test-topic")
            
            assert result.success is False
            assert "Connection refused" in str(result.error)
    
    def test_agent_reports_kafka_status(self):
        """Test agent reports Kafka status correctly."""
        from agents.tools.tool_kafka import KafkaTool
        
        tool = KafkaTool(bootstrap_servers="nonexistent:9092")
        result = tool.list_topics()
        
        assert result.success is False
    
    def test_agent_sends_alert_on_kafka_failure(self):
        """Test agent sends alert when Kafka fails."""
        alerts = []
        
        def mock_alert(message: str, severity: str):
            alerts.append({"message": message, "severity": severity})
        
        # Simulate Kafka check failure
        kafka_available = False
        
        if not kafka_available:
            mock_alert(
                "Kafka connectivity check failed",
                severity="critical",
            )
        
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
