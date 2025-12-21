"""
Agent Retry Behavior Chaos Tests.

Tests agent retry and alerting behavior during failures.

Run with:
    pytest tests/chaos/test_agent_retry.py -v
"""

import os
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chaos.chaos_utils import RetryTracker, with_retries


# ============================================================================
# Agent Retry Policy Tests
# ============================================================================


class TestAgentRetryPolicy:
    """Test agent retry policy behavior."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff between retries."""
        delays = []
        attempt_times = []
        
        def failing_operation():
            attempt_times.append(time.time())
            raise ConnectionError("Service unavailable")
        
        with_retries(
            failing_operation,
            max_retries=4,
            delay=0.1,
            backoff=2.0,
        )
        
        # Calculate actual delays
        for i in range(1, len(attempt_times)):
            delays.append(attempt_times[i] - attempt_times[i-1])
        
        # Verify exponential increase (with tolerance)
        if len(delays) >= 2:
            assert delays[1] > delays[0] * 1.5  # Should roughly double
    
    def test_retry_with_jitter(self):
        """Test retry with jitter to prevent thundering herd."""
        import random
        
        base_delay = 1.0
        jitter = 0.2
        
        delays = []
        for _ in range(10):
            actual_delay = base_delay + random.uniform(-jitter, jitter)
            delays.append(actual_delay)
        
        # Verify jitter adds variation
        assert min(delays) < base_delay
        assert max(delays) > base_delay
    
    def test_max_retries_enforced(self):
        """Test that max retries is enforced."""
        attempts = [0]
        
        def counting_failure():
            attempts[0] += 1
            raise Exception("Always fails")
        
        max_retries = 5
        success, _, tracker = with_retries(
            counting_failure,
            max_retries=max_retries,
            delay=0.01,
        )
        
        assert success is False
        assert attempts[0] == max_retries
    
    def test_success_on_nth_attempt(self):
        """Test success on nth retry attempt."""
        for success_on in [1, 2, 3, 4]:
            attempts = [0]
            
            def sometimes_fails():
                attempts[0] += 1
                if attempts[0] < success_on:
                    raise Exception("Not yet")
                return "success"
            
            success, result, tracker = with_retries(
                sometimes_fails,
                max_retries=5,
                delay=0.01,
            )
            
            assert success is True
            assert result == "success"
            assert tracker.attempts == success_on


# ============================================================================
# Agent Alert Behavior Tests
# ============================================================================


class TestAgentAlertBehavior:
    """Test agent alerting behavior during failures."""
    
    @pytest.fixture
    def alert_collector(self):
        """Collect alerts for testing."""
        alerts: List[Dict] = []
        
        def collect_alert(message: str, severity: str, details: dict = None):
            alerts.append({
                "message": message,
                "severity": severity,
                "details": details or {},
                "timestamp": time.time(),
            })
        
        return alerts, collect_alert
    
    def test_alert_on_max_retries_exhausted(self, alert_collector):
        """Test alert is sent when max retries exhausted."""
        alerts, send_alert = alert_collector
        
        def failing_operation():
            raise Exception("Service unavailable")
        
        success, _, tracker = with_retries(
            failing_operation,
            max_retries=3,
            delay=0.01,
        )
        
        if not success:
            send_alert(
                message="Operation failed after max retries",
                severity="critical",
                details={
                    "attempts": tracker.attempts,
                    "operation": "kafka_produce",
                },
            )
        
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert alerts[0]["details"]["attempts"] == 3
    
    def test_alert_escalation(self, alert_collector):
        """Test alert severity escalation."""
        alerts, send_alert = alert_collector
        
        failure_count = 0
        
        for _ in range(5):
            failure_count += 1
            
            if failure_count >= 5:
                severity = "critical"
            elif failure_count >= 3:
                severity = "warning"
            else:
                severity = "info"
            
            send_alert(
                message=f"Failure {failure_count}",
                severity=severity,
            )
        
        # Verify escalation
        severities = [a["severity"] for a in alerts]
        assert severities[0] == "info"
        assert severities[2] == "warning"
        assert severities[4] == "critical"
    
    def test_alert_deduplication(self, alert_collector):
        """Test alert deduplication."""
        alerts, send_alert = alert_collector
        seen_alerts = set()
        
        def deduplicated_alert(message: str, severity: str, **kwargs):
            key = f"{message}:{severity}"
            if key not in seen_alerts:
                seen_alerts.add(key)
                send_alert(message, severity, **kwargs)
        
        # Send same alert multiple times
        for _ in range(10):
            deduplicated_alert("Service down", "critical")
        
        assert len(alerts) == 1
    
    def test_alert_rate_limiting(self, alert_collector):
        """Test alert rate limiting."""
        alerts, send_alert = alert_collector
        last_alert_time = [0]
        min_interval = 0.1  # 100ms minimum between alerts
        
        def rate_limited_alert(message: str, severity: str):
            now = time.time()
            if now - last_alert_time[0] >= min_interval:
                send_alert(message, severity)
                last_alert_time[0] = now
        
        # Rapid fire alerts
        for _ in range(20):
            rate_limited_alert("High frequency event", "warning")
            time.sleep(0.02)  # 20ms between attempts
        
        # Should be rate limited
        assert len(alerts) < 10


# ============================================================================
# Agent Recovery Tests
# ============================================================================


class TestAgentRecovery:
    """Test agent recovery after failures."""
    
    def test_graceful_degradation(self):
        """Test agent degrades gracefully when service unavailable."""
        primary_available = False
        fallback_result = "cached_value"
        
        def get_with_fallback():
            if primary_available:
                return "live_value"
            return fallback_result
        
        result = get_with_fallback()
        
        assert result == fallback_result
    
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker prevents repeated failures."""
        class CircuitBreaker:
            def __init__(self, threshold: int = 5, reset_timeout: float = 30):
                self.failure_count = 0
                self.threshold = threshold
                self.reset_timeout = reset_timeout
                self.last_failure = 0
                self.state = "closed"
            
            def call(self, func):
                # Check if circuit is open
                if self.state == "open":
                    if time.time() - self.last_failure < self.reset_timeout:
                        raise Exception("Circuit is open")
                    self.state = "half-open"
                
                try:
                    result = func()
                    self.failure_count = 0
                    self.state = "closed"
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure = time.time()
                    
                    if self.failure_count >= self.threshold:
                        self.state = "open"
                    
                    raise
        
        breaker = CircuitBreaker(threshold=3)
        
        def failing_service():
            raise Exception("Service down")
        
        # Trigger failures until circuit opens
        for _ in range(3):
            try:
                breaker.call(failing_service)
            except:
                pass
        
        assert breaker.state == "open"
        
        # Circuit should prevent calls
        with pytest.raises(Exception, match="Circuit is open"):
            breaker.call(failing_service)
    
    def test_health_check_recovery(self):
        """Test recovery via health checks."""
        service_healthy = False
        recovery_actions = []
        
        def check_health():
            return service_healthy
        
        def recover_service():
            recovery_actions.append("restart")
            return True
        
        # Check health and recover
        if not check_health():
            recover_service()
        
        assert len(recovery_actions) == 1
        assert recovery_actions[0] == "restart"
