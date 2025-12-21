"""
Chaos Testing Utilities.

Provides utilities for chaos engineering tests.
"""

import os
import time
import subprocess
import logging
from typing import Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ChaosController:
    """Controller for chaos engineering operations."""
    
    def __init__(self, compose_file: str = "docker-compose.e2e.yml"):
        self.compose_file = compose_file
        self.compose_dir = os.path.dirname(os.path.abspath(__file__))
    
    def _run_compose(self, *args) -> subprocess.CompletedProcess:
        """Run docker-compose command."""
        cmd = ["docker-compose", "-f", self.compose_file] + list(args)
        return subprocess.run(
            cmd,
            cwd=os.path.join(self.compose_dir, ".."),
            capture_output=True,
            text=True,
        )
    
    def stop_service(self, service: str) -> bool:
        """Stop a service."""
        result = self._run_compose("stop", service)
        logger.info(f"Stopped {service}: {result.returncode == 0}")
        return result.returncode == 0
    
    def start_service(self, service: str) -> bool:
        """Start a service."""
        result = self._run_compose("start", service)
        logger.info(f"Started {service}: {result.returncode == 0}")
        return result.returncode == 0
    
    def restart_service(self, service: str) -> bool:
        """Restart a service."""
        result = self._run_compose("restart", service)
        logger.info(f"Restarted {service}: {result.returncode == 0}")
        return result.returncode == 0
    
    def pause_service(self, service: str) -> bool:
        """Pause a service (freeze)."""
        result = subprocess.run(
            ["docker", "pause", f"e2e-{service}"],
            capture_output=True,
        )
        return result.returncode == 0
    
    def unpause_service(self, service: str) -> bool:
        """Unpause a service."""
        result = subprocess.run(
            ["docker", "unpause", f"e2e-{service}"],
            capture_output=True,
        )
        return result.returncode == 0
    
    def kill_service(self, service: str) -> bool:
        """Kill a service abruptly."""
        result = subprocess.run(
            ["docker", "kill", f"e2e-{service}"],
            capture_output=True,
        )
        return result.returncode == 0
    
    def is_service_running(self, service: str) -> bool:
        """Check if service is running."""
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", f"e2e-{service}"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == "true"


@contextmanager
def service_down(chaos: ChaosController, service: str, duration: float = 5.0):
    """
    Context manager to temporarily bring down a service.
    
    Usage:
        with service_down(chaos, "kafka", duration=10):
            # Test behavior during outage
            pass
        # Service is restored after context exits
    """
    try:
        chaos.stop_service(service)
        time.sleep(duration)
        yield
    finally:
        chaos.start_service(service)
        time.sleep(5)  # Wait for service to be ready


@contextmanager
def network_partition(chaos: ChaosController, service: str, duration: float = 5.0):
    """
    Simulate network partition by pausing container.
    
    Usage:
        with network_partition(chaos, "kafka", duration=5):
            # Test behavior during network partition
            pass
    """
    try:
        chaos.pause_service(service)
        time.sleep(duration)
        yield
    finally:
        chaos.unpause_service(service)


class RetryTracker:
    """Track retry attempts for validation."""
    
    def __init__(self):
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.start_time = None
        self.end_time = None
    
    def record_attempt(self, success: bool):
        if self.start_time is None:
            self.start_time = time.time()
        
        self.attempts += 1
        if success:
            self.successes += 1
            self.end_time = time.time()
        else:
            self.failures += 1
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0
        return self.successes / self.attempts


def with_retries(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    tracker: Optional[RetryTracker] = None,
) -> tuple:
    """
    Execute function with exponential backoff retry.
    
    Returns:
        (success: bool, result: any, tracker: RetryTracker)
    """
    if tracker is None:
        tracker = RetryTracker()
    
    current_delay = delay
    
    for attempt in range(max_retries):
        try:
            result = func()
            tracker.record_attempt(success=True)
            return True, result, tracker
        except Exception as e:
            tracker.record_attempt(success=False)
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(current_delay)
                current_delay *= backoff
    
    return False, None, tracker
