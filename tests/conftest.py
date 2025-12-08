"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_event():
    """Sample event for testing."""
    return {
        "event_id": "evt_123",
        "timestamp": "2024-01-01T00:00:00Z",
        "user_id": "user_456",
        "action": "transaction",
        "metadata": {"amount": 100.0, "currency": "USD"},
    }


@pytest.fixture
def sample_features():
    """Sample features for testing."""
    return {
        "user_id": "user_456",
        "total_spend": 1500.0,
        "transaction_count": 25,
        "avg_transaction": 60.0,
        "last_active_days": 3,
    }


@pytest.fixture
def mock_model_response():
    """Mock model prediction response."""
    return {
        "prediction": "APPROVE",
        "confidence": 0.95,
        "model_version": "1.0.0",
    }
