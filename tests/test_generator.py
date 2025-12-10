"""Unit tests for the sample event generator."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Add data directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from sample_generator import (
    SCHEMA_FIELDS,
    generate_event,
    generate_events,
    validate_event,
)


class TestSchemaConformance:
    """Tests for schema conformance of generated events."""

    def test_single_event_has_required_fields(self) -> None:
        """Test that a single generated event has all required fields."""
        from faker import Faker
        from datetime import datetime, timezone
        
        fake = Faker()
        Faker.seed(42)
        event = generate_event(fake, datetime.now(timezone.utc))
        
        required_fields = [
            field for field, spec in SCHEMA_FIELDS.items() 
            if spec["required"]
        ]
        
        for field in required_fields:
            assert field in event, f"Missing required field: {field}"

    def test_single_event_field_types(self) -> None:
        """Test that a single event has correct field types."""
        from faker import Faker
        from datetime import datetime, timezone
        
        fake = Faker()
        Faker.seed(42)
        event = generate_event(fake, datetime.now(timezone.utc))
        
        for field, spec in SCHEMA_FIELDS.items():
            if field in event:
                assert isinstance(event[field], spec["type"]), (
                    f"Field {field} has wrong type: "
                    f"expected {spec['type']}, got {type(event[field])}"
                )

    def test_validate_event_returns_true_for_valid_event(self) -> None:
        """Test that validate_event returns True for a valid event."""
        from faker import Faker
        from datetime import datetime, timezone
        
        fake = Faker()
        event = generate_event(fake, datetime.now(timezone.utc))
        
        assert validate_event(event) is True

    def test_validate_event_returns_false_for_missing_required_field(self) -> None:
        """Test that validate_event returns False when required field is missing."""
        invalid_event = {
            "id": "test-id",
            # Missing other required fields
        }
        
        assert validate_event(invalid_event) is False

    def test_validate_event_returns_false_for_wrong_type(self) -> None:
        """Test that validate_event returns False for wrong field type."""
        invalid_event = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00Z",
            "user_id": "user_123",
            "product_id": "prod_456",
            "price": "not a number",  # Should be float
            "quantity": 1,
            "location": "Test City, Test Country",
        }
        
        assert validate_event(invalid_event) is False


class TestEventGeneration:
    """Tests for bulk event generation."""

    def test_generate_events_returns_correct_count(self) -> None:
        """Test that generate_events returns the requested number of events."""
        count = 100
        events = generate_events(count, seed=42)
        
        assert len(events) == count

    def test_generate_events_with_seed_is_reproducible(self) -> None:
        """Test that the same seed produces identical events."""
        events1 = generate_events(10, seed=42)
        events2 = generate_events(10, seed=42)
        
        assert events1 == events2

    def test_generate_events_different_seeds_produce_different_events(self) -> None:
        """Test that different seeds produce different events."""
        events1 = generate_events(10, seed=42)
        events2 = generate_events(10, seed=43)
        
        assert events1 != events2

    def test_generate_events_all_validate(self) -> None:
        """Test that all generated events pass validation."""
        events = generate_events(100, seed=42)
        
        for i, event in enumerate(events):
            assert validate_event(event), f"Event {i} failed validation"

    def test_generated_events_are_sorted_by_timestamp(self) -> None:
        """Test that generated events are sorted by timestamp."""
        events = generate_events(100, seed=42)
        
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)


class TestEventFields:
    """Tests for individual event field values."""

    def test_id_is_valid_uuid(self) -> None:
        """Test that event IDs are valid UUIDs."""
        import uuid
        
        events = generate_events(10, seed=42)
        
        for event in events:
            # This will raise ValueError if not a valid UUID
            uuid.UUID(event["id"])

    def test_timestamp_is_iso_format(self) -> None:
        """Test that timestamps are valid ISO format."""
        from datetime import datetime
        
        events = generate_events(10, seed=42)
        
        for event in events:
            # This will raise ValueError if not valid ISO format
            datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))

    def test_price_is_positive(self) -> None:
        """Test that all prices are positive."""
        events = generate_events(100, seed=42)
        
        for event in events:
            assert event["price"] > 0, f"Price should be positive: {event['price']}"

    def test_quantity_is_positive_integer(self) -> None:
        """Test that all quantities are positive integers."""
        events = generate_events(100, seed=42)
        
        for event in events:
            assert event["quantity"] >= 1, f"Quantity should be >= 1: {event['quantity']}"
            assert isinstance(event["quantity"], int)

    def test_metadata_has_channel(self) -> None:
        """Test that metadata always contains channel."""
        events = generate_events(100, seed=42)
        
        for event in events:
            assert "metadata" in event
            assert "channel" in event["metadata"]
            assert event["metadata"]["channel"] in ["web", "mobile", "api", "pos"]

    def test_user_id_format(self) -> None:
        """Test that user_id has expected prefix format."""
        events = generate_events(10, seed=42)
        
        for event in events:
            assert event["user_id"].startswith("usr_")

    def test_product_id_format(self) -> None:
        """Test that product_id has expected prefix format."""
        events = generate_events(10, seed=42)
        
        for event in events:
            assert event["product_id"].startswith("prod_")


class TestMetadataFields:
    """Tests for metadata field values."""

    def test_metadata_is_json_serializable(self) -> None:
        """Test that metadata can be serialized to JSON."""
        events = generate_events(100, seed=42)
        
        for event in events:
            # Should not raise
            json.dumps(event["metadata"])

    def test_session_id_format(self) -> None:
        """Test that session_id has expected format."""
        events = generate_events(100, seed=42)
        
        for event in events:
            assert event["metadata"]["session_id"].startswith("sess_")

    def test_device_type_for_pos_channel(self) -> None:
        """Test that POS channel events have pos_terminal device type."""
        events = generate_events(1000, seed=42)  # Generate more to ensure we get POS events
        
        pos_events = [e for e in events if e["metadata"]["channel"] == "pos"]
        
        for event in pos_events:
            assert event["metadata"]["device_type"] == "pos_terminal"
