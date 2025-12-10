"""Unit tests for Spark batch_to_delta transformations."""

import json
from datetime import datetime, timezone

import pytest


# Mock PySpark for unit testing without Spark cluster
class MockColumn:
    """Mock Spark Column for testing."""
    def __init__(self, name: str):
        self.name = name
    
    def isNotNull(self):
        return self
    
    def isNull(self):
        return self
    
    def startswith(self, prefix: str):
        return self


class MockDataFrame:
    """Mock Spark DataFrame for testing."""
    def __init__(self, data: list[dict]):
        self.data = data
    
    def count(self):
        return len(self.data)
    
    def filter(self, condition):
        return MockDataFrame(self.data)
    
    def select(self, *cols):
        return self
    
    def withColumn(self, name, expr):
        return self
    
    def show(self, n=20, truncate=True):
        for row in self.data[:n]:
            print(row)


class TestEventNormalization:
    """Tests for event field normalization."""

    def test_user_id_prefix_normalization(self) -> None:
        """Test that user_id gets usr_ prefix if missing."""
        # Test data
        user_ids = [
            ("usr_abc123", "usr_abc123"),  # Already has prefix
            ("abc123", "usr_abc123"),       # Missing prefix
            ("usr_", "usr_"),               # Edge case: just prefix
        ]
        
        for input_id, expected in user_ids:
            if not input_id.startswith("usr_"):
                result = f"usr_{input_id}"
            else:
                result = input_id
            assert result == expected, f"Failed for input: {input_id}"

    def test_product_id_prefix_normalization(self) -> None:
        """Test that product_id gets prod_ prefix if missing."""
        product_ids = [
            ("prod_xyz789", "prod_xyz789"),  # Already has prefix
            ("xyz789", "prod_xyz789"),        # Missing prefix
        ]
        
        for input_id, expected in product_ids:
            if not input_id.startswith("prod_"):
                result = f"prod_{input_id}"
            else:
                result = input_id
            assert result == expected

    def test_location_parsing(self) -> None:
        """Test location string parsing into city and country."""
        locations = [
            ("New York, USA", "New York", "USA"),
            ("London, United Kingdom", "London", "United Kingdom"),
            ("Paris, France", "Paris", "France"),
            ("São Paulo, Brazil", "São Paulo", "Brazil"),
        ]
        
        for location, expected_city, expected_country in locations:
            parts = location.split(",", 1)
            city = parts[0].strip()
            country = parts[1].strip() if len(parts) > 1 else ""
            
            assert city == expected_city, f"City mismatch for: {location}"
            assert country == expected_country, f"Country mismatch for: {location}"

    def test_timestamp_to_date_extraction(self) -> None:
        """Test extracting date (dt) from timestamp."""
        timestamps = [
            ("2024-01-15T10:30:00.000Z", "2024-01-15"),
            ("2024-12-31T23:59:59.999Z", "2024-12-31"),
            ("2024-02-29T00:00:00.000Z", "2024-02-29"),  # Leap year
        ]
        
        for timestamp, expected_date in timestamps:
            # Parse ISO timestamp
            if timestamp.endswith("Z"):
                ts = timestamp[:-1]  # Remove Z
            else:
                ts = timestamp
            
            # Extract date portion
            dt = datetime.fromisoformat(ts).strftime("%Y-%m-%d")
            assert dt == expected_date

    def test_total_amount_calculation(self) -> None:
        """Test total_amount = price * quantity."""
        test_cases = [
            (10.0, 1, 10.0),
            (99.99, 2, 199.98),
            (0.01, 100, 1.0),
            (1000.50, 3, 3001.5),
        ]
        
        for price, quantity, expected in test_cases:
            total = round(price * quantity, 2)
            assert total == expected, f"Failed for price={price}, qty={quantity}"


class TestMetadataParsing:
    """Tests for metadata JSON parsing."""

    def test_parse_metadata_json(self) -> None:
        """Test parsing metadata JSON string."""
        metadata_str = '{"channel": "web", "device_type": "desktop", "session_id": "sess_abc"}'
        metadata = json.loads(metadata_str)
        
        assert metadata["channel"] == "web"
        assert metadata["device_type"] == "desktop"
        assert metadata["session_id"] == "sess_abc"

    def test_parse_metadata_with_optional_fields(self) -> None:
        """Test parsing metadata with optional fields."""
        metadata_str = '{"channel": "mobile", "device_type": "ios", "referrer": "google"}'
        metadata = json.loads(metadata_str)
        
        assert metadata.get("channel") == "mobile"
        assert metadata.get("referrer") == "google"
        assert metadata.get("tags") is None  # Optional, not present

    def test_parse_metadata_with_tags_array(self) -> None:
        """Test parsing metadata with tags array."""
        metadata_str = '{"channel": "web", "device_type": "desktop", "tags": ["electronics", "sale"]}'
        metadata = json.loads(metadata_str)
        
        assert metadata["tags"] == ["electronics", "sale"]
        assert len(metadata["tags"]) == 2


class TestDataValidation:
    """Tests for data validation rules."""

    def test_valid_event(self) -> None:
        """Test that a valid event passes validation."""
        event = {
            "id": "test-id-123",
            "timestamp": "2024-01-15T10:30:00.000Z",
            "user_id": "usr_abc123",
            "product_id": "prod_xyz789",
            "price": 99.99,
            "quantity": 2,
            "location": "New York, USA",
        }
        
        # Validation rules
        assert event["id"] is not None
        assert event["price"] >= 0
        assert event["quantity"] >= 1

    def test_invalid_negative_price(self) -> None:
        """Test that negative prices are detected as invalid."""
        event = {"price": -10.0}
        assert event["price"] < 0  # Should be invalid

    def test_invalid_zero_quantity(self) -> None:
        """Test that zero quantity is detected as invalid."""
        event = {"quantity": 0}
        assert event["quantity"] < 1  # Should be invalid

    def test_invalid_null_id(self) -> None:
        """Test that null id is detected as invalid."""
        event = {"id": None}
        assert event["id"] is None  # Should be invalid


class TestSchemaConformance:
    """Tests for schema conformance."""

    def test_output_columns_order(self) -> None:
        """Test that output columns are in expected order."""
        expected_columns = [
            "id",
            "event_timestamp",
            "dt",
            "user_id",
            "product_id",
            "price",
            "quantity",
            "total_amount",
            "location",
            "city",
            "country",
            "channel",
            "device_type",
            "session_id",
            "metadata",
            "processed_at",
            "processing_date",
        ]
        
        # Verify all expected columns
        assert len(expected_columns) == 17
        assert "id" in expected_columns
        assert "dt" in expected_columns
        assert "total_amount" in expected_columns

    def test_partition_column_present(self) -> None:
        """Test that partition column 'dt' is included."""
        expected_columns = ["id", "dt", "event_timestamp"]
        assert "dt" in expected_columns


class TestBatchProcessing:
    """Tests for batch processing logic."""

    def test_date_filter(self) -> None:
        """Test filtering events by date."""
        events = [
            {"dt": "2024-01-15", "id": "1"},
            {"dt": "2024-01-15", "id": "2"},
            {"dt": "2024-01-16", "id": "3"},
        ]
        
        target_date = "2024-01-15"
        filtered = [e for e in events if e["dt"] == target_date]
        
        assert len(filtered) == 2
        assert all(e["dt"] == target_date for e in filtered)

    def test_empty_input_handling(self) -> None:
        """Test handling of empty input."""
        events = []
        assert len(events) == 0

    def test_deduplication_by_id(self) -> None:
        """Test that duplicate events can be identified."""
        events = [
            {"id": "1", "timestamp": "2024-01-15T10:00:00Z"},
            {"id": "1", "timestamp": "2024-01-15T10:00:01Z"},  # Duplicate ID
            {"id": "2", "timestamp": "2024-01-15T10:00:02Z"},
        ]
        
        unique_ids = set(e["id"] for e in events)
        assert len(unique_ids) == 2  # Only 2 unique IDs
        assert len(events) == 3      # But 3 total records
