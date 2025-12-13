"""
Unit tests for streaming transforms.

Tests the pure Python transform functions without requiring Spark infrastructure.
"""

import json
import pytest
from datetime import datetime


# Import the testable functions
import sys
sys.path.insert(0, str(__file__).replace("tests/test_streaming_transforms.py", "spark_jobs"))

from transforms import (
    validate_event,
    normalize_event_dict,
    to_canonical_event,
    REQUIRED_FIELDS,
)
from geo_lookup import (
    lookup_region_by_coords,
    lookup_region_by_city,
)


class TestValidateEvent:
    """Tests for event validation logic."""
    
    def test_valid_event_passes(self):
        """Test that a complete valid event passes validation."""
        event = {
            "id": "evt-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "user_id": "user_1",
            "product_id": "prod_1",
            "price": 99.99,
            "quantity": 2,
            "location": "New York, USA",
            "metadata": {"channel": "web"}
        }
        
        is_valid, errors = validate_event(event)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_missing_required_field_fails(self):
        """Test that missing required fields are caught."""
        event = {
            "id": "evt-123",
            "timestamp": "2024-01-15T10:30:00Z",
            # Missing user_id
            "product_id": "prod_1",
            "price": 99.99,
            "quantity": 2,
            "location": "New York, USA",
        }
        
        is_valid, errors = validate_event(event)
        
        assert is_valid is False
        assert any("user_id" in e for e in errors)
    
    def test_null_required_field_fails(self):
        """Test that null required fields are caught."""
        event = {
            "id": "evt-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "user_id": None,  # Null value
            "product_id": "prod_1",
            "price": 99.99,
            "quantity": 2,
            "location": "New York, USA",
        }
        
        is_valid, errors = validate_event(event)
        
        assert is_valid is False
        assert any("user_id" in e for e in errors)
    
    def test_negative_price_fails(self):
        """Test that negative prices are rejected."""
        event = {
            "id": "evt-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "user_id": "user_1",
            "product_id": "prod_1",
            "price": -10.00,  # Invalid
            "quantity": 2,
            "location": "New York, USA",
        }
        
        is_valid, errors = validate_event(event)
        
        assert is_valid is False
        assert any("price" in e for e in errors)
    
    def test_zero_quantity_fails(self):
        """Test that quantity < 1 is rejected."""
        event = {
            "id": "evt-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "user_id": "user_1",
            "product_id": "prod_1",
            "price": 99.99,
            "quantity": 0,  # Invalid
            "location": "New York, USA",
        }
        
        is_valid, errors = validate_event(event)
        
        assert is_valid is False
        assert any("quantity" in e for e in errors)
    
    def test_invalid_timestamp_format_fails(self):
        """Test that invalid timestamp formats are caught."""
        event = {
            "id": "evt-123",
            "timestamp": "not-a-timestamp",  # Invalid
            "user_id": "user_1",
            "product_id": "prod_1",
            "price": 99.99,
            "quantity": 2,
            "location": "New York, USA",
        }
        
        is_valid, errors = validate_event(event)
        
        assert is_valid is False
        assert any("timestamp" in e for e in errors)
    
    def test_multiple_errors_reported(self):
        """Test that all validation errors are collected."""
        event = {
            "id": None,
            "timestamp": "invalid",
            "user_id": None,
            "product_id": "prod_1",
            "price": -5,
            "quantity": 0,
            "location": "NYC",
        }
        
        is_valid, errors = validate_event(event)
        
        assert is_valid is False
        assert len(errors) >= 4  # Multiple issues


class TestNormalizeEvent:
    """Tests for event normalization logic."""
    
    def test_user_id_prefix_added(self):
        """Test that user_id gets usr_ prefix if missing."""
        event = {"user_id": "12345"}
        
        result = normalize_event_dict(event)
        
        assert result["user_id"] == "usr_12345"
    
    def test_user_id_prefix_not_duplicated(self):
        """Test that existing usr_ prefix is not duplicated."""
        event = {"user_id": "usr_12345"}
        
        result = normalize_event_dict(event)
        
        assert result["user_id"] == "usr_12345"
    
    def test_product_id_prefix_added(self):
        """Test that product_id gets prod_ prefix if missing."""
        event = {"product_id": "ABC"}
        
        result = normalize_event_dict(event)
        
        assert result["product_id"] == "prod_ABC"
    
    def test_product_id_prefix_not_duplicated(self):
        """Test that existing prod_ prefix is not duplicated."""
        event = {"product_id": "prod_ABC"}
        
        result = normalize_event_dict(event)
        
        assert result["product_id"] == "prod_ABC"
    
    def test_city_extracted_from_location(self):
        """Test that city is extracted from location string."""
        event = {"location": "New York, USA"}
        
        result = normalize_event_dict(event)
        
        assert result["city"] == "New York"
    
    def test_country_extracted_from_location(self):
        """Test that country is extracted from location string."""
        event = {"location": "New York, USA"}
        
        result = normalize_event_dict(event)
        
        assert result["country"] == "USA"
    
    def test_total_amount_calculated(self):
        """Test that total_amount is calculated correctly."""
        event = {"price": 10.50, "quantity": 3}
        
        result = normalize_event_dict(event)
        
        assert result["total_amount"] == 31.50
    
    def test_total_amount_rounded(self):
        """Test that total_amount is rounded to 2 decimal places."""
        event = {"price": 10.333, "quantity": 3}
        
        result = normalize_event_dict(event)
        
        assert result["total_amount"] == 31.0  # 10.333 * 3 = 30.999 -> 31.0


class TestToCanonicalEvent:
    """Tests for canonical event formatting."""
    
    def test_canonical_structure(self):
        """Test that canonical event has correct structure."""
        event = {
            "id": "evt-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "user_id": "usr_1",
            "product_id": "prod_1",
            "price": 99.99,
            "quantity": 2,
            "total_amount": 199.98,
            "location": "New York, USA",
            "city": "New York",
            "country": "USA",
            "session_id": "sess_abc",
            "channel": "web",
            "device_type": "desktop",
            "dt": "2024-01-15",
        }
        
        result = to_canonical_event(event, region="NA")
        
        assert result["event_id"] == "evt-123"
        assert result["event_type"] == "transaction"
        assert result["event_time"] == "2024-01-15T10:30:00Z"
        assert result["user"]["id"] == "usr_1"
        assert result["product"]["id"] == "prod_1"
        assert result["transaction"]["price"] == 99.99
        assert result["transaction"]["quantity"] == 2
        assert result["transaction"]["total"] == 199.98
        assert result["location"]["city"] == "New York"
        assert result["location"]["country"] == "USA"
        assert result["location"]["region"] == "NA"
        assert result["channel"] == "web"
    
    def test_canonical_event_serializable(self):
        """Test that canonical event can be JSON serialized."""
        event = {
            "id": "evt-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "user_id": "usr_1",
            "product_id": "prod_1",
            "price": 99.99,
            "quantity": 2,
        }
        
        result = to_canonical_event(event)
        
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "evt-123"


class TestGeoLookup:
    """Tests for geo enrichment logic."""
    
    # Sample regions for testing
    TEST_REGIONS = [
        {"name": "North America", "code": "NA", 
         "bounds": {"min_lat": 15.0, "max_lat": 72.0, "min_lon": -170.0, "max_lon": -50.0}},
        {"name": "Europe", "code": "EU", 
         "bounds": {"min_lat": 35.0, "max_lat": 72.0, "min_lon": -25.0, "max_lon": 65.0}},
        {"name": "Asia", "code": "AS", 
         "bounds": {"min_lat": -10.0, "max_lat": 77.0, "min_lon": 65.0, "max_lon": 180.0}},
    ]
    
    TEST_CITIES = {
        "New York": {"region": "NA"},
        "London": {"region": "EU"},
        "Tokyo": {"region": "AS"},
    }
    
    def test_lookup_new_york_coords(self):
        """Test coordinates for New York return NA."""
        result = lookup_region_by_coords(
            lat=40.7128, lon=-74.0060,
            regions=self.TEST_REGIONS,
            default="UNKNOWN"
        )
        assert result == "NA"
    
    def test_lookup_london_coords(self):
        """Test coordinates for London return EU."""
        result = lookup_region_by_coords(
            lat=51.5074, lon=-0.1278,
            regions=self.TEST_REGIONS,
            default="UNKNOWN"
        )
        assert result == "EU"
    
    def test_lookup_tokyo_coords(self):
        """Test coordinates for Tokyo return AS."""
        result = lookup_region_by_coords(
            lat=35.6762, lon=139.6503,
            regions=self.TEST_REGIONS,
            default="UNKNOWN"
        )
        assert result == "AS"
    
    def test_lookup_null_coords_returns_default(self):
        """Test that null coordinates return default region."""
        result = lookup_region_by_coords(
            lat=None, lon=None,
            regions=self.TEST_REGIONS,
            default="UNKNOWN"
        )
        assert result == "UNKNOWN"
    
    def test_lookup_unmapped_coords_returns_default(self):
        """Test that coordinates outside all regions return default."""
        # Antarctica
        result = lookup_region_by_coords(
            lat=-85.0, lon=0.0,
            regions=self.TEST_REGIONS,
            default="UNKNOWN"
        )
        assert result == "UNKNOWN"
    
    def test_city_lookup_new_york(self):
        """Test city lookup for New York."""
        result = lookup_region_by_city(
            city="New York",
            city_overrides=self.TEST_CITIES,
            default="UNKNOWN"
        )
        assert result == "NA"
    
    def test_city_lookup_case_insensitive(self):
        """Test city lookup handles different cases."""
        result = lookup_region_by_city(
            city="new york",  # lowercase
            city_overrides=self.TEST_CITIES,
            default="UNKNOWN"
        )
        assert result == "NA"
    
    def test_city_lookup_unknown_city_returns_none(self):
        """Test that unknown city returns None (not default)."""
        result = lookup_region_by_city(
            city="Random City",
            city_overrides=self.TEST_CITIES,
            default="UNKNOWN"
        )
        # Should return None so caller falls back to coords
        assert result is None
    
    def test_city_lookup_null_returns_default(self):
        """Test that null city returns default."""
        result = lookup_region_by_city(
            city=None,
            city_overrides=self.TEST_CITIES,
            default="UNKNOWN"
        )
        assert result == "UNKNOWN"


class TestRequiredFields:
    """Tests for schema constants."""
    
    def test_required_fields_complete(self):
        """Test that all required fields are defined."""
        expected = ["id", "timestamp", "user_id", "product_id", "price", "quantity", "location"]
        
        for field in expected:
            assert field in REQUIRED_FIELDS
    
    def test_required_fields_count(self):
        """Test required field count."""
        assert len(REQUIRED_FIELDS) == 7
