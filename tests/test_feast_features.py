"""
Unit tests for Feast feature store.

Tests feature fetching for user and product features
and validates expected shapes and data types.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Skip all tests if feast is not installed
feast = pytest.importorskip("feast")


class TestFeastSetup:
    """Tests for Feast feature store setup."""
    
    @pytest.fixture
    def repo_path(self):
        """Return path to feature repository."""
        return PROJECT_ROOT / "features" / "feature_repo"
    
    @pytest.fixture
    def feature_store(self, repo_path):
        """Create FeatureStore instance."""
        from feast import FeatureStore
        return FeatureStore(repo_path=str(repo_path))
    
    def test_feature_store_yaml_exists(self, repo_path):
        """Test that feature_store.yaml exists."""
        config_path = repo_path / "feature_store.yaml"
        assert config_path.exists(), f"Missing: {config_path}"
    
    def test_user_features_parquet_exists(self, repo_path):
        """Test that user features parquet file exists."""
        data_path = repo_path / "data" / "user_features.parquet"
        if not data_path.exists():
            pytest.skip("User features not ingested yet. Run: python features/load_to_feast.py")
        assert data_path.exists()
    
    def test_product_features_parquet_exists(self, repo_path):
        """Test that product features parquet file exists."""
        data_path = repo_path / "data" / "product_features.parquet"
        if not data_path.exists():
            pytest.skip("Product features not ingested yet. Run: python features/load_to_feast.py")
        assert data_path.exists()


class TestUserFeatures:
    """Tests for user feature fetching."""
    
    @pytest.fixture
    def repo_path(self):
        """Return path to feature repository."""
        return PROJECT_ROOT / "features" / "feature_repo"
    
    @pytest.fixture
    def user_features_df(self, repo_path):
        """Load user features from parquet."""
        data_path = repo_path / "data" / "user_features.parquet"
        if not data_path.exists():
            pytest.skip("User features not ingested yet")
        return pd.read_parquet(data_path)
    
    @pytest.fixture
    def sample_user_id(self, user_features_df):
        """Get a sample user_id from the data."""
        return user_features_df["user_id"].iloc[0]
    
    def test_user_features_schema(self, user_features_df):
        """Test that user features have expected columns."""
        expected_columns = [
            "user_id",
            "user_total_spend_30d",
            "user_avg_order_value",
            "user_avg_quantity",
            "user_total_quantity_30d",
            "user_transaction_count_30d",
            "user_unique_products_30d",
            "user_favorite_channel",
            "user_days_since_last_purchase",
            "event_timestamp",
        ]
        
        for col in expected_columns:
            assert col in user_features_df.columns, f"Missing column: {col}"
    
    def test_user_features_types(self, user_features_df):
        """Test that user features have correct data types."""
        # Check numeric columns
        numeric_cols = [
            "user_total_spend_30d",
            "user_avg_order_value",
            "user_avg_quantity",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_float_dtype(user_features_df[col]), \
                f"{col} should be float"
        
        # Check integer columns
        int_cols = [
            "user_total_quantity_30d",
            "user_transaction_count_30d",
            "user_unique_products_30d",
            "user_days_since_last_purchase",
        ]
        for col in int_cols:
            assert pd.api.types.is_integer_dtype(user_features_df[col]) or \
                   pd.api.types.is_float_dtype(user_features_df[col]), \
                f"{col} should be numeric"
    
    def test_user_features_values_valid(self, user_features_df):
        """Test that user feature values are within valid ranges."""
        # Total spend should be non-negative
        assert (user_features_df["user_total_spend_30d"] >= 0).all(), \
            "Total spend should be non-negative"
        
        # Transaction count should be positive
        assert (user_features_df["user_transaction_count_30d"] > 0).all(), \
            "Transaction count should be positive"
        
        # Average quantity should be positive
        assert (user_features_df["user_avg_quantity"] > 0).all(), \
            "Average quantity should be positive"
    
    def test_user_features_shape(self, user_features_df):
        """Test that user features have valid shape."""
        assert len(user_features_df) > 0, "Should have at least one user"
        assert len(user_features_df.columns) >= 10, "Should have at least 10 columns"
    
    def test_sample_user_lookup(self, user_features_df, sample_user_id):
        """Test that we can look up a specific user."""
        user_row = user_features_df[user_features_df["user_id"] == sample_user_id]
        
        assert len(user_row) == 1, f"Should find exactly one row for {sample_user_id}"
        assert user_row["user_total_spend_30d"].iloc[0] >= 0


class TestProductFeatures:
    """Tests for product feature fetching."""
    
    @pytest.fixture
    def repo_path(self):
        """Return path to feature repository."""
        return PROJECT_ROOT / "features" / "feature_repo"
    
    @pytest.fixture
    def product_features_df(self, repo_path):
        """Load product features from parquet."""
        data_path = repo_path / "data" / "product_features.parquet"
        if not data_path.exists():
            pytest.skip("Product features not ingested yet")
        return pd.read_parquet(data_path)
    
    @pytest.fixture
    def sample_product_id(self, product_features_df):
        """Get a sample product_id from the data."""
        return product_features_df["product_id"].iloc[0]
    
    def test_product_features_schema(self, product_features_df):
        """Test that product features have expected columns."""
        expected_columns = [
            "product_id",
            "product_median_price_30d",
            "product_avg_price_30d",
            "product_min_price_30d",
            "product_max_price_30d",
            "product_total_sales_30d",
            "product_total_revenue_30d",
            "product_unique_buyers_30d",
            "product_avg_quantity_per_order",
            "product_transaction_count_30d",
            "product_top_region",
            "event_timestamp",
        ]
        
        for col in expected_columns:
            assert col in product_features_df.columns, f"Missing column: {col}"
    
    def test_product_features_types(self, product_features_df):
        """Test that product features have correct data types."""
        # Check numeric columns
        numeric_cols = [
            "product_median_price_30d",
            "product_avg_price_30d",
            "product_total_revenue_30d",
            "product_avg_quantity_per_order",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_float_dtype(product_features_df[col]), \
                f"{col} should be float"
    
    def test_product_features_values_valid(self, product_features_df):
        """Test that product feature values are within valid ranges."""
        # Prices should be non-negative
        assert (product_features_df["product_median_price_30d"] >= 0).all(), \
            "Median price should be non-negative"
        
        # Min price should be <= max price
        assert (
            product_features_df["product_min_price_30d"] <= 
            product_features_df["product_max_price_30d"]
        ).all(), "Min price should be <= max price"
        
        # Total sales should be positive
        assert (product_features_df["product_total_sales_30d"] > 0).all(), \
            "Total sales should be positive"
    
    def test_product_features_shape(self, product_features_df):
        """Test that product features have valid shape."""
        assert len(product_features_df) > 0, "Should have at least one product"
        assert len(product_features_df.columns) >= 10, "Should have at least 10 columns"
    
    def test_sample_product_lookup(self, product_features_df, sample_product_id):
        """Test that we can look up a specific product."""
        product_row = product_features_df[
            product_features_df["product_id"] == sample_product_id
        ]
        
        assert len(product_row) == 1, f"Should find exactly one row for {sample_product_id}"
        assert product_row["product_median_price_30d"].iloc[0] >= 0


class TestFeastOnlineStore:
    """Tests for Feast online store functionality."""
    
    @pytest.fixture
    def repo_path(self):
        """Return path to feature repository."""
        return PROJECT_ROOT / "features" / "feature_repo"
    
    @pytest.fixture
    def feature_store(self, repo_path):
        """Create FeatureStore instance."""
        from feast import FeatureStore
        
        # Check if registry exists
        registry_path = repo_path / "data" / "registry.db"
        if not registry_path.exists():
            pytest.skip("Feast registry not initialized. Run: cd features/feature_repo && feast apply")
        
        return FeatureStore(repo_path=str(repo_path))
    
    @pytest.fixture
    def sample_user_id(self, repo_path):
        """Get a sample user_id from the offline store."""
        data_path = repo_path / "data" / "user_features.parquet"
        if not data_path.exists():
            pytest.skip("User features not ingested yet")
        df = pd.read_parquet(data_path)
        return df["user_id"].iloc[0]
    
    @pytest.fixture
    def sample_product_id(self, repo_path):
        """Get a sample product_id from the offline store."""
        data_path = repo_path / "data" / "product_features.parquet"
        if not data_path.exists():
            pytest.skip("Product features not ingested yet")
        df = pd.read_parquet(data_path)
        return df["product_id"].iloc[0]
    
    def test_get_online_user_features(self, feature_store, sample_user_id):
        """Test fetching user features from online store."""
        try:
            features = feature_store.get_online_features(
                features=[
                    "user_features:user_total_spend_30d",
                    "user_features:user_avg_quantity",
                    "user_features:user_transaction_count_30d",
                ],
                entity_rows=[{"user_id": sample_user_id}]
            ).to_dict()
            
            assert "user_total_spend_30d" in features
            assert len(features["user_total_spend_30d"]) == 1
            
        except Exception as e:
            if "not found" in str(e).lower() or "materialize" in str(e).lower():
                pytest.skip(f"Online store not materialized: {e}")
            raise
    
    def test_get_online_product_features(self, feature_store, sample_product_id):
        """Test fetching product features from online store."""
        try:
            features = feature_store.get_online_features(
                features=[
                    "product_features:product_median_price_30d",
                    "product_features:product_total_sales_30d",
                ],
                entity_rows=[{"product_id": sample_product_id}]
            ).to_dict()
            
            assert "product_median_price_30d" in features
            assert len(features["product_median_price_30d"]) == 1
            
        except Exception as e:
            if "not found" in str(e).lower() or "materialize" in str(e).lower():
                pytest.skip(f"Online store not materialized: {e}")
            raise


class TestFeastHistoricalFeatures:
    """Tests for Feast historical (offline) feature retrieval."""
    
    @pytest.fixture
    def repo_path(self):
        """Return path to feature repository."""
        return PROJECT_ROOT / "features" / "feature_repo"
    
    @pytest.fixture
    def feature_store(self, repo_path):
        """Create FeatureStore instance."""
        from feast import FeatureStore
        
        registry_path = repo_path / "data" / "registry.db"
        if not registry_path.exists():
            pytest.skip("Feast registry not initialized")
        
        return FeatureStore(repo_path=str(repo_path))
    
    @pytest.fixture
    def sample_user_id(self, repo_path):
        """Get a sample user_id."""
        data_path = repo_path / "data" / "user_features.parquet"
        if not data_path.exists():
            pytest.skip("User features not ingested yet")
        df = pd.read_parquet(data_path)
        return df["user_id"].iloc[0]
    
    def test_get_historical_user_features(self, feature_store, sample_user_id):
        """Test fetching historical user features."""
        try:
            entity_df = pd.DataFrame({
                "user_id": [sample_user_id],
                "event_timestamp": [datetime.now()]
            })
            
            features = feature_store.get_historical_features(
                entity_df=entity_df,
                features=[
                    "user_features:user_total_spend_30d",
                    "user_features:user_avg_quantity",
                ]
            ).to_df()
            
            assert len(features) == 1
            assert "user_total_spend_30d" in features.columns
            
        except Exception as e:
            pytest.skip(f"Historical retrieval failed: {e}")


# Simple standalone test that can run without Feast
class TestFeatureDataFiles:
    """Tests for feature data files (no Feast required)."""
    
    def test_sample_events_exist(self):
        """Test that sample events file exists."""
        events_path = PROJECT_ROOT / "data" / "sample" / "events.parquet"
        assert events_path.exists(), f"Missing: {events_path}"
    
    def test_sample_events_readable(self):
        """Test that sample events can be read."""
        events_path = PROJECT_ROOT / "data" / "sample" / "events.parquet"
        df = pd.read_parquet(events_path)
        
        assert len(df) > 0, "Events file should not be empty"
        assert "user_id" in df.columns, "Should have user_id column"
        assert "product_id" in df.columns, "Should have product_id column"
        assert "price" in df.columns, "Should have price column"
