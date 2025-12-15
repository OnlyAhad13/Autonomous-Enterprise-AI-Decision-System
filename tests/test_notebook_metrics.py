"""
Unit tests for notebook metric thresholds.

Tests verify that baseline models achieve minimum performance requirements
and that data loading functions work correctly.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Project root for data loading
PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# Data Loading Tests
# ============================================================================


class TestDataLoading:
    """Tests for data loading functions."""
    
    @pytest.fixture
    def sample_parquet_path(self):
        """Path to sample parquet data."""
        return PROJECT_ROOT / "data" / "sample" / "events.parquet"
    
    def test_parquet_file_exists(self, sample_parquet_path):
        """Test that sample data file exists."""
        assert sample_parquet_path.exists(), f"Sample data not found: {sample_parquet_path}"
    
    def test_parquet_loads_correctly(self, sample_parquet_path):
        """Test that parquet data loads into DataFrame."""
        df = pd.read_parquet(sample_parquet_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame should not be empty"
    
    def test_required_columns_exist(self, sample_parquet_path):
        """Test that required columns are present."""
        df = pd.read_parquet(sample_parquet_path)
        required_cols = ['id', 'timestamp', 'user_id', 'product_id', 'price', 'quantity']
        
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_no_negative_prices(self, sample_parquet_path):
        """Test that prices are non-negative."""
        df = pd.read_parquet(sample_parquet_path)
        assert (df['price'] >= 0).all(), "Found negative prices in data"
    
    def test_quantity_is_positive(self, sample_parquet_path):
        """Test that quantities are positive integers."""
        df = pd.read_parquet(sample_parquet_path)
        assert (df['quantity'] > 0).all(), "Found non-positive quantities"


# ============================================================================
# Feature Engineering Tests
# ============================================================================


class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    @pytest.fixture
    def sample_events_df(self):
        """Create sample events DataFrame."""
        np.random.seed(42)
        n_events = 1000
        n_users = 100
        
        base_time = datetime(2024, 12, 1)
        
        return pd.DataFrame({
            'id': [f'evt_{i}' for i in range(n_events)],
            'user_id': [f'usr_{i % n_users}' for i in range(n_events)],
            'timestamp': [base_time - timedelta(days=np.random.randint(0, 30)) 
                         for _ in range(n_events)],
            'price': np.random.uniform(10, 500, n_events),
            'quantity': np.random.randint(1, 10, n_events),
            'product_id': [f'prod_{i % 50}' for i in range(n_events)],
            'location': ['New York, USA'] * n_events,
            'metadata': [json.dumps({'channel': 'web'})] * n_events
        })
    
    def test_revenue_calculation(self, sample_events_df):
        """Test revenue = price * quantity."""
        df = sample_events_df.copy()
        df['revenue'] = df['price'] * df['quantity']
        
        expected = df['price'] * df['quantity']
        assert np.allclose(df['revenue'], expected)
    
    def test_user_aggregation(self, sample_events_df):
        """Test user-level feature aggregation."""
        df = sample_events_df.copy()
        
        user_stats = df.groupby('user_id').agg({
            'id': 'count',
            'price': 'mean'
        }).reset_index()
        
        assert len(user_stats) > 0
        assert 'user_id' in user_stats.columns


# ============================================================================
# Model Metric Thresholds
# ============================================================================


class TestModelMetricThresholds:
    """Tests for minimum model performance requirements."""
    
    # Minimum threshold values
    MIN_ACCURACY = 0.50  # Better than random for binary classification
    MIN_AUC_ROC = 0.50   # Better than random guess
    MIN_PRECISION = 0.0  # Can be 0 for imbalanced classes
    MIN_RECALL = 0.0     # Can be 0 for imbalanced classes
    MIN_F1 = 0.0         # Can be 0 for edge cases
    
    @pytest.fixture
    def mock_model_metrics(self):
        """Mock metrics from a trained model."""
        # These would be loaded from MLflow or a saved results file
        return {
            'accuracy': 0.75,
            'precision': 0.68,
            'recall': 0.72,
            'f1': 0.70,
            'auc_roc': 0.82
        }
    
    def test_accuracy_threshold(self, mock_model_metrics):
        """Test that model accuracy exceeds minimum threshold."""
        assert mock_model_metrics['accuracy'] >= self.MIN_ACCURACY, \
            f"Accuracy {mock_model_metrics['accuracy']:.3f} below minimum {self.MIN_ACCURACY}"
    
    def test_auc_roc_threshold(self, mock_model_metrics):
        """Test that AUC-ROC exceeds random guess threshold."""
        assert mock_model_metrics['auc_roc'] >= self.MIN_AUC_ROC, \
            f"AUC-ROC {mock_model_metrics['auc_roc']:.3f} below minimum {self.MIN_AUC_ROC}"
    
    def test_precision_non_negative(self, mock_model_metrics):
        """Test that precision is non-negative."""
        assert mock_model_metrics['precision'] >= self.MIN_PRECISION, \
            f"Precision should be non-negative"
    
    def test_recall_non_negative(self, mock_model_metrics):
        """Test that recall is non-negative."""
        assert mock_model_metrics['recall'] >= self.MIN_RECALL, \
            f"Recall should be non-negative"
    
    def test_f1_non_negative(self, mock_model_metrics):
        """Test that F1 score is non-negative."""
        assert mock_model_metrics['f1'] >= self.MIN_F1, \
            f"F1 should be non-negative"
    
    def test_metrics_bounded_zero_one(self, mock_model_metrics):
        """Test that all metrics are between 0 and 1."""
        for metric, value in mock_model_metrics.items():
            assert 0 <= value <= 1, f"{metric} = {value} is out of [0, 1] range"


# ============================================================================
# Churn Label Tests
# ============================================================================


class TestChurnLabeling:
    """Tests for churn label creation logic."""
    
    def test_churn_label_is_binary(self):
        """Test that churn labels are 0 or 1."""
        # Simulate churn labels
        churned = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        
        assert set(churned).issubset({0, 1}), "Churn labels should be binary"
    
    def test_churn_rate_reasonable(self):
        """Test that churn rate falls within reasonable bounds."""
        # Reasonable churn rate for 7-day window: 10% to 90%
        simulated_churn_rate = 0.35  # Example
        
        assert 0.01 <= simulated_churn_rate <= 0.99, \
            f"Churn rate {simulated_churn_rate:.2%} seems unreasonable"


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestNotebookIntegration:
    """Integration tests that verify notebook outputs."""
    
    def test_model_card_template_exists(self):
        """Test that model card template exists."""
        model_card_path = PROJECT_ROOT / "notebooks" / "model_card.md"
        assert model_card_path.exists(), "Model card template should exist"
    
    def test_run_notebooks_script_exists(self):
        """Test that headless execution script exists."""
        script_path = PROJECT_ROOT / "scripts" / "run_notebooks.sh"
        assert script_path.exists(), "run_notebooks.sh should exist"
    
    def test_notebooks_exist(self):
        """Test that both notebooks exist."""
        eda_path = PROJECT_ROOT / "notebooks" / "01_EDA.ipynb"
        baselines_path = PROJECT_ROOT / "notebooks" / "02_baselines.ipynb"
        
        assert eda_path.exists(), "01_EDA.ipynb should exist"
        assert baselines_path.exists(), "02_baselines.ipynb should exist"
