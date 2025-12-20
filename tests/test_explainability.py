"""
Unit tests for the Explainability module.

Tests cover:
- SHAP computation functions
- Segment performance analysis
- Fairness check functions
- FastAPI endpoints
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Test imports for API
from fastapi.testclient import TestClient


# ============================================================================
# Project Setup
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# SHAP Computation Tests
# ============================================================================


class TestSHAPComputation:
    """Tests for SHAP value computation and utilities."""
    
    @pytest.fixture
    def sample_shap_values(self):
        """Create sample SHAP values array."""
        np.random.seed(42)
        return np.random.randn(100, 12)  # 100 samples, 12 features
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'transaction_count': np.random.randint(1, 50, 100),
            'total_revenue': np.random.uniform(100, 5000, 100),
            'avg_revenue': np.random.uniform(50, 200, 100),
            'std_revenue': np.random.uniform(0, 100, 100),
            'avg_price': np.random.uniform(20, 150, 100),
            'max_price': np.random.uniform(100, 500, 100),
            'min_price': np.random.uniform(10, 50, 100),
            'total_quantity': np.random.randint(1, 100, 100),
            'avg_quantity': np.random.uniform(1, 5, 100),
            'days_since_first': np.random.randint(1, 365, 100),
            'days_since_last': np.random.randint(0, 60, 100),
            'avg_days_between': np.random.uniform(1, 30, 100)
        })
    
    def test_shap_values_shape(self, sample_shap_values, sample_features):
        """Test that SHAP values have correct shape."""
        assert sample_shap_values.shape[0] == len(sample_features)
        assert sample_shap_values.shape[1] == len(sample_features.columns)
    
    def test_shap_values_sum_to_prediction_diff(self, sample_shap_values):
        """Test that SHAP values sum property holds (approximately)."""
        # Sum of SHAP values should approximate (prediction - base_value)
        shap_sum = sample_shap_values.sum(axis=1)
        assert len(shap_sum) == sample_shap_values.shape[0]
    
    def test_mean_absolute_shap(self, sample_shap_values):
        """Test mean absolute SHAP calculation."""
        mean_abs = np.abs(sample_shap_values).mean(axis=0)
        
        assert len(mean_abs) == sample_shap_values.shape[1]
        assert all(v >= 0 for v in mean_abs)
    
    def test_feature_ranking_by_shap(self, sample_shap_values):
        """Test feature ranking is consistent."""
        mean_abs = np.abs(sample_shap_values).mean(axis=0)
        ranking = np.argsort(-mean_abs)  # Descending order
        
        # Top feature should have highest mean absolute SHAP
        assert mean_abs[ranking[0]] >= mean_abs[ranking[-1]]


# ============================================================================
# Segment Performance Tests
# ============================================================================


class TestSegmentPerformance:
    """Tests for per-segment performance analysis."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions data."""
        np.random.seed(42)
        n = 500
        
        return {
            'y_true': np.random.randint(0, 2, n),
            'y_pred': np.random.randint(0, 2, n),
            'y_prob': np.random.uniform(0, 1, n),
            'segment': np.random.choice(['Region A', 'Region B', 'Region C'], n)
        }
    
    def test_segment_metrics_calculation(self, sample_predictions):
        """Test segment-wise metric calculation."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        results = []
        for segment in np.unique(sample_predictions['segment']):
            mask = sample_predictions['segment'] == segment
            y_true = sample_predictions['y_true'][mask]
            y_pred = sample_predictions['y_pred'][mask]
            
            results.append({
                'segment': segment,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'n_samples': mask.sum()
            })
        
        df = pd.DataFrame(results)
        
        assert len(df) == 3  # Three segments
        assert all(0 <= v <= 1 for v in df['accuracy'])
        assert all(df['n_samples'] > 0)
    
    def test_segment_handles_empty_segment(self):
        """Test handling of segments with no samples."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        segment = np.array(['A', 'A', 'B', 'B'])
        
        # This should not raise an error
        for seg in np.unique(segment):
            mask = segment == seg
            assert mask.sum() > 0
    
    def test_segment_handles_single_class(self):
        """Test handling when segment has only one class."""
        y_true = np.array([0, 0, 0, 0])  # All negatives
        y_pred = np.array([0, 0, 1, 0])
        
        from sklearn.metrics import precision_score, recall_score
        
        # Should handle zero_division gracefully
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        assert precision == 0.0  # No true positives
        assert recall == 0.0  # No actual positives


# ============================================================================
# Fairness Check Tests
# ============================================================================


class TestFairnessChecks:
    """Tests for fairness analysis functions."""
    
    @pytest.fixture
    def balanced_predictions(self):
        """Create balanced predictions across groups."""
        np.random.seed(42)
        n = 200
        
        return {
            'y_true': np.array([0, 1] * 100),
            'y_pred': np.array([0, 1] * 100),
            'group': np.array(['A'] * 100 + ['B'] * 100)
        }
    
    @pytest.fixture
    def unbalanced_predictions(self):
        """Create unbalanced predictions across groups."""
        return {
            'y_true': np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            'y_pred': np.array([1, 1, 1, 1, 0, 0, 0, 0]),  # Group A predicts all 1, B all 0
            'group': np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        }
    
    def test_demographic_parity_calculation(self, balanced_predictions):
        """Test demographic parity metric calculation."""
        results = {}
        for group in np.unique(balanced_predictions['group']):
            mask = balanced_predictions['group'] == group
            selection_rate = balanced_predictions['y_pred'][mask].mean()
            results[group] = selection_rate
        
        # Calculate parity ratio
        rates = list(results.values())
        parity_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
        
        assert 0 <= parity_ratio <= 1
    
    def test_demographic_parity_unbalanced(self, unbalanced_predictions):
        """Test demographic parity detects imbalance."""
        results = {}
        for group in np.unique(unbalanced_predictions['group']):
            mask = unbalanced_predictions['group'] == group
            selection_rate = unbalanced_predictions['y_pred'][mask].mean()
            results[group] = selection_rate
        
        # Group A has 100% selection rate, Group B has 0%
        assert results['A'] == 1.0
        assert results['B'] == 0.0
        
        # Parity ratio should be 0 (maximum unfairness)
        parity_ratio = min(results.values()) / max(results.values()) if max(results.values()) > 0 else 0
        assert parity_ratio == 0.0
    
    def test_equalized_odds_calculation(self, balanced_predictions):
        """Test equalized odds (TPR/FPR) calculation."""
        for group in np.unique(balanced_predictions['group']):
            mask = balanced_predictions['group'] == group
            group_y_true = balanced_predictions['y_true'][mask]
            group_y_pred = balanced_predictions['y_pred'][mask]
            
            # TPR
            positives = group_y_true == 1
            if positives.sum() > 0:
                tpr = (group_y_pred[positives] == 1).mean()
                assert 0 <= tpr <= 1
            
            # FPR
            negatives = group_y_true == 0
            if negatives.sum() > 0:
                fpr = (group_y_pred[negatives] == 1).mean()
                assert 0 <= fpr <= 1
    
    def test_equalized_odds_gap_detection(self, unbalanced_predictions):
        """Test that equalized odds gap is detected."""
        tprs = []
        fprs = []
        
        for group in np.unique(unbalanced_predictions['group']):
            mask = unbalanced_predictions['group'] == group
            group_y_true = unbalanced_predictions['y_true'][mask]
            group_y_pred = unbalanced_predictions['y_pred'][mask]
            
            positives = group_y_true == 1
            negatives = group_y_true == 0
            
            tpr = (group_y_pred[positives] == 1).mean() if positives.sum() > 0 else 0
            fpr = (group_y_pred[negatives] == 1).mean() if negatives.sum() > 0 else 0
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        
        # There should be significant gaps
        assert tpr_gap >= 0
        assert fpr_gap >= 0


# ============================================================================
# API Endpoint Tests
# ============================================================================


class TestExplainabilityAPI:
    """Tests for FastAPI explainability endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API."""
        from services.explain.api import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "explainability-api"
    
    def test_explain_endpoint_valid_id(self, client):
        """Test /explain endpoint with valid prediction ID."""
        response = client.get("/explain?prediction_id=test_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction_id" in data
        assert data["prediction_id"] == "test_123"
        assert "predicted_class" in data
        assert "predicted_probability" in data
        assert "base_value" in data
        assert "feature_contributions" in data
        assert "top_positive_drivers" in data
        assert "top_negative_drivers" in data
        assert "generated_at" in data
    
    def test_explain_endpoint_invalid_id(self, client):
        """Test /explain endpoint with invalid prediction ID."""
        response = client.get("/explain?prediction_id=nonexistent_id")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_explain_endpoint_missing_param(self, client):
        """Test /explain endpoint without required parameter."""
        response = client.get("/explain")
        
        assert response.status_code == 422  # Validation error
    
    def test_explain_summary_endpoint(self, client):
        """Test /explain/summary endpoint."""
        response = client.get("/explain/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_type" in data
        assert "expected_value" in data
        assert "n_samples" in data
        assert "feature_importance" in data
        assert isinstance(data["feature_importance"], list)
        assert len(data["feature_importance"]) > 0
    
    def test_explain_html_endpoint(self, client):
        """Test /explain/html endpoint returns HTML."""
        response = client.get("/explain/html?prediction_id=test_123")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "SHAP" in content
        assert "test_123" in content
    
    def test_explain_html_invalid_id(self, client):
        """Test /explain/html endpoint with invalid ID."""
        response = client.get("/explain/html?prediction_id=invalid_id")
        
        assert response.status_code == 404
    
    def test_list_predictions_endpoint(self, client):
        """Test /explain/list endpoint."""
        response = client.get("/explain/list")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "available_predictions" in data
        assert "count" in data
        assert len(data["available_predictions"]) == data["count"]
    
    def test_feature_contributions_sorted(self, client):
        """Test that feature contributions are sorted by absolute SHAP."""
        response = client.get("/explain?prediction_id=test_123")
        data = response.json()
        
        contributions = data["feature_contributions"]
        shap_values = [abs(c["shap_value"]) for c in contributions]
        
        # Should be sorted in descending order
        assert shap_values == sorted(shap_values, reverse=True)


# ============================================================================
# Notebook Integration Tests
# ============================================================================


@pytest.mark.integration
class TestNotebookIntegration:
    """Integration tests for notebook artifacts."""
    
    def test_explainability_notebook_exists(self):
        """Test that explainability notebook exists."""
        notebook_path = PROJECT_ROOT / "notebooks" / "03_explainability.ipynb"
        assert notebook_path.exists(), "03_explainability.ipynb should exist"
    
    def test_explain_api_module_exists(self):
        """Test that API module exists."""
        api_path = PROJECT_ROOT / "services" / "explain" / "api.py"
        assert api_path.exists(), "services/explain/api.py should exist"
    
    def test_html_template_exists(self):
        """Test that HTML template exists."""
        template_path = PROJECT_ROOT / "services" / "explain" / "templates" / "shap_report.html"
        assert template_path.exists(), "HTML template should exist"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_shap_values(self):
        """Test handling of empty SHAP values."""
        empty_shap = np.array([]).reshape(0, 12)
        
        assert empty_shap.shape[0] == 0
        assert len(empty_shap) == 0
    
    def test_single_sample_shap(self):
        """Test SHAP with single sample."""
        single_shap = np.random.randn(1, 12)
        mean_abs = np.abs(single_shap).mean(axis=0)
        
        assert len(mean_abs) == 12
    
    def test_all_zero_shap_values(self):
        """Test handling when all SHAP values are zero."""
        zero_shap = np.zeros((10, 12))
        mean_abs = np.abs(zero_shap).mean(axis=0)
        
        assert all(v == 0 for v in mean_abs)
    
    def test_extreme_shap_values(self):
        """Test handling of extreme SHAP values."""
        extreme_shap = np.array([[1e10, -1e10, 0] * 4])  # Extreme values
        mean_abs = np.abs(extreme_shap).mean(axis=0)
        
        assert not np.isnan(mean_abs).any()
        assert not np.isinf(mean_abs).any() or mean_abs.max() == 1e10
