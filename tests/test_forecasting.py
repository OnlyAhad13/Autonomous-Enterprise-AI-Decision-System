"""
Tests for the forecasting training pipeline.

Validates that:
- Training runs complete for all model types
- Forecasts are generated with expected lengths
- Metrics are computed correctly
- Predict service can load models and generate forecasts
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Mark module as providing forecasting tests
pytestmark = [pytest.mark.filterwarnings("ignore")]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_time_series() -> pd.DataFrame:
    """Create sample time series data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    np.random.seed(42)
    values = 1000 + np.cumsum(np.random.randn(30) * 50)

    return pd.DataFrame({
        "ds": dates,
        "y": values,
    })


@pytest.fixture
def train_val_data(sample_time_series: pd.DataFrame):
    """Split sample data into train/val."""
    train = sample_time_series.iloc[:-7].copy()
    val = sample_time_series.iloc[-7:].copy()
    return train, val


@pytest.fixture
def sample_events_data(tmp_path: Path) -> Path:
    """Create sample events parquet file."""
    np.random.seed(42)
    n_events = 500

    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    events = pd.DataFrame({
        "id": [f"evt_{i}" for i in range(n_events)],
        "timestamp": np.random.choice(dates, n_events),
        "user_id": [f"usr_{np.random.randint(1, 50)}" for _ in range(n_events)],
        "product_id": [f"prod_{np.random.randint(1, 20)}" for _ in range(n_events)],
        "price": np.random.uniform(10, 200, n_events),
        "quantity": np.random.randint(1, 5, n_events),
        "location": ["Location A"] * n_events,
        "metadata": ['{"channel": "web"}'] * n_events,
    })

    parquet_path = tmp_path / "events.parquet"
    events.to_parquet(parquet_path)
    return parquet_path


# ============================================================================
# Data Preparation Tests
# ============================================================================


class TestDataPreparation:
    """Tests for data loading and preparation functions."""

    def test_load_and_prepare_produces_dataframe(self, sample_events_data: Path):
        """Test that load_and_prepare_data returns a DataFrame."""
        from models.forecasting.train_forecast import load_and_prepare_data

        df = load_and_prepare_data(sample_events_data)

        assert isinstance(df, pd.DataFrame)
        assert "ds" in df.columns
        assert "y" in df.columns

    def test_daily_aggregation_shape(self, sample_events_data: Path):
        """Test that daily aggregation produces expected number of rows."""
        from models.forecasting.train_forecast import load_and_prepare_data

        df = load_and_prepare_data(sample_events_data)

        # Should have at most 30 unique days
        assert len(df) <= 30
        assert len(df) > 0

    def test_train_val_split(self, sample_time_series: pd.DataFrame):
        """Test train/validation split."""
        from models.forecasting.train_forecast import train_val_split

        train, val = train_val_split(sample_time_series, val_days=7)

        assert len(val) == 7
        assert len(train) == 23  # 30 - 7
        assert train["ds"].max() < val["ds"].min()


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    """Tests for metric calculation functions."""

    def test_calculate_metrics_returns_dict(self):
        """Test that calculate_metrics returns expected dictionary."""
        from models.forecasting.train_forecast import calculate_metrics

        y_true = np.array([100, 110, 105, 95, 100])
        y_pred = np.array([102, 108, 103, 97, 101])

        metrics = calculate_metrics(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert "mape" in metrics
        assert "mse" in metrics

    def test_mape_is_positive(self):
        """Test that MAPE is non-negative."""
        from models.forecasting.train_forecast import calculate_metrics

        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mape"] >= 0

    def test_mse_is_positive(self):
        """Test that MSE is non-negative."""
        from models.forecasting.train_forecast import calculate_metrics

        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["mse"] >= 0

    def test_perfect_predictions_zero_error(self):
        """Test that identical predictions have zero error."""
        from models.forecasting.train_forecast import calculate_metrics

        y = np.array([100, 200, 300])

        metrics = calculate_metrics(y, y)

        assert metrics["mape"] == 0
        assert metrics["mse"] == 0


# ============================================================================
# Model Training Tests
# ============================================================================


class TestProphetModel:
    """Tests for Prophet model training."""

    def test_train_prophet_returns_tuple(self, train_val_data):
        """Test that train_prophet returns expected tuple."""
        from models.forecasting.train_forecast import train_prophet

        train, val = train_val_data

        model, metrics, forecast = train_prophet(train, val)

        assert model is not None
        assert isinstance(metrics, dict)
        assert isinstance(forecast, pd.DataFrame)

    def test_prophet_metrics_valid(self, train_val_data):
        """Test that Prophet metrics are valid numbers."""
        from models.forecasting.train_forecast import train_prophet

        train, val = train_val_data

        _, metrics, _ = train_prophet(train, val)

        assert np.isfinite(metrics["mape"])
        assert np.isfinite(metrics["mse"])

    def test_prophet_forecast_length(self, train_val_data):
        """Test that Prophet forecast has correct length."""
        from models.forecasting.train_forecast import train_prophet

        train, val = train_val_data

        _, _, forecast = train_prophet(train, val)

        # Forecast should include training + validation periods
        assert len(forecast) >= len(val)


class TestLSTMModel:
    """Tests for LSTM model training."""

    def test_train_lstm_returns_tuple(self, train_val_data):
        """Test that train_lstm returns expected tuple."""
        from models.forecasting.train_forecast import train_lstm

        train, val = train_val_data

        model, metrics, predictions = train_lstm(train, val, max_epochs=5)

        assert model is not None
        assert isinstance(metrics, dict)
        assert isinstance(predictions, np.ndarray)

    def test_lstm_predictions_length(self, train_val_data):
        """Test that LSTM predictions match validation length."""
        from models.forecasting.train_forecast import train_lstm

        train, val = train_val_data

        _, _, predictions = train_lstm(train, val, max_epochs=5)

        assert len(predictions) == len(val)


class TestETSModel:
    """Tests for ETS model training."""

    def test_train_ets_returns_tuple(self, train_val_data):
        """Test that train_ets returns expected tuple."""
        from models.forecasting.train_forecast import train_ets

        train, val = train_val_data

        model, metrics, predictions = train_ets(train, val)

        # model can be None if fitting fails (fallback mode)
        assert isinstance(metrics, dict)
        assert isinstance(predictions, np.ndarray)

    def test_ets_predictions_length(self, train_val_data):
        """Test that ETS predictions match validation length."""
        from models.forecasting.train_forecast import train_ets

        train, val = train_val_data

        _, _, predictions = train_ets(train, val)

        assert len(predictions) == len(val)


# ============================================================================
# Training Pipeline Tests
# ============================================================================


class TestTrainingPipeline:
    """Tests for the complete training pipeline."""

    def test_pipeline_completes(self, sample_events_data: Path, tmp_path: Path):
        """Test that training pipeline completes without errors."""
        from models.forecasting.train_forecast import run_training_pipeline

        # Mock the saved models directory
        with patch("models.forecasting.train_forecast.SAVED_MODELS_DIR", tmp_path):
            with patch("models.forecasting.train_forecast.MLFLOW_TRACKING_URI", tmp_path / "mlruns"):
                results = run_training_pipeline(
                    data_path=sample_events_data,
                    val_days=5,
                )

        assert isinstance(results, dict)
        assert "prophet" in results
        assert "lstm" in results
        assert "ets" in results

    def test_pipeline_saves_models(self, sample_events_data: Path, tmp_path: Path):
        """Test that pipeline saves model files."""
        from models.forecasting.train_forecast import run_training_pipeline

        with patch("models.forecasting.train_forecast.SAVED_MODELS_DIR", tmp_path):
            with patch("models.forecasting.train_forecast.MLFLOW_TRACKING_URI", tmp_path / "mlruns"):
                run_training_pipeline(
                    data_path=sample_events_data,
                    val_days=5,
                )

        # Check that model files were created
        assert (tmp_path / "prophet_model.json").exists()
        assert (tmp_path / "lstm_model.pt").exists()
        assert (tmp_path / "ets_config.json").exists()


# ============================================================================
# Forecast Service Tests
# ============================================================================


class TestForecastService:
    """Tests for the prediction service."""

    def test_service_initialization(self):
        """Test that ForecastService can be initialized."""
        from models.forecasting.predict_service import ForecastService

        service = ForecastService(model_type="prophet")

        assert service.model_type == "prophet"
        assert service.model is None

    def test_service_invalid_model_type(self):
        """Test that invalid model type raises error."""
        from models.forecasting.predict_service import ForecastService

        with pytest.raises(ValueError):
            ForecastService(model_type="invalid")

    def test_service_get_model_info(self):
        """Test get_model_info returns expected structure."""
        from models.forecasting.predict_service import ForecastService

        service = ForecastService(model_type="lstm")
        info = service.get_model_info()

        assert isinstance(info, dict)
        assert "model_type" in info
        assert "is_loaded" in info

    def test_forecast_length_matches_date_range(
        self, sample_events_data: Path, tmp_path: Path
    ):
        """Test that forecast length matches requested date range."""
        from models.forecasting.train_forecast import run_training_pipeline
        from models.forecasting.predict_service import ForecastService

        # Train models first
        with patch("models.forecasting.train_forecast.SAVED_MODELS_DIR", tmp_path):
            with patch("models.forecasting.train_forecast.MLFLOW_TRACKING_URI", tmp_path / "mlruns"):
                run_training_pipeline(data_path=sample_events_data, val_days=5)

        # Test prediction service
        with patch("models.forecasting.predict_service.SAVED_MODELS_DIR", tmp_path):
            service = ForecastService(model_type="prophet")
            service.model_path = tmp_path / "prophet_model.json"
            service.load_model()

            forecast = service.predict(
                start_date="2024-02-01",
                end_date="2024-02-07",
            )

        assert len(forecast) == 7  # 7 days inclusive


# ============================================================================
# Optuna Study Tests
# ============================================================================


class TestOptunaStudy:
    """Tests for Optuna hyperparameter optimization."""

    def test_prophet_objective_returns_float(self, train_val_data):
        """Test that Prophet objective returns a float."""
        from models.forecasting.optuna_study import create_prophet_objective

        train, val = train_val_data

        objective = create_prophet_objective(train, val)

        # Create mock trial
        trial = MagicMock()
        trial.suggest_categorical = MagicMock(side_effect=["multiplicative", True, False])
        trial.suggest_float = MagicMock(return_value=0.05)

        result = objective(trial)

        assert isinstance(result, float)

    def test_run_study_completes(self, sample_events_data: Path, tmp_path: Path):
        """Test that Optuna study completes."""
        from models.forecasting.optuna_study import run_optuna_study

        with patch("models.forecasting.train_forecast.DATA_PATH", sample_events_data):
            with patch("models.forecasting.optuna_study.MLFLOW_TRACKING_URI", tmp_path / "mlruns"):
                study = run_optuna_study(
                    model_type="ets",  # ETS is fastest
                    n_trials=2,
                    val_days=5,
                    log_to_mlflow=False,
                )

        assert study is not None
        assert len(study.trials) == 2


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_metrics_with_zeros(self):
        """Test metrics calculation with zero values."""
        from models.forecasting.train_forecast import calculate_metrics

        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 110, 210])

        # Should not raise, but may have inf MAPE for zero values
        metrics = calculate_metrics(y_true, y_pred)

        assert isinstance(metrics["mse"], float)

    def test_small_dataset_handling(self):
        """Test that models handle very small datasets gracefully."""
        from models.forecasting.train_forecast import train_ets

        # Only 5 points
        train = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=5),
            "y": [100, 102, 98, 105, 100],
        })
        val = pd.DataFrame({
            "ds": pd.date_range("2024-01-06", periods=2),
            "y": [103, 101],
        })

        # Should not raise
        model, metrics, preds = train_ets(train, val)

        assert len(preds) == 2
