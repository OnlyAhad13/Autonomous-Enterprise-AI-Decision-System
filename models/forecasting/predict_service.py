"""
Forecast Prediction Service.

Provides a unified interface for loading trained forecasting models
and generating predictions for a specified date range.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Union, List

import pandas as pd
import numpy as np
import torch

from prophet import Prophet
from prophet.serialize import model_from_json

from models.forecasting.train_forecast import (
    LSTMForecaster,
    load_and_prepare_data,
    DATA_PATH,
    SAVED_MODELS_DIR,
)


class ForecastService:
    """
    Service for loading trained models and generating forecasts.

    Supports Prophet, LSTM, and ETS model types.
    """

    def __init__(self, model_type: str = "prophet", model_path: Optional[Path] = None):
        """
        Initialize the forecast service.

        Args:
            model_type: Type of model ('prophet', 'lstm', or 'ets').
            model_path: Optional custom path to model file.
        """
        self.model_type = model_type.lower()
        self.model_path = model_path or self._default_model_path()
        self.model = None
        self._data_stats: Dict[str, float] = {}

        if self.model_type not in ["prophet", "lstm", "ets"]:
            raise ValueError(f"Unknown model type: {model_type}")

    def _default_model_path(self) -> Path:
        """Get default model path based on model type."""
        if self.model_type == "prophet":
            return SAVED_MODELS_DIR / "prophet_model.json"
        elif self.model_type == "lstm":
            return SAVED_MODELS_DIR / "lstm_model.pt"
        elif self.model_type == "ets":
            return SAVED_MODELS_DIR / "ets_config.json"
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def load_model(self) -> None:
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        if self.model_type == "prophet":
            with open(self.model_path, "r") as f:
                self.model = model_from_json(f.read())

        elif self.model_type == "lstm":
            # Load model architecture and weights
            self.model = LSTMForecaster()
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()

            # Load data stats for normalization
            self._load_data_stats()

        elif self.model_type == "ets":
            # ETS stores just config; forecasts use mean fallback
            with open(self.model_path, "r") as f:
                config = json.load(f)
                self._data_stats["train_mean"] = config.get("train_mean", 0)
                self.model = config  # Store config as "model"

        print(f"Loaded {self.model_type.upper()} model from: {self.model_path}")

    def _load_data_stats(self) -> None:
        """Load data statistics for LSTM normalization."""
        try:
            df = load_and_prepare_data()
            self._data_stats["mean"] = df["y"].mean()
            self._data_stats["std"] = df["y"].std()
        except Exception:
            self._data_stats["mean"] = 0
            self._data_stats["std"] = 1

    def predict(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        include_history: bool = False,
    ) -> pd.DataFrame:
        """
        Generate forecasts for a date range.

        Args:
            start_date: Start date for forecast.
            end_date: End date for forecast.
            include_history: Whether to include historical data in output.

        Returns:
            DataFrame with columns ['ds', 'yhat'] at minimum.
        """
        if self.model is None:
            self.load_model()

        # Parse dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        horizon = len(date_range)

        if self.model_type == "prophet":
            return self._predict_prophet(date_range, include_history)
        elif self.model_type == "lstm":
            return self._predict_lstm(date_range)
        elif self.model_type == "ets":
            return self._predict_ets(date_range)

    def _predict_prophet(
        self, date_range: pd.DatetimeIndex, include_history: bool
    ) -> pd.DataFrame:
        """Generate Prophet predictions."""
        future = pd.DataFrame({"ds": date_range})
        forecast = self.model.predict(future)

        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result = result.rename(
            columns={
                "yhat": "forecast",
                "yhat_lower": "lower_bound",
                "yhat_upper": "upper_bound",
            }
        )
        return result

    def _predict_lstm(self, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate LSTM predictions."""
        # Load recent data for sequence input
        try:
            df = load_and_prepare_data()
            recent_data = df["y"].values[-7:]  # Last 7 days as input
        except Exception:
            recent_data = np.zeros(7)

        # Normalize
        mean_val = self._data_stats.get("mean", 0)
        std_val = self._data_stats.get("std", 1)
        normalized = (recent_data - mean_val) / (std_val + 1e-8)

        # Generate predictions
        predictions = []
        current_seq = torch.FloatTensor(normalized)

        with torch.no_grad():
            for _ in range(len(date_range)):
                pred = self.model(current_seq.unsqueeze(0))
                predictions.append(pred.item())
                current_seq = torch.cat([current_seq[1:], pred.flatten()])

        # Denormalize
        forecasts = np.array(predictions) * (std_val + 1e-8) + mean_val

        return pd.DataFrame({
            "ds": date_range,
            "forecast": forecasts,
        })

    def _predict_ets(self, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate ETS predictions (simple mean-based fallback)."""
        train_mean = self._data_stats.get("train_mean", 0)
        forecasts = np.full(len(date_range), train_mean)

        return pd.DataFrame({
            "ds": date_range,
            "forecast": forecasts,
        })

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "is_loaded": self.model is not None,
            "data_stats": self._data_stats,
        }


def forecast_for_range(
    model_type: str,
    start_date: str,
    end_date: str,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate forecasts.

    Args:
        model_type: Type of model ('prophet', 'lstm', or 'ets').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        model_path: Optional path to model file.

    Returns:
        DataFrame with forecast results.
    """
    service = ForecastService(
        model_type=model_type,
        model_path=Path(model_path) if model_path else None,
    )
    return service.predict(start_date, end_date)


# ============================================================================
# Main (CLI)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate forecasts")
    parser.add_argument(
        "--model-type",
        type=str,
        default="prophet",
        choices=["prophet", "lstm", "ets"],
        help="Model type to use",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        help="Forecast start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
        help="Forecast end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to model file",
    )

    args = parser.parse_args()

    print(f"Generating {args.model_type.upper()} forecast...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print("-" * 40)

    result = forecast_for_range(
        model_type=args.model_type,
        start_date=args.start_date,
        end_date=args.end_date,
        model_path=args.model_path,
    )

    print(result.to_string(index=False))
