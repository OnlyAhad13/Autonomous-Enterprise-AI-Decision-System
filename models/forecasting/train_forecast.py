"""
Forecasting Training Pipeline.

Trains Prophet, LSTM (PyTorch Lightning), and ETS models on time series data.
Evaluates using MAPE and MSE, logs experiments to MLflow.
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List

import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
import mlflow.prophet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ============================================================================
# Constants
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample" / "events.parquet"
SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "forecasting" / "saved"
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "forecasting-pipeline"


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================


def load_and_prepare_data(
    data_path: Optional[Path] = None,
    target_column: str = "revenue",
    date_column: str = "ds",
) -> pd.DataFrame:
    """
    Load events data and aggregate to daily time series.

    Args:
        data_path: Path to parquet file. Defaults to sample events.
        target_column: Name of the target column to forecast.
        date_column: Name of the date column (Prophet requires 'ds').

    Returns:
        DataFrame with columns ['ds', 'y'] for Prophet compatibility.
    """
    data_path = data_path or DATA_PATH

    # Load raw events
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["revenue"] = df["price"] * df["quantity"]

    # Aggregate to daily revenue
    daily = (
        df.groupby(df["timestamp"].dt.date)
        .agg({"revenue": "sum", "id": "count"})
        .reset_index()
    )
    daily.columns = ["ds", "y", "transaction_count"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)

    print(f"Loaded {len(daily)} days of data")
    print(f"Date range: {daily['ds'].min()} to {daily['ds'].max()}")
    print(f"Mean daily revenue: ${daily['y'].mean():,.2f}")

    return daily


def train_val_split(
    df: pd.DataFrame, val_days: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series into train and validation sets."""
    split_idx = len(df) - val_days
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    return train_df, val_df


# ============================================================================
# Metrics
# ============================================================================


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate MAPE and MSE metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with 'mape' and 'mse' keys.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Avoid division by zero in MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("inf")

    mse = np.mean((y_true - y_pred) ** 2)

    return {"mape": float(mape), "mse": float(mse)}


# ============================================================================
# Prophet Model
# ============================================================================


def train_prophet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Tuple[Prophet, Dict[str, float], pd.DataFrame]:
    """
    Train Facebook Prophet model.

    Args:
        train_df: Training data with 'ds' and 'y' columns.
        val_df: Validation data.
        hyperparams: Optional hyperparameters for Prophet.

    Returns:
        Tuple of (trained model, metrics dict, forecast DataFrame).
    """
    hyperparams = hyperparams or {
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": False,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05,
    }

    model = Prophet(**hyperparams)
    model.fit(train_df[["ds", "y"]])

    # Generate forecast for validation period
    future = model.make_future_dataframe(periods=len(val_df))
    forecast = model.predict(future)

    # Extract validation predictions
    val_forecast = forecast.tail(len(val_df))
    y_pred = val_forecast["yhat"].values
    y_true = val_df["y"].values

    metrics = calculate_metrics(y_true, y_pred)
    metrics["model_type"] = "prophet"

    return model, metrics, forecast


# ============================================================================
# LSTM Model (PyTorch Lightning)
# ============================================================================


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""

    def __init__(self, data: np.ndarray, seq_length: int = 7):
        self.data = data
        self.seq_length = seq_length

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMForecaster(pl.LightningModule):
    """LSTM-based time series forecasting model."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
    max_epochs: int = 50,
) -> Tuple[LSTMForecaster, Dict[str, float], np.ndarray]:
    """
    Train LSTM model using PyTorch Lightning.

    Args:
        train_df: Training data with 'y' column.
        val_df: Validation data.
        hyperparams: Model hyperparameters.
        max_epochs: Maximum training epochs.

    Returns:
        Tuple of (trained model, metrics dict, predictions array).
    """
    hyperparams = hyperparams or {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "seq_length": 7,
    }

    seq_length = hyperparams.pop("seq_length", 7)

    # Prepare data
    full_data = pd.concat([train_df, val_df])["y"].values
    mean_val = full_data.mean()
    std_val = full_data.std()
    normalized = (full_data - mean_val) / (std_val + 1e-8)

    train_size = len(train_df)
    train_normalized = normalized[:train_size]

    # Create datasets
    train_dataset = TimeSeriesDataset(train_normalized, seq_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Handle edge case: not enough data for sequences
    if len(train_dataset) == 0:
        print("[WARN] Not enough data for LSTM training, using fallback predictions")
        model = LSTMForecaster(**hyperparams)
        y_pred = np.full(len(val_df), train_df["y"].mean())
        metrics = calculate_metrics(val_df["y"].values, y_pred)
        metrics["model_type"] = "lstm"
        return model, metrics, y_pred

    # Initialize model
    model = LSTMForecaster(**hyperparams)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="train_loss", patience=10, mode="min"),
    ]

    # Train
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        accelerator="auto",
    )
    trainer.fit(model, train_loader)

    # Generate predictions for validation set
    model.eval()
    predictions = []

    with torch.no_grad():
        # Use rolling prediction
        current_seq = torch.FloatTensor(normalized[train_size - seq_length : train_size])
        for _ in range(len(val_df)):
            pred = model(current_seq.unsqueeze(0))
            predictions.append(pred.item())
            # Roll and append
            current_seq = torch.cat([current_seq[1:], pred.flatten()])

    # Denormalize predictions
    y_pred = np.array(predictions) * (std_val + 1e-8) + mean_val
    y_true = val_df["y"].values

    metrics = calculate_metrics(y_true, y_pred)
    metrics["model_type"] = "lstm"

    return model, metrics, y_pred


# ============================================================================
# ETS Model (Exponential Smoothing)
# ============================================================================


def train_ets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, float], np.ndarray]:
    """
    Train ETS (Exponential Smoothing) model.

    Args:
        train_df: Training data with 'y' column.
        val_df: Validation data.
        hyperparams: ETS hyperparameters.

    Returns:
        Tuple of (trained model, metrics dict, predictions array).
    """
    hyperparams = hyperparams or {
        "trend": "add",
        "seasonal": None,  # Not enough data for seasonal
        "damped_trend": True,
    }

    y_train = train_df["y"].values

    try:
        model = ExponentialSmoothing(
            y_train,
            trend=hyperparams.get("trend", "add"),
            seasonal=hyperparams.get("seasonal"),
            damped_trend=hyperparams.get("damped_trend", True),
        )
        fitted = model.fit(optimized=True)
        y_pred = fitted.forecast(len(val_df))
    except Exception as e:
        print(f"[WARN] ETS fitting failed: {e}. Using simple mean fallback.")
        fitted = None
        y_pred = np.full(len(val_df), y_train.mean())

    y_true = val_df["y"].values
    metrics = calculate_metrics(y_true, y_pred)
    metrics["model_type"] = "ets"

    return fitted, metrics, y_pred


# ============================================================================
# Training Pipeline
# ============================================================================


def run_training_pipeline(
    data_path: Optional[Path] = None,
    val_days: int = 7,
    prophet_params: Optional[Dict] = None,
    lstm_params: Optional[Dict] = None,
    ets_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run the complete forecasting training pipeline.

    Trains Prophet, LSTM, and ETS models, evaluates them, and logs to MLflow.

    Args:
        data_path: Path to data file.
        val_days: Number of days for validation.
        prophet_params: Prophet hyperparameters.
        lstm_params: LSTM hyperparameters.
        ets_params: ETS hyperparameters.

    Returns:
        Dictionary with results for each model.
    """
    # Setup MLflow
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Ensure saved models directory exists
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("=" * 60)
    print("FORECASTING TRAINING PIPELINE")
    print("=" * 60)

    df = load_and_prepare_data(data_path)
    train_df, val_df = train_val_split(df, val_days=val_days)

    print(f"\nTraining set: {len(train_df)} days")
    print(f"Validation set: {len(val_df)} days")

    results = {}

    # ------------------------------------
    # Train Prophet
    # ------------------------------------
    print("\n" + "-" * 40)
    print("Training Prophet...")
    print("-" * 40)

    with mlflow.start_run(run_name="prophet_forecast"):
        prophet_model, prophet_metrics, prophet_forecast = train_prophet(
            train_df, val_df, prophet_params
        )
        mlflow.log_metrics(
            {"prophet_mape": prophet_metrics["mape"], "prophet_mse": prophet_metrics["mse"]}
        )
        mlflow.log_params(prophet_params or {"default": True})

        # Save model
        prophet_path = SAVED_MODELS_DIR / "prophet_model.json"
        with open(prophet_path, "w") as f:
            from prophet.serialize import model_to_json
            f.write(model_to_json(prophet_model))

        mlflow.log_artifact(str(prophet_path))
        results["prophet"] = {
            "metrics": prophet_metrics,
            "model_path": str(prophet_path),
        }
        print(f"Prophet MAPE: {prophet_metrics['mape']:.2f}%")
        print(f"Prophet MSE: {prophet_metrics['mse']:,.2f}")

    # ------------------------------------
    # Train LSTM
    # ------------------------------------
    print("\n" + "-" * 40)
    print("Training LSTM...")
    print("-" * 40)

    with mlflow.start_run(run_name="lstm_forecast"):
        lstm_model, lstm_metrics, lstm_preds = train_lstm(
            train_df, val_df, lstm_params
        )
        mlflow.log_metrics(
            {"lstm_mape": lstm_metrics["mape"], "lstm_mse": lstm_metrics["mse"]}
        )
        mlflow.log_params(lstm_params or {"default": True})

        # Save model
        lstm_path = SAVED_MODELS_DIR / "lstm_model.pt"
        torch.save(lstm_model.state_dict(), lstm_path)

        mlflow.log_artifact(str(lstm_path))
        results["lstm"] = {
            "metrics": lstm_metrics,
            "model_path": str(lstm_path),
        }
        print(f"LSTM MAPE: {lstm_metrics['mape']:.2f}%")
        print(f"LSTM MSE: {lstm_metrics['mse']:,.2f}")

    # ------------------------------------
    # Train ETS
    # ------------------------------------
    print("\n" + "-" * 40)
    print("Training ETS...")
    print("-" * 40)

    with mlflow.start_run(run_name="ets_forecast"):
        ets_model, ets_metrics, ets_preds = train_ets(train_df, val_df, ets_params)
        mlflow.log_metrics(
            {"ets_mape": ets_metrics["mape"], "ets_mse": ets_metrics["mse"]}
        )
        mlflow.log_params(ets_params or {"default": True})

        # Save ETS config (model is recreated from data + params)
        ets_config = {"params": ets_params or {}, "train_mean": float(train_df["y"].mean())}
        ets_path = SAVED_MODELS_DIR / "ets_config.json"
        with open(ets_path, "w") as f:
            json.dump(ets_config, f)

        mlflow.log_artifact(str(ets_path))
        results["ets"] = {
            "metrics": ets_metrics,
            "model_path": str(ets_path),
        }
        print(f"ETS MAPE: {ets_metrics['mape']:.2f}%")
        print(f"ETS MSE: {ets_metrics['mse']:,.2f}")

    # ------------------------------------
    # Summary
    # ------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_model = min(results.items(), key=lambda x: x[1]["metrics"]["mape"])
    print(f"\nBest Model: {best_model[0].upper()}")
    print(f"Best MAPE: {best_model[1]['metrics']['mape']:.2f}%")
    print(f"\nModels saved to: {SAVED_MODELS_DIR}")
    print(f"MLflow experiments: {MLFLOW_TRACKING_URI}")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = run_training_pipeline()
