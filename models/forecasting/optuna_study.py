"""
Optuna Hyperparameter Optimization for Forecasting Models.

Provides hyperparameter tuning for Prophet, LSTM, and ETS models
using Optuna's efficient search algorithms.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import pandas as pd
import numpy as np
import optuna
from optuna.trial import Trial
import mlflow

from models.forecasting.train_forecast import (
    load_and_prepare_data,
    train_val_split,
    train_prophet,
    train_lstm,
    train_ets,
    calculate_metrics,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
)

warnings.filterwarnings("ignore")


# ============================================================================
# Objective Functions
# ============================================================================


def create_prophet_objective(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Callable[[Trial], float]:
    """
    Create Optuna objective function for Prophet hyperparameter tuning.

    Args:
        train_df: Training data.
        val_df: Validation data.

    Returns:
        Objective function that returns MAPE.
    """

    def objective(trial: Trial) -> float:
        params = {
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.5, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.01, 10.0, log=True
            ),
            "weekly_seasonality": trial.suggest_categorical(
                "weekly_seasonality", [True, False]
            ),
            "daily_seasonality": False,
            "yearly_seasonality": False,
        }

        try:
            _, metrics, _ = train_prophet(train_df, val_df, hyperparams=params)
            return metrics["mape"]
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    return objective


def create_lstm_objective(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Callable[[Trial], float]:
    """
    Create Optuna objective function for LSTM hyperparameter tuning.

    Args:
        train_df: Training data.
        val_df: Validation data.

    Returns:
        Objective function that returns MAPE.
    """

    def objective(trial: Trial) -> float:
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "seq_length": trial.suggest_int("seq_length", 3, 10),
        }

        try:
            _, metrics, _ = train_lstm(
                train_df, val_df, hyperparams=params, max_epochs=30
            )
            return metrics["mape"]
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    return objective


def create_ets_objective(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Callable[[Trial], float]:
    """
    Create Optuna objective function for ETS hyperparameter tuning.

    Args:
        train_df: Training data.
        val_df: Validation data.

    Returns:
        Objective function that returns MAPE.
    """

    def objective(trial: Trial) -> float:
        params = {
            "trend": trial.suggest_categorical("trend", ["add", "mul", None]),
            "damped_trend": trial.suggest_categorical("damped_trend", [True, False]),
            "seasonal": None,  # Not enough data for seasonal in sample
        }

        # Can't have damped with no trend
        if params["trend"] is None:
            params["damped_trend"] = False

        try:
            _, metrics, _ = train_ets(train_df, val_df, hyperparams=params)
            return metrics["mape"]
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

    return objective


# ============================================================================
# Study Runner
# ============================================================================


def run_optuna_study(
    model_type: str = "prophet",
    n_trials: int = 20,
    val_days: int = 7,
    data_path: Optional[Path] = None,
    log_to_mlflow: bool = True,
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization study.

    Args:
        model_type: Model to tune ('prophet', 'lstm', or 'ets').
        n_trials: Number of optimization trials.
        val_days: Days to use for validation.
        data_path: Optional path to data file.
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Completed Optuna study.
    """
    print("=" * 60)
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION - {model_type.upper()}")
    print("=" * 60)

    # Load data
    df = load_and_prepare_data(data_path)
    train_df, val_df = train_val_split(df, val_days=val_days)

    print(f"Training set: {len(train_df)} days")
    print(f"Validation set: {len(val_df)} days")
    print(f"Running {n_trials} trials...")
    print("-" * 40)

    # Create objective
    if model_type == "prophet":
        objective = create_prophet_objective(train_df, val_df)
    elif model_type == "lstm":
        objective = create_lstm_objective(train_df, val_df)
    elif model_type == "ets":
        objective = create_ets_objective(train_df, val_df)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create and run study
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{model_type}_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest MAPE: {study.best_value:.2f}%")
    print(f"\nBest Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Log to MLflow
    if log_to_mlflow:
        mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))
        mlflow.set_experiment(f"{EXPERIMENT_NAME}-optuna")

        with mlflow.start_run(run_name=f"{model_type}_optuna_best"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_mape", study.best_value)
            mlflow.log_metric("n_trials", n_trials)

            # Log study statistics
            mlflow.log_metric("n_completed_trials", len(study.trials))

        print(f"\nResults logged to MLflow: {MLFLOW_TRACKING_URI}")

    return study


def run_all_studies(
    n_trials: int = 10,
    val_days: int = 7,
) -> Dict[str, optuna.Study]:
    """
    Run Optuna optimization for all model types.

    Args:
        n_trials: Number of trials per model.
        val_days: Days for validation.

    Returns:
        Dictionary of studies by model type.
    """
    studies = {}

    for model_type in ["prophet", "lstm", "ets"]:
        print(f"\n{'#' * 60}")
        print(f"# Optimizing {model_type.upper()}")
        print(f"{'#' * 60}\n")

        studies[model_type] = run_optuna_study(
            model_type=model_type,
            n_trials=n_trials,
            val_days=val_days,
        )

    # Summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    for model_type, study in studies.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Best MAPE: {study.best_value:.2f}%")
        print(f"  Best params: {study.best_params}")

    best_overall = min(studies.items(), key=lambda x: x[1].best_value)
    print(f"\n🏆 Best Model: {best_overall[0].upper()} with MAPE {best_overall[1].best_value:.2f}%")

    return studies


# ============================================================================
# Main (CLI)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=["prophet", "lstm", "ets", "all"],
        help="Model type to optimize (or 'all')",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=7,
        help="Days for validation split",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )

    args = parser.parse_args()

    if args.model_type == "all":
        run_all_studies(
            n_trials=args.n_trials,
            val_days=args.val_days,
        )
    else:
        run_optuna_study(
            model_type=args.model_type,
            n_trials=args.n_trials,
            val_days=args.val_days,
            log_to_mlflow=not args.no_mlflow,
        )
