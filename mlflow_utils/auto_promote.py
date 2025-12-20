"""
MLflow Auto-Promotion Module.

Automatically promotes ML models based on metric improvements compared to
production baseline. Triggers deployment webhooks on promotion.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml
import requests
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "mlflow_local.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PromotionThresholds:
    """Thresholds for model promotion decisions."""
    mape_improvement_pct: float = 5.0
    mse_improvement_pct: float = 10.0
    min_samples: int = 100


@dataclass
class PromotionResult:
    """Result of a promotion evaluation."""
    promoted: bool
    candidate_run_id: str
    candidate_metrics: Dict[str, float]
    baseline_metrics: Optional[Dict[str, float]]
    improvements: Dict[str, float]
    reason: str
    model_version: Optional[str] = None
    webhook_triggered: bool = False
    webhook_response: Optional[Dict] = None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = config_path or DEFAULT_CONFIG_PATH
    
    if not path.exists():
        logger.warning(f"Config not found at {path}, using defaults")
        return {}
    
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    config = _expand_env_vars(config)
    return config


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in config."""
    if isinstance(obj, str):
        if obj.startswith("${") and ":-" in obj:
            # Handle default values: ${VAR:-default}
            var_part = obj[2:-1]
            var_name, default = var_part.split(":-", 1)
            return os.environ.get(var_name, default)
        elif obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


# ============================================================================
# Auto-Promoter Class
# ============================================================================


class AutoPromoter:
    """
    Automated model promotion based on metric comparisons.
    
    Compares candidate runs against production baseline and promotes
    if improvements exceed configured thresholds.
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        tracking_uri: Optional[str] = None,
        thresholds: Optional[PromotionThresholds] = None,
    ):
        """
        Initialize the AutoPromoter.
        
        Args:
            config_path: Path to YAML config file.
            tracking_uri: MLflow tracking URI (overrides config).
            thresholds: Custom promotion thresholds.
        """
        self.config = load_config(config_path)
        
        # Set tracking URI
        self.tracking_uri = tracking_uri or self.config.get("tracking", {}).get(
            "uri", str(PROJECT_ROOT / "mlruns")
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
        
        # Set thresholds
        if thresholds:
            self.thresholds = thresholds
        else:
            thresh_config = self.config.get("promotion", {}).get("thresholds", {})
            self.thresholds = PromotionThresholds(
                mape_improvement_pct=thresh_config.get("mape_improvement_pct", 5.0),
                mse_improvement_pct=thresh_config.get("mse_improvement_pct", 10.0),
                min_samples=thresh_config.get("min_samples", 100),
            )
        
        # Webhook config
        self.webhook_config = self.config.get("webhook", {})
        
        logger.info(f"AutoPromoter initialized with tracking URI: {self.tracking_uri}")
    
    def get_new_runs(
        self,
        experiment_name: str,
        since: Optional[datetime] = None,
        status: str = "FINISHED",
    ) -> List[mlflow.entities.Run]:
        """
        Get new runs from an experiment.
        
        Args:
            experiment_name: Name of the MLflow experiment.
            since: Only return runs after this timestamp.
            status: Filter by run status.
            
        Returns:
            List of MLflow Run objects.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return []
        
        # Build filter string
        filter_string = f"status = '{status}'"
        if since:
            timestamp_ms = int(since.timestamp() * 1000)
            filter_string += f" and start_time > {timestamp_ms}"
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"],
        )
        
        logger.info(f"Found {len(runs)} runs in experiment '{experiment_name}'")
        return runs
    
    def get_production_baseline(
        self,
        model_name: str,
    ) -> Optional[Dict[str, float]]:
        """
        Get metrics from the current production model.
        
        Args:
            model_name: Registered model name.
            
        Returns:
            Dictionary of metrics or None if no production model.
        """
        try:
            # Get production model version
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            
            if not versions:
                logger.info(f"No production model found for '{model_name}'")
                return None
            
            prod_version = versions[0]
            run_id = prod_version.run_id
            
            # Get metrics from the run
            run = self.client.get_run(run_id)
            metrics = run.data.metrics
            
            logger.info(f"Production baseline metrics: {metrics}")
            return metrics
            
        except mlflow.exceptions.MlflowException as e:
            logger.warning(f"Could not get production baseline: {e}")
            return None
    
    def compare_metrics(
        self,
        candidate_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> tuple[bool, Dict[str, float], str]:
        """
        Compare candidate metrics against baseline.
        
        Args:
            candidate_metrics: Metrics from candidate run.
            baseline_metrics: Metrics from production baseline.
            
        Returns:
            Tuple of (should_promote, improvements_dict, reason_str).
        """
        improvements = {}
        reasons = []
        should_promote = False
        
        # Check MAPE improvement (lower is better)
        for key in candidate_metrics:
            if "mape" in key.lower():
                baseline_val = baseline_metrics.get(key, float("inf"))
                candidate_val = candidate_metrics[key]
                
                if baseline_val > 0:
                    improvement_pct = ((baseline_val - candidate_val) / baseline_val) * 100
                    improvements[f"{key}_improvement_pct"] = improvement_pct
                    
                    if improvement_pct >= self.thresholds.mape_improvement_pct:
                        should_promote = True
                        reasons.append(
                            f"{key} improved by {improvement_pct:.2f}% "
                            f"(threshold: {self.thresholds.mape_improvement_pct}%)"
                        )
        
        # Check MSE improvement (lower is better)
        for key in candidate_metrics:
            if "mse" in key.lower():
                baseline_val = baseline_metrics.get(key, float("inf"))
                candidate_val = candidate_metrics[key]
                
                if baseline_val > 0:
                    improvement_pct = ((baseline_val - candidate_val) / baseline_val) * 100
                    improvements[f"{key}_improvement_pct"] = improvement_pct
                    
                    if improvement_pct >= self.thresholds.mse_improvement_pct:
                        should_promote = True
                        reasons.append(
                            f"{key} reduced by {improvement_pct:.2f}% "
                            f"(threshold: {self.thresholds.mse_improvement_pct}%)"
                        )
        
        reason = "; ".join(reasons) if reasons else "No significant improvement"
        return should_promote, improvements, reason
    
    def promote_to_staging(
        self,
        model_name: str,
        run_id: str,
        artifact_path: str = "model",
    ) -> Optional[ModelVersion]:
        """
        Register model and transition to Staging.
        
        Args:
            model_name: Name for the registered model.
            run_id: Run ID containing the model artifact.
            artifact_path: Path to model artifact within the run.
            
        Returns:
            ModelVersion object or None if failed.
        """
        try:
            # Register the model
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            
            logger.info(f"Registered model '{model_name}' version {version}")
            
            # Transition to staging
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            
            logger.info(f"Transitioned '{model_name}' v{version} to Staging")
            
            return self.client.get_model_version(model_name, version)
            
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to promote model: {e}")
            return None
    
    def trigger_webhook(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float],
        run_id: str,
    ) -> Optional[Dict]:
        """
        Trigger deployment webhook.
        
        Args:
            model_name: Name of the model.
            model_version: Version number.
            metrics: Model metrics.
            run_id: MLflow run ID.
            
        Returns:
            Response data or None if disabled/failed.
        """
        if not self.webhook_config.get("enabled", False):
            logger.info("Webhook disabled, skipping")
            return None
        
        url = self.webhook_config.get("url")
        if not url:
            logger.warning("Webhook URL not configured")
            return None
        
        payload = {
            "event": "model_promoted_to_staging",
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "source": "mlflow-auto-promoter",
        }
        
        headers = {"Content-Type": "application/json"}
        
        # Add authentication
        auth_config = self.webhook_config.get("auth", {})
        if auth_config.get("type") == "bearer":
            token = auth_config.get("token", "")
            headers["Authorization"] = f"Bearer {token}"
        
        timeout = self.webhook_config.get("timeout_seconds", 30)
        retries = self.webhook_config.get("retry_attempts", 3)
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                
                logger.info(f"Webhook triggered successfully: {response.status_code}")
                return {"status_code": response.status_code, "body": response.text}
                
            except requests.RequestException as e:
                logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    import time
                    delay = self.webhook_config.get("retry_delay_seconds", 5)
                    time.sleep(delay)
        
        logger.error(f"Webhook failed after {retries} attempts")
        return None
    
    def evaluate_and_promote(
        self,
        experiment_name: str,
        model_name: str,
        artifact_path: str = "model",
        since: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> List[PromotionResult]:
        """
        Evaluate new runs and promote if criteria met.
        
        Args:
            experiment_name: MLflow experiment name.
            model_name: Registered model name.
            artifact_path: Path to model artifact.
            since: Only evaluate runs after this time.
            dry_run: If True, don't actually promote.
            
        Returns:
            List of PromotionResult for each evaluated run.
        """
        results = []
        
        # Get new runs
        runs = self.get_new_runs(experiment_name, since=since)
        if not runs:
            logger.info("No new runs to evaluate")
            return results
        
        # Get production baseline
        baseline = self.get_production_baseline(model_name)
        
        for run in runs:
            candidate_metrics = run.data.metrics
            
            # Skip if no baseline (first model goes straight to staging)
            if baseline is None:
                should_promote = True
                improvements = {}
                reason = "No production baseline - first model"
            else:
                should_promote, improvements, reason = self.compare_metrics(
                    candidate_metrics, baseline
                )
            
            result = PromotionResult(
                promoted=False,
                candidate_run_id=run.info.run_id,
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline,
                improvements=improvements,
                reason=reason,
            )
            
            if should_promote:
                if dry_run:
                    logger.info(f"[DRY RUN] Would promote run {run.info.run_id}: {reason}")
                    result.reason = f"[DRY RUN] {reason}"
                else:
                    # Promote to staging
                    version = self.promote_to_staging(
                        model_name, run.info.run_id, artifact_path
                    )
                    
                    if version:
                        result.promoted = True
                        result.model_version = version.version
                        
                        # Trigger webhook
                        webhook_response = self.trigger_webhook(
                            model_name,
                            version.version,
                            candidate_metrics,
                            run.info.run_id,
                        )
                        result.webhook_triggered = webhook_response is not None
                        result.webhook_response = webhook_response
            else:
                logger.info(f"Run {run.info.run_id} did not meet promotion criteria: {reason}")
            
            results.append(result)
        
        return results


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """CLI entry point for auto-promotion."""
    parser = argparse.ArgumentParser(
        description="Automatically promote ML models based on metric improvements"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Registered model name",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=24,
        help="Only evaluate runs from last N hours",
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Path to model artifact within run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate without actually promoting",
    )
    parser.add_argument(
        "--mape-threshold",
        type=float,
        default=None,
        help="Override MAPE improvement threshold (%)",
    )
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=None,
        help="Override MSE improvement threshold (%)",
    )
    
    args = parser.parse_args()
    
    # Build thresholds
    thresholds = None
    if args.mape_threshold or args.mse_threshold:
        thresholds = PromotionThresholds(
            mape_improvement_pct=args.mape_threshold or 5.0,
            mse_improvement_pct=args.mse_threshold or 10.0,
        )
    
    # Initialize promoter
    promoter = AutoPromoter(
        config_path=args.config,
        tracking_uri=args.tracking_uri,
        thresholds=thresholds,
    )
    
    # Calculate since time
    since = datetime.now() - timedelta(hours=args.since_hours)
    
    # Run evaluation
    results = promoter.evaluate_and_promote(
        experiment_name=args.experiment,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        since=since,
        dry_run=args.dry_run,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("AUTO-PROMOTION RESULTS")
    print("=" * 60)
    
    for result in results:
        status = "✓ PROMOTED" if result.promoted else "✗ NOT PROMOTED"
        print(f"\nRun: {result.candidate_run_id[:8]}...")
        print(f"  Status: {status}")
        print(f"  Reason: {result.reason}")
        if result.model_version:
            print(f"  Version: {result.model_version}")
        if result.webhook_triggered:
            print(f"  Webhook: Triggered")
    
    if not results:
        print("\nNo runs to evaluate.")
    
    return 0 if all(r.promoted for r in results if results) else 1


if __name__ == "__main__":
    exit(main())
