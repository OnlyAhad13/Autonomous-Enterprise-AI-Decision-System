"""
MLflow Tool Wrapper.

Provides functions to interact with MLflow for experiment tracking and model registry.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import for MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


@dataclass
class ToolResult:
    """Standard result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    requires_confirmation: bool = False
    action_description: Optional[str] = None


class MLflowTool:
    """
    Tool wrapper for MLflow operations.
    
    Provides safe functions for LLM agents to interact with MLflow.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow tool.
        
        Args:
            tracking_uri: MLflow tracking server URI. Defaults to local mlruns.
        """
        if not HAS_MLFLOW:
            raise ImportError("mlflow not installed. Run: pip install mlflow")
        
        self.tracking_uri = tracking_uri or str(Path.cwd() / "mlruns")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
    
    def get_latest_run_metrics(
        self,
        experiment_name: str,
        metric_keys: Optional[List[str]] = None,
    ) -> ToolResult:
        """
        Fetch metrics from the latest run of an experiment.
        
        Args:
            experiment_name: Name of the MLflow experiment.
            metric_keys: Optional list of specific metrics to retrieve.
            
        Returns:
            ToolResult with metrics data.
            
        Example:
            >>> tool.get_latest_run_metrics("forecasting-pipeline")
            ToolResult(success=True, data={"run_id": "...", "metrics": {"mape": 12.5}})
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            
            if not experiment:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Experiment '{experiment_name}' not found",
                )
            
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            
            if not runs:
                return ToolResult(
                    success=True,
                    data={"message": "No runs found in experiment"},
                )
            
            run = runs[0]
            metrics = run.data.metrics
            
            if metric_keys:
                metrics = {k: v for k, v in metrics.items() if k in metric_keys}
            
            return ToolResult(
                success=True,
                data={
                    "run_id": run.info.run_id,
                    "experiment_name": experiment_name,
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(
                        run.info.start_time / 1000
                    ).isoformat(),
                    "metrics": metrics,
                    "params": dict(run.data.params),
                },
            )
            
        except Exception as e:
            logger.error(f"MLflow error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
    ) -> ToolResult:
        """
        Register a model from an MLflow run to the model registry.
        
        Args:
            run_id: The run ID containing the model artifact.
            model_name: Name for the registered model.
            artifact_path: Path to model artifact in the run.
            
        Returns:
            ToolResult with model version info.
            
        Example:
            >>> tool.register_model("abc123", "churn-model")
            ToolResult(success=True, data={"name": "churn-model", "version": "1"})
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            result = mlflow.register_model(model_uri, model_name)
            
            return ToolResult(
                success=True,
                data={
                    "name": model_name,
                    "version": result.version,
                    "run_id": run_id,
                    "source": model_uri,
                },
                requires_confirmation=True,
                action_description=f"Registered model '{model_name}' version {result.version} from run {run_id[:8]}",
            )
            
        except Exception as e:
            logger.error(f"MLflow error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def get_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None,
    ) -> ToolResult:
        """
        List versions of a registered model.
        
        Args:
            model_name: Name of the registered model.
            stages: Optional list of stages to filter (e.g., ["Production", "Staging"]).
            
        Returns:
            ToolResult with model versions.
            
        Example:
            >>> tool.get_model_versions("churn-model")
            ToolResult(success=True, data={"versions": [{"version": "1", ...}]})
        """
        try:
            if stages:
                versions = self.client.get_latest_versions(model_name, stages=stages)
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")
            
            version_data = [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "created": datetime.fromtimestamp(
                        v.creation_timestamp / 1000
                    ).isoformat(),
                }
                for v in versions
            ]
            
            return ToolResult(
                success=True,
                data={
                    "model_name": model_name,
                    "versions": version_data,
                    "count": len(version_data),
                },
            )
            
        except Exception as e:
            logger.error(f"MLflow error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> ToolResult:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model.
            version: Version number to transition.
            stage: Target stage (Staging, Production, Archived).
            archive_existing: Whether to archive existing versions in target stage.
            
        Returns:
            ToolResult indicating success.
        """
        if stage not in ["Staging", "Production", "Archived", "None"]:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid stage: {stage}. Must be Staging, Production, Archived, or None.",
            )
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            
            return ToolResult(
                success=True,
                data={
                    "model_name": model_name,
                    "version": version,
                    "new_stage": stage,
                },
                requires_confirmation=True,
                action_description=f"Transitioned '{model_name}' v{version} to {stage}",
            )
            
        except Exception as e:
            logger.error(f"MLflow error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def list_experiments(self) -> ToolResult:
        """
        List all MLflow experiments.
        
        Returns:
            ToolResult with list of experiments.
        """
        try:
            experiments = self.client.search_experiments()
            
            exp_data = [
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "artifact_location": e.artifact_location,
                    "lifecycle_stage": e.lifecycle_stage,
                }
                for e in experiments
            ]
            
            return ToolResult(
                success=True,
                data={"experiments": exp_data, "count": len(exp_data)},
            )
            
        except Exception as e:
            logger.error(f"MLflow error: {e}")
            return ToolResult(success=False, data=None, error=str(e))


# Convenience functions
_default_tool: Optional[MLflowTool] = None


def get_tool(tracking_uri: Optional[str] = None) -> MLflowTool:
    """Get or create default MLflow tool instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = MLflowTool(tracking_uri=tracking_uri)
    return _default_tool


def get_latest_run_metrics(
    experiment_name: str,
    metric_keys: Optional[List[str]] = None,
) -> ToolResult:
    """Get latest run metrics. See MLflowTool.get_latest_run_metrics for details."""
    return get_tool().get_latest_run_metrics(experiment_name, metric_keys)


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
) -> ToolResult:
    """Register a model. See MLflowTool.register_model for details."""
    return get_tool().register_model(run_id, model_name, artifact_path)
