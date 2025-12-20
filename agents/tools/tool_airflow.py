"""
Airflow Tool Wrapper.

Provides functions to interact with Apache Airflow via REST API.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    requires_confirmation: bool = False
    action_description: Optional[str] = None


class AirflowTool:
    """
    Tool wrapper for Apache Airflow operations.
    
    Provides safe functions for LLM agents to interact with Airflow.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        username: str = "airflow",
        password: str = "airflow",
        timeout: int = 30,
    ):
        """
        Initialize Airflow tool.
        
        Args:
            base_url: Airflow webserver URL.
            username: Basic auth username.
            password: Basic auth password.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.timeout = timeout
        self.api_base = f"{self.base_url}/api/v1"
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> requests.Response:
        """Make authenticated request to Airflow API."""
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        kwargs.setdefault("auth", self.auth)
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", {"Content-Type": "application/json"})
        
        return requests.request(method, url, **kwargs)
    
    def trigger_dag(
        self,
        dag_id: str,
        conf: Optional[Dict[str, Any]] = None,
        logical_date: Optional[str] = None,
    ) -> ToolResult:
        """
        Trigger an Airflow DAG run.
        
        Args:
            dag_id: The DAG identifier to trigger.
            conf: Optional configuration dictionary to pass to the DAG.
            logical_date: Optional execution date (ISO format).
            
        Returns:
            ToolResult with run_id on success.
            
        Example:
            >>> tool.trigger_dag("etl_pipeline", conf={"date": "2024-01-01"})
            ToolResult(success=True, data={"run_id": "manual__2024-01-01..."})
        """
        payload: Dict[str, Any] = {}
        
        if conf:
            payload["conf"] = conf
        if logical_date:
            payload["logical_date"] = logical_date
        
        try:
            response = self._request(
                "POST",
                f"dags/{dag_id}/dagRuns",
                json=payload,
            )
            
            if response.status_code == 200:
                data = response.json()
                return ToolResult(
                    success=True,
                    data={
                        "run_id": data.get("dag_run_id"),
                        "dag_id": dag_id,
                        "state": data.get("state"),
                        "execution_date": data.get("logical_date"),
                    },
                    requires_confirmation=True,
                    action_description=f"Triggered DAG '{dag_id}' with config: {conf}",
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Failed to trigger DAG: {response.status_code} - {response.text}",
                )
                
        except requests.RequestException as e:
            logger.error(f"Airflow API error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def get_dag_status(
        self,
        dag_id: str,
        run_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Get the status of a DAG or specific DAG run.
        
        Args:
            dag_id: The DAG identifier.
            run_id: Optional specific run ID. If None, gets latest run.
            
        Returns:
            ToolResult with DAG run status.
            
        Example:
            >>> tool.get_dag_status("etl_pipeline")
            ToolResult(success=True, data={"state": "success", ...})
        """
        try:
            if run_id:
                endpoint = f"dags/{dag_id}/dagRuns/{run_id}"
            else:
                # Get latest run
                endpoint = f"dags/{dag_id}/dagRuns"
            
            response = self._request("GET", endpoint)
            
            if response.status_code == 200:
                data = response.json()
                
                # If fetching list, get the latest
                if "dag_runs" in data:
                    runs = data["dag_runs"]
                    if not runs:
                        return ToolResult(
                            success=True,
                            data={"message": "No runs found for this DAG"},
                        )
                    latest = max(runs, key=lambda x: x.get("execution_date", ""))
                    data = latest
                
                return ToolResult(
                    success=True,
                    data={
                        "dag_id": dag_id,
                        "run_id": data.get("dag_run_id"),
                        "state": data.get("state"),
                        "execution_date": data.get("logical_date"),
                        "start_date": data.get("start_date"),
                        "end_date": data.get("end_date"),
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Failed to get DAG status: {response.status_code}",
                )
                
        except requests.RequestException as e:
            logger.error(f"Airflow API error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def list_dags(
        self,
        only_active: bool = True,
    ) -> ToolResult:
        """
        List available DAGs.
        
        Args:
            only_active: If True, only return active DAGs.
            
        Returns:
            ToolResult with list of DAG info.
            
        Example:
            >>> tool.list_dags()
            ToolResult(success=True, data={"dags": [{"dag_id": "etl_pipeline", ...}]})
        """
        try:
            params = {"only_active": str(only_active).lower()}
            response = self._request("GET", "dags", params=params)
            
            if response.status_code == 200:
                data = response.json()
                dags = [
                    {
                        "dag_id": dag.get("dag_id"),
                        "is_paused": dag.get("is_paused"),
                        "is_active": dag.get("is_active"),
                        "description": dag.get("description"),
                        "schedule_interval": dag.get("schedule_interval"),
                    }
                    for dag in data.get("dags", [])
                ]
                return ToolResult(success=True, data={"dags": dags, "total": len(dags)})
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Failed to list DAGs: {response.status_code}",
                )
                
        except requests.RequestException as e:
            logger.error(f"Airflow API error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def pause_dag(self, dag_id: str) -> ToolResult:
        """
        Pause a DAG.
        
        Args:
            dag_id: The DAG to pause.
            
        Returns:
            ToolResult indicating success.
        """
        try:
            response = self._request(
                "PATCH",
                f"dags/{dag_id}",
                json={"is_paused": True},
            )
            
            if response.status_code == 200:
                return ToolResult(
                    success=True,
                    data={"dag_id": dag_id, "is_paused": True},
                    requires_confirmation=True,
                    action_description=f"Paused DAG '{dag_id}'",
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Failed to pause DAG: {response.status_code}",
                )
                
        except requests.RequestException as e:
            return ToolResult(success=False, data=None, error=str(e))


# Convenience functions for direct usage
_default_tool: Optional[AirflowTool] = None


def get_tool(
    base_url: str = "http://localhost:8080",
    **kwargs,
) -> AirflowTool:
    """Get or create default Airflow tool instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = AirflowTool(base_url=base_url, **kwargs)
    return _default_tool


def trigger_dag(
    dag_id: str,
    conf: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    """Trigger a DAG run. See AirflowTool.trigger_dag for details."""
    return get_tool().trigger_dag(dag_id, conf)


def get_dag_status(dag_id: str, run_id: Optional[str] = None) -> ToolResult:
    """Get DAG status. See AirflowTool.get_dag_status for details."""
    return get_tool().get_dag_status(dag_id, run_id)
