"""
Prometheus Tool Wrapper.

Provides functions to query Prometheus metrics and alerts.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
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


class PrometheusTool:
    """
    Tool wrapper for Prometheus operations.
    
    Provides safe functions for LLM agents to query metrics and alerts.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:9090",
        timeout: int = 30,
    ):
        """
        Initialize Prometheus tool.
        
        Args:
            base_url: Prometheus server URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
    ) -> requests.Response:
        """Make request to Prometheus API."""
        url = f"{self.base_url}/api/v1/{endpoint.lstrip('/')}"
        return requests.get(url, params=params, timeout=self.timeout)
    
    def query(
        self,
        promql: str,
        time: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute an instant PromQL query.
        
        Args:
            promql: The PromQL query string.
            time: Optional evaluation timestamp (RFC3339 or Unix timestamp).
            
        Returns:
            ToolResult with query results.
            
        Example:
            >>> tool.query("up{job='prometheus'}")
            ToolResult(success=True, data={"result": [{"metric": {...}, "value": [1234, "1"]}]})
        """
        try:
            params = {"query": promql}
            if time:
                params["time"] = time
            
            response = self._request("query", params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    result = data.get("data", {})
                    
                    # Format results for readability
                    formatted = []
                    for item in result.get("result", []):
                        metric = item.get("metric", {})
                        value = item.get("value", [])
                        
                        formatted.append({
                            "labels": metric,
                            "timestamp": value[0] if len(value) > 0 else None,
                            "value": value[1] if len(value) > 1 else None,
                        })
                    
                    return ToolResult(
                        success=True,
                        data={
                            "query": promql,
                            "result_type": result.get("resultType"),
                            "results": formatted,
                            "count": len(formatted),
                        },
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=data.get("error", "Query failed"),
                    )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Prometheus error: {response.status_code}",
                )
                
        except requests.RequestException as e:
            logger.error(f"Prometheus error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def query_range(
        self,
        promql: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        step: str = "1m",
    ) -> ToolResult:
        """
        Execute a range PromQL query.
        
        Args:
            promql: The PromQL query string.
            start: Start time (RFC3339, Unix timestamp, or datetime).
            end: End time (RFC3339, Unix timestamp, or datetime).
            step: Query resolution step (e.g., "1m", "5m", "1h").
            
        Returns:
            ToolResult with time series data.
            
        Example:
            >>> tool.query_range("rate(http_requests_total[5m])", "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z")
            ToolResult(success=True, data={"results": [{"labels": {...}, "values": [[ts, val], ...]}]})
        """
        try:
            # Convert datetime to timestamp
            if isinstance(start, datetime):
                start = start.isoformat()
            if isinstance(end, datetime):
                end = end.isoformat()
            
            params = {
                "query": promql,
                "start": start,
                "end": end,
                "step": step,
            }
            
            response = self._request("query_range", params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    result = data.get("data", {})
                    
                    formatted = []
                    for item in result.get("result", []):
                        metric = item.get("metric", {})
                        values = item.get("values", [])
                        
                        formatted.append({
                            "labels": metric,
                            "values": [
                                {"timestamp": v[0], "value": v[1]}
                                for v in values
                            ],
                            "point_count": len(values),
                        })
                    
                    return ToolResult(
                        success=True,
                        data={
                            "query": promql,
                            "start": start,
                            "end": end,
                            "step": step,
                            "results": formatted,
                            "series_count": len(formatted),
                        },
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=data.get("error", "Query failed"),
                    )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Prometheus error: {response.status_code}",
                )
                
        except requests.RequestException as e:
            logger.error(f"Prometheus error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def get_alerts(
        self,
        state: Optional[str] = None,
    ) -> ToolResult:
        """
        Get active Prometheus alerts.
        
        Args:
            state: Optional filter by state (firing, pending, inactive).
            
        Returns:
            ToolResult with list of alerts.
            
        Example:
            >>> tool.get_alerts(state="firing")
            ToolResult(success=True, data={"alerts": [{"name": "HighCPU", ...}]})
        """
        try:
            response = self._request("alerts")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    alerts = data.get("data", {}).get("alerts", [])
                    
                    # Filter by state if specified
                    if state:
                        alerts = [a for a in alerts if a.get("state") == state]
                    
                    formatted = [
                        {
                            "name": a.get("labels", {}).get("alertname"),
                            "state": a.get("state"),
                            "severity": a.get("labels", {}).get("severity"),
                            "summary": a.get("annotations", {}).get("summary"),
                            "active_at": a.get("activeAt"),
                            "labels": a.get("labels"),
                        }
                        for a in alerts
                    ]
                    
                    return ToolResult(
                        success=True,
                        data={
                            "alerts": formatted,
                            "count": len(formatted),
                            "firing": len([a for a in alerts if a.get("state") == "firing"]),
                            "pending": len([a for a in alerts if a.get("state") == "pending"]),
                        },
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=data.get("error", "Failed to get alerts"),
                    )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Prometheus error: {response.status_code}",
                )
                
        except requests.RequestException as e:
            logger.error(f"Prometheus error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def get_targets(
        self,
        state: Optional[str] = None,
    ) -> ToolResult:
        """
        Get Prometheus scrape targets.
        
        Args:
            state: Optional filter by state (active, dropped).
            
        Returns:
            ToolResult with list of targets.
        """
        try:
            params = {}
            if state:
                params["state"] = state
            
            response = self._request("targets", params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    targets = data.get("data", {})
                    
                    active = [
                        {
                            "job": t.get("labels", {}).get("job"),
                            "instance": t.get("labels", {}).get("instance"),
                            "health": t.get("health"),
                            "last_scrape": t.get("lastScrape"),
                            "scrape_duration": t.get("lastScrapeDuration"),
                        }
                        for t in targets.get("activeTargets", [])
                    ]
                    
                    return ToolResult(
                        success=True,
                        data={
                            "active_targets": active,
                            "active_count": len(active),
                            "dropped_count": len(targets.get("droppedTargets", [])),
                        },
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=data.get("error", "Failed to get targets"),
                    )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Prometheus error: {response.status_code}",
                )
                
        except requests.RequestException as e:
            logger.error(f"Prometheus error: {e}")
            return ToolResult(success=False, data=None, error=str(e))


# Convenience functions
_default_tool: Optional[PrometheusTool] = None


def get_tool(base_url: str = "http://localhost:9090") -> PrometheusTool:
    """Get or create default Prometheus tool instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = PrometheusTool(base_url=base_url)
    return _default_tool


def query(promql: str, time: Optional[str] = None) -> ToolResult:
    """Execute PromQL query. See PrometheusTool.query for details."""
    return get_tool().query(promql, time)


def query_range(
    promql: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    step: str = "1m",
) -> ToolResult:
    """Execute range query. See PrometheusTool.query_range for details."""
    return get_tool().query_range(promql, start, end, step)


def get_alerts(state: Optional[str] = None) -> ToolResult:
    """Get alerts. See PrometheusTool.get_alerts for details."""
    return get_tool().get_alerts(state)
