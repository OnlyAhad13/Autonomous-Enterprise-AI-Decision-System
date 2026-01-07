"""
Monitoring Router - System Metrics and Alerts.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================

class SystemMetric(BaseModel):
    """Single system metric."""
    name: str
    value: float
    unit: str
    status: str  # healthy, warning, critical


class AlertItem(BaseModel):
    """Single alert."""
    id: str
    severity: str  # info, warning, critical
    title: str
    message: str
    timestamp: str
    resolved: bool
    source: str


class ServiceStatus(BaseModel):
    """Service health status."""
    name: str
    status: str  # healthy, degraded, down
    uptime: str
    last_check: str
    details: Optional[str] = None


class MetricsOverview(BaseModel):
    """Complete metrics overview."""
    system_metrics: List[SystemMetric]
    services: List[ServiceStatus]
    alerts: List[AlertItem]
    metrics_history: Dict[str, List[Dict[str, Any]]]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/overview", response_model=MetricsOverview)
async def get_metrics_overview():
    """Get complete monitoring overview."""
    
    now = datetime.now()
    
    # System metrics
    system_metrics = [
        SystemMetric(name="CPU Usage", value=round(random.uniform(20, 60), 1), unit="%", status="healthy"),
        SystemMetric(name="Memory", value=round(random.uniform(40, 70), 1), unit="%", status="healthy"),
        SystemMetric(name="Disk", value=round(random.uniform(30, 50), 1), unit="%", status="healthy"),
        SystemMetric(name="Network In", value=round(random.uniform(100, 500), 1), unit="MB/s", status="healthy"),
        SystemMetric(name="Network Out", value=round(random.uniform(50, 200), 1), unit="MB/s", status="healthy"),
        SystemMetric(name="GPU", value=round(random.uniform(0, 30), 1), unit="%", status="healthy"),
    ]
    
    # Service statuses
    services = [
        ServiceStatus(name="Kafka Broker", status="healthy", uptime="7d 12h 34m", last_check=now.isoformat()),
        ServiceStatus(name="Spark Streaming", status="healthy", uptime="7d 12h 30m", last_check=now.isoformat()),
        ServiceStatus(name="Delta Lake", status="healthy", uptime="7d 12h 34m", last_check=now.isoformat()),
        ServiceStatus(name="Feast (Online)", status="healthy", uptime="7d 10h 15m", last_check=now.isoformat()),
        ServiceStatus(name="MLflow", status="healthy", uptime="1h 30m", last_check=now.isoformat()),
        ServiceStatus(name="Prediction API", status="healthy", uptime="45m", last_check=now.isoformat()),
        ServiceStatus(name="Agent Core", status="healthy", uptime="7d 12h 34m", last_check=now.isoformat()),
    ]
    
    # Alerts
    alerts = [
        AlertItem(
            id="alert_001",
            severity="info",
            title="Model Retrained",
            message="Prophet model successfully retrained with MAPE 4.2%",
            timestamp=(now - timedelta(hours=2)).isoformat(),
            resolved=True,
            source="MLflow",
        ),
        AlertItem(
            id="alert_002",
            severity="warning",
            title="Kafka Consumer Lag",
            message="Consumer lag reached 500 messages on partition 2",
            timestamp=(now - timedelta(hours=6)).isoformat(),
            resolved=True,
            source="Kafka",
        ),
        AlertItem(
            id="alert_003",
            severity="info",
            title="Feature Store Updated",
            message="Materialized 15,000 features to online store",
            timestamp=(now - timedelta(hours=1)).isoformat(),
            resolved=True,
            source="Feast",
        ),
    ]
    
    # Metrics history (last 24 hours)
    hours = [(now - timedelta(hours=i)).strftime("%H:00") for i in range(24, 0, -1)]
    metrics_history = {
        "cpu": [{"time": h, "value": random.uniform(20, 60)} for h in hours],
        "memory": [{"time": h, "value": random.uniform(40, 70)} for h in hours],
        "requests": [{"time": h, "value": random.randint(1000, 5000)} for h in hours],
    }
    
    return MetricsOverview(
        system_metrics=system_metrics,
        services=services,
        alerts=alerts,
        metrics_history=metrics_history,
    )


@router.get("/services")
async def get_services():
    """Get service health statuses."""
    overview = await get_metrics_overview()
    return {"services": overview.services}


@router.get("/alerts")
async def get_alerts(resolved: Optional[bool] = None):
    """Get alerts with optional filter."""
    overview = await get_metrics_overview()
    alerts = overview.alerts
    
    if resolved is not None:
        alerts = [a for a in alerts if a.resolved == resolved]
    
    return {"alerts": alerts, "total": len(alerts)}
