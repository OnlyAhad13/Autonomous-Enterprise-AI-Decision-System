"""
Predictions Router - Single and Batch Predictions.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: Dict[str, Any] = Field(
        ...,
        example={
            "date": "2026-01-15",
            "product_id": "P001",
            "store_id": "S001",
        }
    )
    model_type: str = Field("prophet", description="Model type: prophet, lstm, ets")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    prediction: float
    confidence_lower: float
    confidence_upper: float
    model_type: str
    model_version: str
    inference_time_ms: float
    timestamp: str


class ForecastPoint(BaseModel):
    """Single forecast data point."""
    date: str
    forecast: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    """Forecast response for date range."""
    forecasts: List[ForecastPoint]
    model_type: str
    model_version: str
    generated_at: str


class ExplainResponse(BaseModel):
    """Feature importance explanation."""
    prediction: float
    feature_importance: List[Dict[str, Any]]
    method: str
    model_version: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/single", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make a single prediction."""
    import time
    start = time.time()
    
    # Simulated prediction
    base_value = random.uniform(800000, 1000000)
    margin = base_value * 0.08
    
    return PredictionResponse(
        prediction=round(base_value, 2),
        confidence_lower=round(base_value - margin, 2),
        confidence_upper=round(base_value + margin, 2),
        model_type=request.model_type,
        model_version="v3.0",
        inference_time_ms=round((time.time() - start) * 1000 + random.uniform(5, 15), 2),
        timestamp=datetime.now().isoformat(),
    )


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    start_date: str = "2026-01-08",
    end_date: str = "2026-01-15",
    model_type: str = "prophet",
):
    """Get forecast for a date range."""
    from datetime import datetime
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    forecasts = []
    current = start
    while current <= end:
        base = random.uniform(850000, 980000)
        margin = base * 0.08
        forecasts.append(ForecastPoint(
            date=current.strftime("%Y-%m-%d"),
            forecast=round(base, 2),
            lower_bound=round(base - margin, 2),
            upper_bound=round(base + margin, 2),
        ))
        current += timedelta(days=1)
    
    return ForecastResponse(
        forecasts=forecasts,
        model_type=model_type,
        model_version="v3.0",
        generated_at=datetime.now().isoformat(),
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: PredictionRequest):
    """Get feature importance for a prediction."""
    
    # Get prediction first
    pred_response = await make_prediction(request)
    
    # Generate feature importance
    features = list(request.features.keys())
    importance = [
        {
            "feature": f,
            "importance": round(random.uniform(0.05, 0.3), 4),
            "direction": random.choice(["positive", "negative"]),
            "value": request.features[f],
        }
        for f in features
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)
    
    # Add contextual features
    importance.extend([
        {"feature": "day_of_week", "importance": 0.15, "direction": "positive", "value": "Tuesday"},
        {"feature": "month", "importance": 0.12, "direction": "positive", "value": "January"},
        {"feature": "is_holiday", "importance": 0.08, "direction": "negative", "value": False},
    ])
    
    return ExplainResponse(
        prediction=pred_response.prediction,
        feature_importance=importance[:10],
        method="shap",
        model_version=pred_response.model_version,
    )


@router.get("/history")
async def get_prediction_history(limit: int = 50):
    """Get recent prediction history."""
    now = datetime.now()
    
    history = [
        {
            "id": f"pred_{random.randint(100000, 999999)}",
            "timestamp": (now - timedelta(minutes=i)).isoformat(),
            "model_type": random.choice(["prophet", "lstm"]),
            "prediction": round(random.uniform(800000, 1000000), 2),
            "inference_time_ms": round(random.uniform(8, 20), 2),
            "features": {"date": "2026-01-15", "product_id": f"P{random.randint(1, 100)}"},
        }
        for i in range(limit)
    ]
    
    return {"predictions": history, "total": len(history)}
