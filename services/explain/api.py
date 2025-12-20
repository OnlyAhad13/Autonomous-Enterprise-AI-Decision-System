"""
FastAPI Explainability Service

Provides endpoints for model explainability via SHAP values.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ============================================================================
# Pydantic Models
# ============================================================================


class FeatureSHAP(BaseModel):
    """SHAP value for a single feature."""
    
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value for this prediction")
    shap_value: float = Field(..., description="SHAP contribution to prediction")


class PredictionExplanation(BaseModel):
    """Complete explanation for a single prediction."""
    
    prediction_id: str = Field(..., description="Unique prediction identifier")
    predicted_class: int = Field(..., description="Predicted class (0 or 1)")
    predicted_probability: float = Field(..., description="Probability of positive class")
    base_value: float = Field(..., description="Expected value (base prediction)")
    feature_contributions: list[FeatureSHAP] = Field(
        ..., description="SHAP values per feature, sorted by absolute contribution"
    )
    top_positive_drivers: list[str] = Field(
        ..., description="Features pushing prediction higher"
    )
    top_negative_drivers: list[str] = Field(
        ..., description="Features pushing prediction lower"
    )
    generated_at: str = Field(..., description="ISO timestamp of explanation generation")


class GlobalSHAPSummary(BaseModel):
    """Global SHAP feature importance summary."""
    
    model_type: str = Field(..., description="Type of model (XGBoost, LightGBM, etc.)")
    expected_value: float = Field(..., description="Base prediction value")
    n_samples: int = Field(..., description="Number of samples used for SHAP")
    feature_importance: list[dict] = Field(
        ..., description="Features ranked by mean absolute SHAP"
    )
    generated_at: str = Field(..., description="ISO timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    service: str = "explainability-api"
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: str
    prediction_id: Optional[str] = None


# ============================================================================
# Mock Data Store (Replace with actual storage in production)
# ============================================================================

# Feature names used in the model
FEATURE_NAMES = [
    'transaction_count', 'total_revenue', 'avg_revenue', 'std_revenue',
    'avg_price', 'max_price', 'min_price', 'total_quantity', 'avg_quantity',
    'days_since_first', 'days_since_last', 'avg_days_between'
]

# Mock predictions store (in production, load from database/file)
MOCK_PREDICTIONS = {
    "pred_0": {
        "predicted_class": 1,
        "predicted_probability": 0.78,
        "feature_values": [2, 150.0, 75.0, 25.0, 50.0, 100.0, 25.0, 3, 1.5, 30, 5, 15.0],
        "shap_values": [0.15, -0.42, -0.18, 0.05, -0.08, -0.12, 0.03, 0.02, -0.01, 0.22, 0.35, 0.08]
    },
    "pred_1": {
        "predicted_class": 0,
        "predicted_probability": 0.23,
        "feature_values": [15, 2500.0, 166.7, 80.0, 120.0, 300.0, 50.0, 20, 1.3, 180, 3, 12.0],
        "shap_values": [-0.35, 0.52, 0.28, 0.12, 0.15, 0.08, -0.02, -0.05, 0.03, -0.18, -0.45, -0.12]
    },
    "pred_2": {
        "predicted_class": 1,
        "predicted_probability": 0.92,
        "feature_values": [1, 45.0, 45.0, 0.0, 45.0, 45.0, 45.0, 1, 1.0, 60, 55, 60.0],
        "shap_values": [0.28, -0.55, -0.32, 0.0, -0.05, -0.02, 0.01, 0.08, 0.04, 0.15, 0.65, 0.38]
    },
    "test_123": {  # For demo purposes
        "predicted_class": 1,
        "predicted_probability": 0.67,
        "feature_values": [3, 280.0, 93.3, 45.0, 75.0, 150.0, 40.0, 4, 1.3, 45, 12, 15.0],
        "shap_values": [0.08, -0.25, -0.12, 0.06, -0.04, -0.08, 0.02, 0.01, -0.02, 0.18, 0.28, 0.05]
    }
}

# Global SHAP summary (precomputed)
GLOBAL_SHAP_SUMMARY = {
    "model_type": "XGBoost",
    "expected_value": 0.35,
    "n_samples": 2000,
    "feature_importance": [
        {"feature": "days_since_last", "mean_abs_shap": 0.42},
        {"feature": "total_revenue", "mean_abs_shap": 0.38},
        {"feature": "avg_revenue", "mean_abs_shap": 0.24},
        {"feature": "transaction_count", "mean_abs_shap": 0.21},
        {"feature": "days_since_first", "mean_abs_shap": 0.18},
        {"feature": "avg_days_between", "mean_abs_shap": 0.15},
        {"feature": "avg_price", "mean_abs_shap": 0.12},
        {"feature": "max_price", "mean_abs_shap": 0.09},
        {"feature": "std_revenue", "mean_abs_shap": 0.07},
        {"feature": "total_quantity", "mean_abs_shap": 0.05},
        {"feature": "min_price", "mean_abs_shap": 0.04},
        {"feature": "avg_quantity", "mean_abs_shap": 0.03},
    ]
}


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Model Explainability API",
    description="API for SHAP-based model explanations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/explain", response_model=PredictionExplanation, tags=["Explainability"])
async def get_explanation(
    prediction_id: str = Query(..., description="Unique prediction ID to explain")
):
    """
    Get SHAP explanation for a specific prediction.
    
    Returns feature contributions, top drivers, and prediction details.
    """
    if prediction_id not in MOCK_PREDICTIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction '{prediction_id}' not found. "
                   f"Available: {list(MOCK_PREDICTIONS.keys())}"
        )
    
    pred = MOCK_PREDICTIONS[prediction_id]
    
    # Build feature contributions
    contributions = []
    for fname, fval, shap_val in zip(
        FEATURE_NAMES, pred["feature_values"], pred["shap_values"]
    ):
        contributions.append(FeatureSHAP(
            feature=fname,
            value=fval,
            shap_value=shap_val
        ))
    
    # Sort by absolute SHAP value
    contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
    
    # Identify top drivers
    positive_drivers = [c.feature for c in contributions if c.shap_value > 0][:3]
    negative_drivers = [c.feature for c in contributions if c.shap_value < 0][:3]
    
    return PredictionExplanation(
        prediction_id=prediction_id,
        predicted_class=pred["predicted_class"],
        predicted_probability=pred["predicted_probability"],
        base_value=GLOBAL_SHAP_SUMMARY["expected_value"],
        feature_contributions=contributions,
        top_positive_drivers=positive_drivers,
        top_negative_drivers=negative_drivers,
        generated_at=datetime.now().isoformat()
    )


@app.get("/explain/summary", response_model=GlobalSHAPSummary, tags=["Explainability"])
async def get_global_summary():
    """
    Get global SHAP feature importance summary.
    
    Returns aggregated feature importance across all predictions.
    """
    return GlobalSHAPSummary(
        model_type=GLOBAL_SHAP_SUMMARY["model_type"],
        expected_value=GLOBAL_SHAP_SUMMARY["expected_value"],
        n_samples=GLOBAL_SHAP_SUMMARY["n_samples"],
        feature_importance=GLOBAL_SHAP_SUMMARY["feature_importance"],
        generated_at=datetime.now().isoformat()
    )


@app.get("/explain/html", response_class=HTMLResponse, tags=["Explainability"])
async def get_explanation_html(
    prediction_id: str = Query(..., description="Unique prediction ID to explain")
):
    """
    Get HTML visualization of SHAP explanation.
    
    Returns embeddable HTML for documentation.
    """
    if prediction_id not in MOCK_PREDICTIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction '{prediction_id}' not found"
        )
    
    pred = MOCK_PREDICTIONS[prediction_id]
    base_value = GLOBAL_SHAP_SUMMARY["expected_value"]
    
    # Build contribution rows
    contributions = list(zip(FEATURE_NAMES, pred["feature_values"], pred["shap_values"]))
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    
    rows_html = ""
    for fname, fval, shap_val in contributions:
        color = "#e74c3c" if shap_val > 0 else "#27ae60"
        bar_width = min(abs(shap_val) * 200, 150)
        direction = "right" if shap_val > 0 else "left"
        
        rows_html += f"""
        <tr>
            <td style="font-weight: 500;">{fname}</td>
            <td style="text-align: right;">{fval:.2f}</td>
            <td style="text-align: right; color: {color};">{shap_val:+.3f}</td>
            <td>
                <div style="width: 160px; height: 16px; background: #f0f0f0; position: relative;">
                    <div style="position: absolute; {'right' if shap_val < 0 else 'left'}: 50%; 
                                width: {bar_width}px; height: 100%; background: {color};"></div>
                    <div style="position: absolute; left: 50%; width: 2px; height: 100%; background: #333;"></div>
                </div>
            </td>
        </tr>
        """
    
    # Read template and fill
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Explanation - {prediction_id}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                background: #fafafa;
            }}
            .card {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.1);
                padding: 24px;
                margin-bottom: 20px;
            }}
            h1 {{
                color: #2c3e50;
                margin-bottom: 8px;
            }}
            .subtitle {{
                color: #7f8c8d;
                margin-bottom: 24px;
            }}
            .prediction-box {{
                display: flex;
                gap: 20px;
                margin-bottom: 24px;
            }}
            .metric {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px 24px;
                border-radius: 8px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 28px;
                font-weight: bold;
            }}
            .metric-label {{
                font-size: 12px;
                opacity: 0.9;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}
            th {{
                background: #f8f9fa;
                font-weight: 600;
                color: #2c3e50;
            }}
            .legend {{
                display: flex;
                gap: 20px;
                margin-top: 16px;
                font-size: 13px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .legend-color {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }}
            footer {{
                text-align: center;
                color: #95a5a6;
                font-size: 12px;
                margin-top: 24px;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🔍 SHAP Explanation</h1>
            <p class="subtitle">Prediction ID: <code>{prediction_id}</code></p>
            
            <div class="prediction-box">
                <div class="metric">
                    <div class="metric-value">{'Churned' if pred['predicted_class'] == 1 else 'Retained'}</div>
                    <div class="metric-label">Prediction</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{pred['predicted_probability']:.1%}</div>
                    <div class="metric-label">Probability</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{base_value:.2f}</div>
                    <div class="metric-label">Base Value</div>
                </div>
            </div>
            
            <h3>Feature Contributions</h3>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Value</th>
                        <th>SHAP</th>
                        <th>Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #e74c3c;"></div>
                    <span>Increases churn risk</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #27ae60;"></div>
                    <span>Decreases churn risk</span>
                </div>
            </div>
        </div>
        
        <footer>
            Generated at {datetime.now().isoformat()} | Model Explainability API v1.0.0
        </footer>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@app.get("/explain/list", tags=["Explainability"])
async def list_available_predictions():
    """List all available prediction IDs for explanation."""
    return {
        "available_predictions": list(MOCK_PREDICTIONS.keys()),
        "count": len(MOCK_PREDICTIONS)
    }


# ============================================================================
# Run with: uvicorn services.explain.api:app --reload --port 8001
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
