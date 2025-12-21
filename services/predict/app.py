"""
Prediction Service - FastAPI Application.

Provides endpoints for ML model predictions, batch processing, and explanations.

Endpoints:
    POST /predict - Single prediction
    POST /batch_predict - Batch predictions from CSV
    POST /explain - Get feature explanations
"""

import os
import io
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


class Settings:
    """Application settings."""
    
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "models/output/model.pkl")
    MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "forecasting-model")
    MODEL_STAGE: str = os.environ.get("MODEL_STAGE", "Production")
    EXPLAIN_SERVICE_URL: str = os.environ.get("EXPLAIN_SERVICE_URL", "http://localhost:8003")
    MAX_BATCH_SIZE: int = int(os.environ.get("MAX_BATCH_SIZE", "10000"))
    REQUEST_TIMEOUT: int = int(os.environ.get("REQUEST_TIMEOUT", "30"))


settings = Settings()


# ============================================================================
# Request/Response Models
# ============================================================================


class PredictionInput(BaseModel):
    """Input schema for single prediction."""
    
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary for prediction",
        example={
            "feature_1": 1.5,
            "feature_2": "category_a",
            "feature_3": 100,
        }
    )
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use",
    )
    return_probabilities: bool = Field(
        False,
        description="Return class probabilities for classification",
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "age": 35,
                    "income": 75000,
                    "tenure_months": 24,
                    "subscription_tier": "premium",
                },
                "return_probabilities": False,
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for single prediction."""
    
    prediction: Union[float, int, str, List[float]] = Field(
        ...,
        description="Model prediction",
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Class probabilities (if applicable)",
    )
    model_version: str = Field(
        ...,
        description="Model version used",
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
    )
    timestamp: str = Field(
        ...,
        description="Prediction timestamp",
    )


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    
    predictions: List[Union[float, int, str]] = Field(
        ...,
        description="List of predictions",
    )
    row_count: int = Field(
        ...,
        description="Number of rows processed",
    )
    model_version: str = Field(
        ...,
        description="Model version used",
    )
    inference_time_ms: float = Field(
        ...,
        description="Total inference time in milliseconds",
    )
    failed_rows: List[int] = Field(
        default_factory=list,
        description="Indices of rows that failed",
    )


class ExplainInput(BaseModel):
    """Input schema for explain endpoint."""
    
    features: Dict[str, Any] = Field(
        ...,
        description="Features to explain",
    )
    method: str = Field(
        "shap",
        description="Explanation method: 'shap' or 'lime'",
    )
    top_k: int = Field(
        10,
        description="Number of top features to return",
    )


class ExplainOutput(BaseModel):
    """Output schema for explain endpoint."""
    
    prediction: Union[float, int, str] = Field(
        ...,
        description="Model prediction for the input",
    )
    feature_importance: List[Dict[str, Any]] = Field(
        ...,
        description="Feature importance scores",
    )
    method: str = Field(
        ...,
        description="Explanation method used",
    )
    model_version: str = Field(
        ...,
        description="Model version used",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    model_loaded: bool = True
    model_version: str = ""
    timestamp: str = ""


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str
    detail: Optional[str] = None
    timestamp: str = ""


# ============================================================================
# Model Manager
# ============================================================================


class ModelManager:
    """Manages model loading and predictions."""
    
    def __init__(self):
        self.model = None
        self.model_version = "unknown"
        self.model_type = "unknown"
        self.feature_names: List[str] = []
        self._loaded = False
    
    def load(self) -> bool:
        """Load the model from configured source."""
        try:
            # Try MLflow first
            if self._load_from_mlflow():
                return True
            
            # Fallback to file path
            if self._load_from_file():
                return True
            
            # Create mock model for demo
            logger.warning("No model found, using mock model")
            self._create_mock_model()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_from_mlflow(self) -> bool:
        """Load model from MLflow registry."""
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            
            model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Get version info
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(
                settings.MODEL_NAME, 
                stages=[settings.MODEL_STAGE]
            )
            if versions:
                self.model_version = versions[0].version
            
            self._loaded = True
            logger.info(f"Loaded model from MLflow: {model_uri}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load from MLflow: {e}")
            return False
    
    def _load_from_file(self) -> bool:
        """Load model from file path."""
        model_path = Path(settings.MODEL_PATH)
        
        if not model_path.exists():
            return False
        
        try:
            import joblib
            self.model = joblib.load(model_path)
            self.model_version = "file"
            self._loaded = True
            logger.info(f"Loaded model from file: {model_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load from file: {e}")
            return False
    
    def _create_mock_model(self):
        """Create a mock model for testing."""
        class MockModel:
            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    return np.random.rand(len(X))
                return np.random.rand(1)
            
            def predict_proba(self, X):
                n = len(X) if isinstance(X, pd.DataFrame) else 1
                probs = np.random.rand(n, 2)
                return probs / probs.sum(axis=1, keepdims=True)
        
        self.model = MockModel()
        self.model_version = "mock-1.0"
        self.model_type = "mock"
        self._loaded = True
    
    def predict(
        self,
        features: Dict[str, Any],
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """Make a single prediction."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.model.predict(df)
        
        result = {
            "prediction": float(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
            "model_version": self.model_version,
            "inference_time_ms": (time.time() - start_time) * 1000,
        }
        
        # Add probabilities if requested
        if return_probabilities and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df)[0]
            result["probabilities"] = {
                f"class_{i}": float(p) for i, p in enumerate(probs)
            }
        
        return result
    
    def predict_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make batch predictions."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        predictions = self.model.predict(df)
        
        return {
            "predictions": predictions.tolist(),
            "row_count": len(df),
            "model_version": self.model_version,
            "inference_time_ms": (time.time() - start_time) * 1000,
            "failed_rows": [],
        }
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Global model manager
model_manager = ModelManager()


# ============================================================================
# Application Lifecycle
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting prediction service...")
    model_manager.load()
    logger.info(f"Model loaded: {model_manager.model_version}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down prediction service...")


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="ML Prediction Service",
    description="API for ML model predictions, batch processing, and explanations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"service": "ML Prediction Service", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.model_version,
        timestamp=datetime.now().isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Predictions"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict(input_data: PredictionInput):
    """
    Make a single prediction.
    
    Accepts feature dictionary and returns model prediction.
    """
    try:
        result = model_manager.predict(
            features=input_data.features,
            return_probabilities=input_data.return_probabilities,
        )
        
        return PredictionOutput(
            prediction=result["prediction"],
            probabilities=result.get("probabilities"),
            model_version=result["model_version"],
            inference_time_ms=result["inference_time_ms"],
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/batch_predict",
    response_model=BatchPredictionOutput,
    tags=["Predictions"],
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def batch_predict(
    file: UploadFile = File(..., description="CSV file with features"),
    has_header: bool = Query(True, description="CSV has header row"),
):
    """
    Make batch predictions from CSV file.
    
    Upload a CSV file and receive predictions for all rows.
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported",
        )
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            header=0 if has_header else None,
        )
        
        # Check size limit
        if len(df) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size {len(df)} exceeds limit {settings.MAX_BATCH_SIZE}",
            )
        
        # Make predictions
        result = model_manager.predict_batch(df)
        
        return BatchPredictionOutput(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post(
    "/batch_predict/download",
    tags=["Predictions"],
    response_class=StreamingResponse,
)
async def batch_predict_download(
    file: UploadFile = File(..., description="CSV file with features"),
    has_header: bool = Query(True, description="CSV has header row"),
):
    """
    Make batch predictions and return CSV with predictions column.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            header=0 if has_header else None,
        )
        
        result = model_manager.predict_batch(df)
        df["prediction"] = result["predictions"]
        
        # Return as CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            },
        )
        
    except Exception as e:
        logger.error(f"Batch download error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post(
    "/explain",
    response_model=ExplainOutput,
    tags=["Explainability"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def explain(input_data: ExplainInput):
    """
    Get feature explanations for a prediction.
    
    Uses SHAP or LIME to explain model predictions.
    """
    import requests
    
    # First get prediction
    try:
        pred_result = model_manager.predict(input_data.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # Call explain service if available
    try:
        response = requests.post(
            f"{settings.EXPLAIN_SERVICE_URL}/explain",
            json={
                "features": input_data.features,
                "method": input_data.method,
                "top_k": input_data.top_k,
            },
            timeout=settings.REQUEST_TIMEOUT,
        )
        
        if response.status_code == 200:
            explain_data = response.json()
            return ExplainOutput(
                prediction=pred_result["prediction"],
                feature_importance=explain_data.get("feature_importance", []),
                method=input_data.method,
                model_version=pred_result["model_version"],
            )
            
    except requests.RequestException:
        logger.warning("Explain service unavailable, using mock explanation")
    
    # Mock explanation if service unavailable
    feature_importance = [
        {
            "feature": name,
            "importance": float(np.random.uniform(0, 1)),
            "direction": "positive" if np.random.random() > 0.5 else "negative",
        }
        for name in list(input_data.features.keys())[:input_data.top_k]
    ]
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return ExplainOutput(
        prediction=pred_result["prediction"],
        feature_importance=feature_importance,
        method=input_data.method,
        model_version=pred_result["model_version"],
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return {
        "model_loaded": 1 if model_manager.is_loaded else 0,
        "model_version": model_manager.model_version,
    }


# ============================================================================
# Sample Requests (for documentation)
# ============================================================================


"""
# Sample Requests

## Single Prediction (curl)
```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {
      "age": 35,
      "income": 75000,
      "tenure_months": 24,
      "subscription_tier": "premium"
    },
    "return_probabilities": true
  }'
```

## Single Prediction (Python)
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {
            "age": 35,
            "income": 75000,
            "tenure_months": 24,
            "subscription_tier": "premium",
        },
        "return_probabilities": True,
    }
)
print(response.json())
```

## Batch Prediction (curl)
```bash
curl -X POST http://localhost:8000/batch_predict \\
  -F "file=@data/test_batch.csv"
```

## Batch Prediction (Python)
```python
import requests

with open("data/test_batch.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/batch_predict",
        files={"file": f},
    )
print(response.json())
```

## Explain (curl)
```bash
curl -X POST http://localhost:8000/explain \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {"age": 35, "income": 75000},
    "method": "shap",
    "top_k": 5
  }'
```
"""


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
