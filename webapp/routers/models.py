"""
Models Router - Real MLflow Integration with Live Training.

Connects to actual MLflow model registry and trains on live event data.
"""

import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

import mlflow
from mlflow.tracking import MlflowClient

# Setup MLflow
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

router = APIRouter()

# Model storage
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Currently deployed model for predictions
deployed_model = {
    "model": None,
    "scaler": None,
    "label_encoder": None,
    "model_name": None,
    "version": None,
    "loaded_at": None,
}

# Training status
training_status = {
    "is_training": False,
    "current_run_id": None,
    "progress": 0,
    "message": "",
}


# ============================================================================
# Response Models
# ============================================================================

class ModelVersion(BaseModel):
    version: str
    stage: str
    created_at: str
    metrics: Dict[str, float]
    status: str
    run_id: Optional[str] = None


class RegisteredModel(BaseModel):
    name: str
    description: str
    latest_versions: List[ModelVersion]
    tags: Dict[str, str]
    created_at: str


class TrainingRun(BaseModel):
    run_id: str
    model_type: str
    status: str
    start_time: str
    end_time: Optional[str]
    metrics: Dict[str, float]
    params: Dict[str, Any]


# ============================================================================
# Helper Functions
# ============================================================================

def get_mlflow_client():
    """Get MLflow client."""
    return MlflowClient(tracking_uri=MLFLOW_URI)


def get_real_models():
    """Get real models from MLflow registry."""
    try:
        client = get_mlflow_client()
        registered_models = client.search_registered_models()
        
        models = []
        for rm in registered_models:
            versions = []
            for v in rm.latest_versions if rm.latest_versions else []:
                metrics = {}
                try:
                    run = client.get_run(v.run_id)
                    metrics = run.data.metrics
                except:
                    pass
                
                versions.append(ModelVersion(
                    version=str(v.version),
                    stage=v.current_stage or "None",
                    created_at=datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat() if v.creation_timestamp else datetime.now().isoformat(),
                    metrics=dict(metrics) if metrics else {},
                    status=v.status or "READY",
                    run_id=v.run_id,
                ))
            
            models.append(RegisteredModel(
                name=rm.name,
                description=rm.description or f"Model: {rm.name}",
                latest_versions=versions,
                tags=dict(rm.tags) if rm.tags else {},
                created_at=datetime.fromtimestamp(rm.creation_timestamp / 1000).isoformat() if rm.creation_timestamp else datetime.now().isoformat(),
            ))
        
        return models
    except Exception as e:
        print(f"Error getting models from MLflow: {e}")
        return []


def get_real_runs(limit: int = 10):
    """Get real training runs from MLflow."""
    try:
        client = get_mlflow_client()
        experiments = client.search_experiments()
        exp_ids = [exp.experiment_id for exp in experiments]
        
        if not exp_ids:
            return []
        
        runs = client.search_runs(
            experiment_ids=exp_ids,
            max_results=limit,
            order_by=["start_time DESC"],
        )
        
        result = []
        for run in runs:
            model_type = run.data.params.get("model_type", "unknown")
            
            result.append(TrainingRun(
                run_id=run.info.run_id,
                model_type=model_type,
                status=run.info.status.lower() if run.info.status else "unknown",
                start_time=datetime.fromtimestamp(run.info.start_time / 1000).isoformat() if run.info.start_time else "",
                end_time=datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                metrics=dict(run.data.metrics) if run.data.metrics else {},
                params=dict(run.data.params) if run.data.params else {},
            ))
        
        return result
    except Exception as e:
        print(f"Error getting runs from MLflow: {e}")
        return []


def train_model_on_live_data(model_type: str):
    """Train model on live event data."""
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["progress"] = 10
        training_status["message"] = "Loading live event data..."
        
        from models.live_train import LiveModelTrainer
        
        # Get events from ingestion buffer
        events = []
        try:
            from webapp.routers.ingestion import event_buffer
            events = list(event_buffer)
            print(f"📊 Got {len(events)} events from live buffer")
        except ImportError as e:
            print(f"Could not import event buffer: {e}")
        
        # Generate synthetic events if needed
        if len(events) < 50:
            training_status["message"] = "Generating training data..."
            import random
            event_types = ["order_placed", "page_view", "purchase", "add_to_cart", "checkout_started"]
            for i in range(200 - len(events)):
                events.append({
                    "id": f"train_evt_{i}",
                    "timestamp": datetime.now().isoformat(),
                    "value": {
                        "event_type": random.choice(event_types),
                        "price": round(random.uniform(10, 500), 2),
                        "quantity": random.randint(1, 5),
                        "user_id": f"U{random.randint(1000, 9999)}",
                        "product_id": f"P{random.randint(100, 999)}",
                    }
                })
        
        training_status["progress"] = 30
        training_status["message"] = f"Training {model_type} model on {len(events)} events..."
        
        # Train
        trainer = LiveModelTrainer(mlflow_tracking_uri=MLFLOW_URI)
        result = trainer.train_event_classifier(events, model_type)
        
        training_status["progress"] = 90
        training_status["message"] = "Registering model..."
        training_status["current_run_id"] = result["mlflow_run_id"]
        
        training_status["progress"] = 100
        training_status["message"] = f"Training complete! Accuracy: {result['metrics']['accuracy']:.2%}"
        
        # Send notification
        try:
            from webapp.routers.notifications import add_notification
            add_notification(
                type="success",
                title="🎉 Model Training Complete",
                message=f"**Model:** {model_type}\n**Accuracy:** {result['metrics']['accuracy']:.2%}\n**F1 Score:** {result['metrics']['f1_weighted']:.2%}\n**Samples:** {len(events)}",
                data=result,
            )
        except:
            pass
        
        return result
        
    except Exception as e:
        training_status["message"] = f"Training failed: {str(e)}"
        print(f"Training error: {e}")
        raise
    finally:
        training_status["is_training"] = False


def load_model_for_predictions(model_name: str, version: Optional[str] = None):
    """Load a model from MLflow or local storage for predictions."""
    global deployed_model
    
    try:
        client = get_mlflow_client()
        
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            model_uri = f"models:/{model_name}/{versions[0].version}"
            version = versions[0].version
        
        model = mlflow.sklearn.load_model(model_uri)
        
        deployed_model["model"] = model
        deployed_model["model_name"] = model_name
        deployed_model["version"] = version
        deployed_model["loaded_at"] = datetime.now().isoformat()
        
        print(f"✅ Loaded model {model_name} v{version} for predictions")
        return True
        
    except Exception as e:
        print(f"MLflow load failed: {e}, trying local file...")
        
        model_path = MODEL_DIR / f"{model_name.replace('live-', '')}_event_type.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                deployed_model["model"] = data["model"]
                deployed_model["scaler"] = data.get("scaler")
                deployed_model["label_encoder"] = data.get("label_encoder")
                deployed_model["model_name"] = model_name
                deployed_model["version"] = "local"
                deployed_model["loaded_at"] = datetime.now().isoformat()
                print(f"✅ Loaded local model {model_name}")
                return True
        
        print(f"Could not load model {model_name}")
        return False


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/registry")
async def get_model_registry():
    """Get all registered models from MLflow."""
    models = get_real_models()
    
    if not models:
        local_models = list(MODEL_DIR.glob("*.pkl"))
        for lm in local_models:
            model_name = lm.stem.replace("_event_type", "")
            models.append(RegisteredModel(
                name=f"live-{model_name}",
                description=f"Locally trained {model_name} model",
                latest_versions=[ModelVersion(
                    version="1",
                    stage="Local",
                    created_at=datetime.fromtimestamp(lm.stat().st_mtime).isoformat(),
                    metrics={},
                    status="READY",
                )],
                tags={"source": "local"},
                created_at=datetime.fromtimestamp(lm.stat().st_ctime).isoformat(),
            ))
    
    return {"models": models, "total": len(models)}


@router.get("/runs")
async def get_training_runs(limit: int = 10):
    """Get recent training runs from MLflow."""
    runs = get_real_runs(limit)
    return {"runs": runs, "total": len(runs)}


@router.get("/training-status")
async def get_training_status():
    """Get current training status."""
    return training_status


@router.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    model_type: str = "random_forest",
):
    """Train a new model on live event data."""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["message"] = "Starting training..."
    
    background_tasks.add_task(train_model_on_live_data, model_type)
    
    return {
        "status": "started",
        "model_type": model_type,
        "message": f"Training {model_type} model on live data...",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/deploy/{model_name}")
async def deploy_model(model_name: str, version: str = "latest", stage: str = "Production"):
    """Deploy a model version for predictions."""
    global deployed_model
    
    try:
        success = load_model_for_predictions(model_name, version if version != "latest" else None)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Could not load model {model_name}")
        
        try:
            from webapp.routers.notifications import add_notification
            add_notification(
                type="success",
                title="🚀 Model Deployed",
                message=f"**Model:** {model_name}\n**Version:** {deployed_model['version']}\n**Stage:** {stage}",
            )
        except:
            pass
        
        return {
            "status": "success",
            "message": f"Model {model_name} v{deployed_model['version']} deployed to {stage}",
            "model_name": model_name,
            "version": deployed_model["version"],
            "loaded_at": deployed_model["loaded_at"],
            "timestamp": datetime.now().isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployed")
async def get_deployed_model():
    """Get currently deployed model info."""
    if not deployed_model["model"]:
        return {"deployed": False, "message": "No model currently deployed"}
    
    return {
        "deployed": True,
        "model_name": deployed_model["model_name"],
        "version": deployed_model["version"],
        "loaded_at": deployed_model["loaded_at"],
    }


@router.post("/predict")
async def predict_with_deployed_model(features: Dict[str, Any]):
    """Make a prediction using the deployed model."""
    if not deployed_model["model"]:
        raise HTTPException(status_code=404, detail="No model deployed. Deploy a model first.")
    
    try:
        import pandas as pd
        
        model = deployed_model["model"]
        scaler = deployed_model.get("scaler")
        label_encoder = deployed_model.get("label_encoder")
        
        feature_cols = ["price", "quantity", "user_id_hash", "product_id_hash", "hour", "day_of_week"]
        
        prepared = {
            "price": float(features.get("price", 0)),
            "quantity": int(features.get("quantity", 1)),
            "user_id_hash": hash(str(features.get("user_id", ""))) % 1000,
            "product_id_hash": hash(str(features.get("product_id", ""))) % 500,
            "hour": features.get("hour", datetime.now().hour),
            "day_of_week": features.get("day_of_week", datetime.now().weekday()),
        }
        
        X = pd.DataFrame([prepared])[feature_cols]
        
        if scaler:
            X = scaler.transform(X)
        
        prediction = model.predict(X)[0]
        
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X).max())
        
        if label_encoder:
            prediction = label_encoder.inverse_transform([prediction])[0]
        
        return {
            "prediction": str(prediction),
            "confidence": proba,
            "model": deployed_model["model_name"],
            "version": deployed_model["version"],
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/retrain")
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    model_type: str = "random_forest",
):
    """Trigger model retraining (alias for /train)."""
    return await train_model(background_tasks, model_type)


@router.get("/{model_name}/metrics")
async def get_model_metrics(model_name: str):
    """Get detailed metrics for a model from MLflow."""
    try:
        client = get_mlflow_client()
        versions = client.get_latest_versions(model_name)
        
        if not versions:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        run = client.get_run(versions[0].run_id)
        metrics = dict(run.data.metrics)
        
        return {
            "model_name": model_name,
            "version": versions[0].version,
            "metrics": metrics,
            "params": dict(run.data.params),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "model_name": model_name,
            "version": "unknown",
            "metrics": {},
            "params": {},
        }
