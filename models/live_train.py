"""
Live Model Training - Train models on streamed event data.

This module provides functions to:
1. Train classification models (e.g., event type prediction, risk scoring)
2. Train on data from the live event buffer
3. Register models with MLflow
4. Make models available for predictions
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# MLflow
import mlflow
import mlflow.sklearn

# Path for models
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class LiveModelTrainer:
    """Train models on live event data."""
    
    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "live-event-models",
    ):
        self.mlflow_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def prepare_features_from_events(
        self,
        events: List[Dict],
        target_column: str = "event_type",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform raw events into features for training.
        
        Args:
            events: List of event dictionaries from Kafka
            target_column: Column to predict
            
        Returns:
            X (features), y (target)
        """
        if not events:
            raise ValueError("No events provided for training")
        
        # Extract features from events
        records = []
        for event in events:
            value = event.get("value", event)
            if isinstance(value, dict):
                try:
                    ts = event.get("timestamp", datetime.now().isoformat()).replace("Z", "")
                    dt = datetime.fromisoformat(ts)
                    hour = dt.hour
                    dow = dt.weekday()
                except:
                    hour = 12
                    dow = 0
                
                record = {
                    "event_type": value.get("event_type", "unknown"),
                    "price": float(value.get("price", 0) or value.get("total", 0) or 0),
                    "quantity": int(value.get("quantity", 1) or 1),
                    "user_id_hash": hash(str(value.get("user_id", ""))) % 1000,
                    "product_id_hash": hash(str(value.get("product_id", ""))) % 500,
                    "hour": hour,
                    "day_of_week": dow,
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        
        if df.empty:
            raise ValueError("Could not extract features from events")
        
        # Encode target
        if target_column not in self.label_encoders:
            self.label_encoders[target_column] = LabelEncoder()
            y = self.label_encoders[target_column].fit_transform(df[target_column])
        else:
            y = self.label_encoders[target_column].transform(df[target_column])
        
        # Features (excluding target)
        feature_cols = ["price", "quantity", "user_id_hash", "product_id_hash", "hour", "day_of_week"]
        X = df[feature_cols].fillna(0)
        
        return X, pd.Series(y, name=target_column)
    
    def train_event_classifier(
        self,
        events: List[Dict],
        model_type: str = "random_forest",
        target: str = "event_type",
    ) -> Dict[str, Any]:
        """
        Train a classifier on event data.
        
        Args:
            events: List of events from the buffer
            model_type: One of 'random_forest', 'gradient_boosting', 'logistic_regression'
            target: Target column to predict
            
        Returns:
            Dictionary with model info, metrics, and paths
        """
        print(f"📊 Training {model_type} on {len(events)} events...")
        
        # Prepare data
        X, y = self.prepare_features_from_events(events, target)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=200)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        start_time = datetime.now()
        model.fit(X_train_scaled, y_train)
        train_duration = (datetime.now() - start_time).total_seconds()
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "training_duration_seconds": train_duration,
            "num_samples": len(events),
            "num_classes": len(self.label_encoders.get(target, LabelEncoder()).classes_) if target in self.label_encoders else 0,
        }
        
        # Log to MLflow
        run_name = f"{model_type}_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log params
            mlflow.log_params({
                "model_type": model_type,
                "target": target,
                "num_samples": len(events),
                "test_size": 0.2,
            })
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"live-{model_type}"
            )
            
            run_id = mlflow.active_run().info.run_id
        
        # Save locally
        model_path = MODEL_DIR / f"{model_type}_{target}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoders.get(target),
                "feature_columns": list(X.columns),
            }, f)
        
        print(f"✅ Model trained: {model_type}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1 Score: {metrics['f1_weighted']:.4f}")
        print(f"   Saved to: {model_path}")
        print(f"   MLflow run: {run_id}")
        
        return {
            "model_type": model_type,
            "target": target,
            "metrics": metrics,
            "model_path": str(model_path),
            "mlflow_run_id": run_id,
            "classes": list(self.label_encoders[target].classes_) if target in self.label_encoders else [],
            "timestamp": datetime.now().isoformat(),
        }
    
    def load_model(self, model_path: str) -> Dict:
        """Load a saved model."""
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    def predict(self, model_path: str, features: Dict) -> Dict:
        """Make prediction with saved model."""
        model_data = self.load_model(model_path)
        model = model_data["model"]
        scaler = model_data["scaler"]
        label_encoder = model_data["label_encoder"]
        
        # Prepare features
        feature_cols = model_data["feature_columns"]
        X = pd.DataFrame([features])[feature_cols].fillna(0)
        X_scaled = scaler.transform(X)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
        
        result = {
            "prediction": label_encoder.inverse_transform([pred])[0] if label_encoder else int(pred),
            "confidence": float(max(proba)) if proba is not None else None,
        }
        
        return result


def train_on_live_buffer(
    events: List[Dict],
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """
    Convenience function to train on live event buffer.
    
    Args:
        events: Events from the ingestion buffer
        model_type: Model type to train
        
    Returns:
        Training result dictionary
    """
    trainer = LiveModelTrainer()
    return trainer.train_event_classifier(events, model_type)


if __name__ == "__main__":
    # Demo: Generate sample events and train
    import random
    
    sample_events = []
    event_types = ["order_placed", "page_view", "purchase", "add_to_cart", "checkout_started"]
    
    for i in range(500):
        event = {
            "id": f"evt_{i}",
            "timestamp": datetime.now().isoformat(),
            "value": {
                "event_type": random.choice(event_types),
                "price": round(random.uniform(10, 500), 2),
                "quantity": random.randint(1, 5),
                "user_id": f"U{random.randint(1000, 9999)}",
                "product_id": f"P{random.randint(100, 999)}",
            }
        }
        sample_events.append(event)
    
    # Train
    result = train_on_live_buffer(sample_events, "random_forest")
    print("\n📋 Training Result:")
    print(json.dumps(result, indent=2, default=str))
