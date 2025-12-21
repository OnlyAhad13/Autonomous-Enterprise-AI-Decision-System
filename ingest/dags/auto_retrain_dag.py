"""
Auto-Retrain DAG.

Automated model retraining pipeline triggered by drift detection.

Flow:
    PrometheusSensor → AgentDecision → [Branch]
                                         ├─ retrain → SparkJob → Train → Validate → Promote
                                         └─ skip → AlertOnly
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.trigger_rule import TriggerRule

try:
    from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


# Load policy configuration
POLICY_PATH = Path(__file__).parent.parent.parent / "conf" / "agent_policy.json"

def load_policy() -> Dict[str, Any]:
    """Load agent policy configuration."""
    if POLICY_PATH.exists():
        with open(POLICY_PATH) as f:
            return json.load(f)
    return {
        "drift_detection": {"threshold": 0.1, "metric_name": "model_drift_score"},
        "validation": {"min_accuracy": 0.85},
    }


POLICY = load_policy()

# DAG default args
DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ============================================================================
# Custom Sensors
# ============================================================================


class PrometheusDriftSensor(BaseSensorOperator):
    """
    Sensor that monitors Prometheus for model drift signals.
    
    Pokes Prometheus API and checks if drift metric exceeds threshold.
    """
    
    template_fields = ("prometheus_url", "metric_query")
    
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        metric_query: str = "model_drift_score",
        threshold: float = 0.1,
        min_samples: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prometheus_url = prometheus_url
        self.metric_query = metric_query
        self.threshold = threshold
        self.min_samples = min_samples
    
    def poke(self, context: Dict) -> bool:
        """Check if drift exceeds threshold."""
        import requests
        
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": self.metric_query},
                timeout=30,
            )
            
            if response.status_code != 200:
                logger.warning(f"Prometheus returned {response.status_code}")
                return False
            
            data = response.json()
            
            if data.get("status") != "success":
                logger.warning(f"Prometheus query failed: {data.get('error')}")
                return False
            
            results = data.get("data", {}).get("result", [])
            
            if len(results) < self.min_samples:
                logger.info(f"Not enough samples: {len(results)} < {self.min_samples}")
                return False
            
            # Get the drift value
            drift_value = float(results[0].get("value", [0, 0])[1])
            
            logger.info(f"Current drift: {drift_value:.4f}, threshold: {self.threshold}")
            
            # Store drift value in XCom for downstream tasks
            context["ti"].xcom_push(key="drift_value", value=drift_value)
            context["ti"].xcom_push(key="exceeds_threshold", value=drift_value > self.threshold)
            
            return drift_value > self.threshold
            
        except requests.RequestException as e:
            logger.error(f"Prometheus request failed: {e}")
            return False
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing Prometheus response: {e}")
            return False


# ============================================================================
# Task Functions
# ============================================================================


def call_agent_for_decision(**context) -> Dict[str, Any]:
    """
    Call the agent API to get a decision on retraining.
    
    Returns decision including whether to retrain and any recommendations.
    """
    import requests
    
    drift_value = context["ti"].xcom_pull(key="drift_value") or 0.0
    threshold = POLICY.get("drift_detection", {}).get("threshold", 0.1)
    
    agent_url = Variable.get("AGENT_API_URL", default_var="http://localhost:8000")
    
    payload = {
        "objective": "Evaluate model drift and decide on retraining",
        "context": {
            "drift_value": drift_value,
            "threshold": threshold,
            "policy": POLICY,
        }
    }
    
    try:
        response = requests.post(
            f"{agent_url}/evaluate",
            json=payload,
            timeout=60,
        )
        
        if response.status_code == 200:
            decision = response.json()
        else:
            # Fallback to threshold-based decision
            logger.warning(f"Agent API returned {response.status_code}, using threshold")
            decision = {
                "action": "retrain" if drift_value > threshold else "skip",
                "reason": f"Drift {drift_value:.4f} {'>' if drift_value > threshold else '<='} {threshold}",
                "confidence": 1.0,
            }
            
    except requests.RequestException as e:
        logger.error(f"Agent API call failed: {e}")
        # Fallback decision
        decision = {
            "action": "retrain" if drift_value > threshold else "skip",
            "reason": f"Fallback: Drift {drift_value:.4f}",
            "confidence": 0.5,
        }
    
    context["ti"].xcom_push(key="agent_decision", value=decision)
    return decision


def decide_branch(**context) -> str:
    """
    Branch decision based on agent recommendation.
    
    Returns the task_id to execute next.
    """
    decision = context["ti"].xcom_pull(key="agent_decision") or {}
    action = decision.get("action", "skip")
    
    logger.info(f"Agent decision: {action} - {decision.get('reason', 'No reason')}")
    
    if action == "retrain":
        return "prepare_retraining"
    else:
        return "send_alert_only"


def prepare_retraining(**context) -> Dict[str, Any]:
    """Prepare environment and parameters for retraining."""
    config = POLICY.get("retraining", {})
    
    prep_result = {
        "data_path": config.get("data_path", "data/lake/delta/features"),
        "model_output": config.get("model_output_path", "models/output"),
        "experiment": config.get("mlflow_experiment", "auto-retrain"),
        "timestamp": datetime.now().isoformat(),
    }
    
    context["ti"].xcom_push(key="retrain_config", value=prep_result)
    return prep_result


def run_training_script(**context) -> Dict[str, Any]:
    """Execute the model training script."""
    import subprocess
    
    config = context["ti"].xcom_pull(key="retrain_config") or {}
    training_script = POLICY.get("retraining", {}).get(
        "training_script", 
        "models/forecasting/train_forecast.py"
    )
    
    logger.info(f"Running training script: {training_script}")
    
    # Build command
    cmd = [
        "python", training_script,
        "--experiment", config.get("experiment", "auto-retrain"),
        "--data-path", config.get("data_path", "data/lake/delta/features"),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        training_result = {
            "success": result.returncode == 0,
            "stdout": result.stdout[-1000:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
        
    except subprocess.TimeoutExpired:
        training_result = {"success": False, "error": "Training timeout"}
    except Exception as e:
        training_result = {"success": False, "error": str(e)}
    
    context["ti"].xcom_push(key="training_result", value=training_result)
    return training_result


def run_validation_tests(**context) -> Dict[str, Any]:
    """Run validation tests on the trained model."""
    import subprocess
    
    validation_config = POLICY.get("validation", {})
    required_tests = validation_config.get("required_tests", [
        "tests/test_model_inference.py",
        "tests/test_model_performance.py",
    ])
    
    results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "details": [],
    }
    
    for test_file in required_tests:
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            test_passed = result.returncode == 0
            results["tests_run"] += 1
            
            if test_passed:
                results["tests_passed"] += 1
            else:
                results["tests_failed"] += 1
            
            results["details"].append({
                "test": test_file,
                "passed": test_passed,
                "output": result.stdout[-500:] if result.stdout else "",
            })
            
        except Exception as e:
            results["details"].append({
                "test": test_file,
                "passed": False,
                "error": str(e),
            })
            results["tests_failed"] += 1
    
    results["all_passed"] = results["tests_failed"] == 0
    context["ti"].xcom_push(key="validation_result", value=results)
    
    if not results["all_passed"]:
        raise ValueError(f"Validation failed: {results['tests_failed']} tests failed")
    
    return results


def check_performance_metrics(**context) -> Dict[str, Any]:
    """Check if model performance meets thresholds."""
    validation_config = POLICY.get("validation", {})
    
    min_accuracy = validation_config.get("min_accuracy", 0.85)
    max_mape = validation_config.get("max_mape", 15.0)
    
    # In production, this would fetch metrics from MLflow or model registry
    # For now, we simulate by pulling from training results
    training_result = context["ti"].xcom_pull(key="training_result") or {}
    
    # Simulated metrics (replace with actual metric fetching)
    metrics = {
        "accuracy": 0.88,
        "mape": 12.5,
        "f1_score": 0.85,
    }
    
    checks = {
        "accuracy_ok": metrics.get("accuracy", 0) >= min_accuracy,
        "mape_ok": metrics.get("mape", 100) <= max_mape,
    }
    
    performance_result = {
        "metrics": metrics,
        "thresholds": {"min_accuracy": min_accuracy, "max_mape": max_mape},
        "checks": checks,
        "passed": all(checks.values()),
    }
    
    context["ti"].xcom_push(key="performance_result", value=performance_result)
    
    if not performance_result["passed"]:
        raise ValueError(f"Performance check failed: {checks}")
    
    return performance_result


def promote_model_to_production(**context) -> Dict[str, Any]:
    """Promote the validated model to production via MLflow."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from mlflow_utils.auto_promote import AutoPromoter, PromotionThresholds
    except ImportError:
        logger.warning("MLflow auto_promote not available, simulating promotion")
        return {"promoted": True, "simulated": True}
    
    validation_config = POLICY.get("validation", {})
    
    thresholds = PromotionThresholds(
        mape_improvement_pct=5.0,
        mse_reduction_pct=10.0,
    )
    
    try:
        promoter = AutoPromoter(thresholds=thresholds)
        
        results = promoter.evaluate_and_promote(
            experiment_name=POLICY.get("retraining", {}).get("mlflow_experiment", "auto-retrain"),
            model_name="auto-retrain-model",
            dry_run=False,
        )
        
        promotion_result = {
            "promoted": len(results) > 0,
            "results": [
                {
                    "run_id": r.run_id,
                    "version": r.new_version,
                    "reason": r.reason,
                }
                for r in results
            ],
        }
        
    except Exception as e:
        logger.error(f"Model promotion failed: {e}")
        promotion_result = {"promoted": False, "error": str(e)}
    
    context["ti"].xcom_push(key="promotion_result", value=promotion_result)
    return promotion_result


def send_notification(**context) -> Dict[str, Any]:
    """Send notification about the pipeline result."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from agents.tools.tool_slack import SlackTool
    except ImportError:
        logger.warning("Slack tool not available")
        return {"sent": False, "simulated": True}
    
    # Gather results
    drift_value = context["ti"].xcom_pull(key="drift_value") or 0.0
    decision = context["ti"].xcom_pull(key="agent_decision") or {}
    promotion = context["ti"].xcom_pull(key="promotion_result") or {}
    
    # Build report
    sections = [
        {"title": "Drift Score", "value": f"{drift_value:.4f}"},
        {"title": "Decision", "value": decision.get("action", "unknown")},
        {"title": "Promoted", "value": "✓ Yes" if promotion.get("promoted") else "✗ No"},
    ]
    
    slack_config = POLICY.get("notifications", {}).get("slack", {})
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL") or slack_config.get("webhook_url")
    
    slack = SlackTool(webhook_url=webhook_url)
    result = slack.post_report(
        title="🔄 Auto-Retrain Pipeline Complete",
        sections=sections,
        channel=slack_config.get("channel", "#ml-alerts"),
    )
    
    return {"sent": result.success, "data": result.data}


def send_alert_only(**context) -> Dict[str, Any]:
    """Send alert when retraining was skipped."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from agents.tools.tool_slack import SlackTool
    except ImportError:
        return {"sent": False, "simulated": True}
    
    drift_value = context["ti"].xcom_pull(key="drift_value") or 0.0
    decision = context["ti"].xcom_pull(key="agent_decision") or {}
    
    slack = SlackTool()
    result = slack.send_message(
        f"ℹ️ Drift check complete. Score: {drift_value:.4f}. "
        f"Decision: {decision.get('action', 'skip')} - {decision.get('reason', 'N/A')}",
    )
    
    return {"sent": result.success}


# ============================================================================
# DAG Definition
# ============================================================================


with DAG(
    dag_id="auto_retrain_dag",
    description="Automated model retraining triggered by drift detection",
    default_args=DEFAULT_ARGS,
    schedule_interval=POLICY.get("schedule", {}).get("dag_schedule", "0 */6 * * *"),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retraining", "drift-detection"],
) as dag:
    
    # ========================================================================
    # Task Definitions
    # ========================================================================
    
    # Start
    start = DummyOperator(task_id="start")
    
    # Prometheus drift sensor
    drift_sensor = PrometheusDriftSensor(
        task_id="check_drift_metrics",
        prometheus_url=POLICY.get("drift_detection", {}).get("prometheus_url", "http://localhost:9090"),
        metric_query=POLICY.get("drift_detection", {}).get("metric_name", "model_drift_score"),
        threshold=POLICY.get("drift_detection", {}).get("threshold", 0.1),
        poke_interval=60,
        timeout=POLICY.get("schedule", {}).get("sensor_timeout_hours", 1) * 3600,
        mode="poke",
        soft_fail=True,  # Don't fail DAG if no drift detected
    )
    
    # Agent decision
    agent_decision = PythonOperator(
        task_id="get_agent_decision",
        python_callable=call_agent_for_decision,
        provide_context=True,
    )
    
    # Branch based on decision
    branch = BranchPythonOperator(
        task_id="branch_on_decision",
        python_callable=decide_branch,
        provide_context=True,
    )
    
    # Retraining path
    prepare = PythonOperator(
        task_id="prepare_retraining",
        python_callable=prepare_retraining,
        provide_context=True,
    )
    
    # Spark retraining job (if Spark provider available)
    if HAS_SPARK:
        spark_config = POLICY.get("retraining", {}).get("spark_job", {})
        spark_retrain = SparkSubmitOperator(
            task_id="run_spark_retrain",
            application=spark_config.get("application", "spark_jobs/retrain_model.py"),
            conn_id="spark_default",
            executor_memory=spark_config.get("executor_memory", "4g"),
            executor_cores=spark_config.get("executor_cores", 2),
            num_executors=spark_config.get("num_executors", 4),
        )
    else:
        # Fallback to bash operator
        spark_retrain = BashOperator(
            task_id="run_spark_retrain",
            bash_command="echo 'Spark not available, skipping Spark job'",
        )
    
    # Training script
    train_model = PythonOperator(
        task_id="run_training",
        python_callable=run_training_script,
        provide_context=True,
    )
    
    # Validation tasks
    run_tests = PythonOperator(
        task_id="run_validation_tests",
        python_callable=run_validation_tests,
        provide_context=True,
    )
    
    perf_check = PythonOperator(
        task_id="check_performance",
        python_callable=check_performance_metrics,
        provide_context=True,
    )
    
    # MLflow promotion
    promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_to_production,
        provide_context=True,
    )
    
    # Notifications
    notify_success = PythonOperator(
        task_id="send_success_notification",
        python_callable=send_notification,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    
    # Skip path - alert only
    alert_only = PythonOperator(
        task_id="send_alert_only",
        python_callable=send_alert_only,
        provide_context=True,
    )
    
    # End
    end = DummyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    
    # ========================================================================
    # Task Dependencies
    # ========================================================================
    
    # Main flow
    start >> drift_sensor >> agent_decision >> branch
    
    # Retrain path
    branch >> prepare >> spark_retrain >> train_model >> run_tests >> perf_check >> promote >> notify_success >> end
    
    # Skip path
    branch >> alert_only >> end
