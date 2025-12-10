"""
Batch Ingest DAG

Daily Airflow DAG that:
1. Checks for new data files in data/sample/
2. Submits Spark job to process and write to Delta Lake
3. Runs data quality checks on the output

Schedule: Daily at 2:00 AM UTC
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule


# Configuration
PROJECT_ROOT = Path("/opt/airflow/project")  # Mount point in Airflow container
DATA_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample"
DELTA_LAKE_PATH = PROJECT_ROOT / "data" / "lake" / "delta" / "events"
SPARK_JOB_PATH = PROJECT_ROOT / "spark_jobs" / "batch_to_delta.py"

# Default arguments for all tasks
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email": ["data-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def check_source_files(**context) -> str:
    """
    Check if source data files exist.
    Returns the branch to execute based on file availability.
    """
    import os
    
    execution_date = context["ds"]
    
    # Check for JSONL and Parquet files
    jsonl_path = DATA_SAMPLE_PATH / "events.jsonl"
    parquet_path = DATA_SAMPLE_PATH / "events.parquet"
    
    jsonl_exists = jsonl_path.exists() if hasattr(jsonl_path, 'exists') else os.path.exists(str(jsonl_path))
    parquet_exists = parquet_path.exists() if hasattr(parquet_path, 'exists') else os.path.exists(str(parquet_path))
    
    if jsonl_exists or parquet_exists:
        print(f"Found source files for {execution_date}:")
        if jsonl_exists:
            print(f"  - {jsonl_path}")
        if parquet_exists:
            print(f"  - {parquet_path}")
        return "submit_spark_job"
    else:
        print(f"No source files found for {execution_date}, skipping processing")
        return "skip_processing"


def run_data_quality_checks(**context) -> dict:
    """
    Run data quality checks on the Delta Lake output.
    
    Checks:
    - Row count validation
    - Null constraint checks for required fields
    - Value range validation
    """
    import json
    
    execution_date = context["ds"]
    
    # Initialize results
    results = {
        "execution_date": execution_date,
        "checks_passed": True,
        "checks": []
    }
    
    try:
        # Try to use Delta Lake for reading
        try:
            from delta import DeltaTable
            from pyspark.sql import SparkSession
            
            spark = SparkSession.builder \
                .appName("DataQualityChecks") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .getOrCreate()
            
            # Read Delta table
            delta_path = str(DELTA_LAKE_PATH)
            df = spark.read.format("delta").load(delta_path)
            
            # Filter to today's partition
            df_today = df.filter(f"dt = '{execution_date}'")
            
        except ImportError:
            # Fallback: Read Parquet files directly
            import pandas as pd
            partition_path = DELTA_LAKE_PATH / f"dt={execution_date}"
            
            if partition_path.exists():
                df_today = pd.read_parquet(partition_path)
                row_count = len(df_today)
            else:
                print(f"No partition found for {execution_date}")
                row_count = 0
                df_today = None
        
        # Check 1: Row count > 0
        if hasattr(df_today, 'count'):
            row_count = df_today.count()
        elif df_today is not None:
            row_count = len(df_today)
        else:
            row_count = 0
            
        check_row_count = {
            "name": "row_count_check",
            "description": "Verify row count > 0",
            "expected": "> 0",
            "actual": row_count,
            "passed": row_count > 0
        }
        results["checks"].append(check_row_count)
        
        if row_count == 0:
            results["checks_passed"] = False
            print(f"FAILED: No rows found for partition dt={execution_date}")
            return results
        
        # Check 2: Null constraints for required fields
        required_fields = ["id", "timestamp", "user_id", "product_id", "price", "quantity", "location"]
        
        for field in required_fields:
            if hasattr(df_today, 'filter'):
                # Spark DataFrame
                null_count = df_today.filter(f"{field} IS NULL").count()
            else:
                # Pandas DataFrame
                null_count = df_today[field].isna().sum() if field in df_today.columns else row_count
            
            check_null = {
                "name": f"null_check_{field}",
                "description": f"Verify {field} has no nulls",
                "expected": 0,
                "actual": null_count,
                "passed": null_count == 0
            }
            results["checks"].append(check_null)
            
            if null_count > 0:
                results["checks_passed"] = False
                print(f"FAILED: {null_count} null values in {field}")
        
        # Check 3: Price range validation (must be positive)
        if hasattr(df_today, 'filter'):
            negative_prices = df_today.filter("price < 0").count()
        else:
            negative_prices = (df_today["price"] < 0).sum()
            
        check_price = {
            "name": "price_range_check",
            "description": "Verify all prices are non-negative",
            "expected": 0,
            "actual": negative_prices,
            "passed": negative_prices == 0
        }
        results["checks"].append(check_price)
        
        if negative_prices > 0:
            results["checks_passed"] = False
            print(f"FAILED: {negative_prices} negative prices found")
        
        # Check 4: Quantity validation (must be >= 1)
        if hasattr(df_today, 'filter'):
            invalid_qty = df_today.filter("quantity < 1").count()
        else:
            invalid_qty = (df_today["quantity"] < 1).sum()
            
        check_qty = {
            "name": "quantity_check",
            "description": "Verify all quantities are >= 1",
            "expected": 0,
            "actual": invalid_qty,
            "passed": invalid_qty == 0
        }
        results["checks"].append(check_qty)
        
        if invalid_qty > 0:
            results["checks_passed"] = False
            print(f"FAILED: {invalid_qty} invalid quantities found")
        
    except Exception as e:
        results["checks_passed"] = False
        results["error"] = str(e)
        print(f"Data quality checks failed with error: {e}")
    
    # Summary
    passed = sum(1 for c in results["checks"] if c["passed"])
    total = len(results["checks"])
    print(f"\nData Quality Summary: {passed}/{total} checks passed")
    
    # Push results to XCom
    context["ti"].xcom_push(key="dq_results", value=json.dumps(results))
    
    if not results["checks_passed"]:
        raise ValueError(f"Data quality checks failed: {passed}/{total} passed")
    
    return results


def notify_success(**context):
    """Send success notification."""
    execution_date = context["ds"]
    print(f"✅ Batch ingest completed successfully for {execution_date}")


def notify_failure(**context):
    """Send failure notification."""
    execution_date = context["ds"]
    print(f"❌ Batch ingest failed for {execution_date}")


# DAG Definition
with DAG(
    dag_id="batch_ingest_daily",
    description="Daily batch ingestion from sample data to Delta Lake",
    default_args=default_args,
    schedule_interval="0 2 * * *",  # Daily at 2:00 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ingest", "batch", "delta-lake", "spark"],
    doc_md="""
    ## Batch Ingest DAG
    
    Daily job that processes events from `data/sample/` and writes to Delta Lake.
    
    ### Pipeline Steps:
    1. **Check Source Files** - Verify input data exists
    2. **Spark Submit** - Run batch_to_delta.py for processing
    3. **Data Quality** - Validate row counts, null constraints, value ranges
    4. **Notification** - Alert on success/failure
    
    ### Configuration:
    - Input: `data/sample/events.jsonl` or `events.parquet`
    - Output: `data/lake/delta/events/` (partitioned by `dt=YYYY-MM-DD`)
    """,
) as dag:
    
    # Task: Start
    start = EmptyOperator(task_id="start")
    
    # Task: Check if source files exist
    check_files = BranchPythonOperator(
        task_id="check_source_files",
        python_callable=check_source_files,
        provide_context=True,
    )
    
    # Task: Skip processing (no files found)
    skip_processing = EmptyOperator(task_id="skip_processing")
    
    # Task: Submit Spark job
    submit_spark_job = BashOperator(
        task_id="submit_spark_job",
        bash_command=f"""
            spark-submit \
                --master local[*] \
                --packages io.delta:delta-core_2.12:2.4.0 \
                --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
                --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
                {SPARK_JOB_PATH} \
                --input-path {DATA_SAMPLE_PATH} \
                --output-path {DELTA_LAKE_PATH} \
                --date {{{{ ds }}}}
        """,
        retries=2,
    )
    
    # Task: Run data quality checks
    data_quality = PythonOperator(
        task_id="data_quality_checks",
        python_callable=run_data_quality_checks,
        provide_context=True,
    )
    
    # Task: Notify success
    notify_ok = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    
    # Task: Notify failure
    notify_fail = PythonOperator(
        task_id="notify_failure",
        python_callable=notify_failure,
        provide_context=True,
        trigger_rule=TriggerRule.ONE_FAILED,
    )
    
    # Task: End
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    
    # Define task dependencies
    start >> check_files
    check_files >> [submit_spark_job, skip_processing]
    submit_spark_job >> data_quality >> notify_ok >> end
    skip_processing >> end
    [submit_spark_job, data_quality] >> notify_fail
