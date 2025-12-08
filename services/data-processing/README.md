# Data Processing Service

Batch and stream processing powered by Apache Spark and Airflow.

## Overview

This service handles:
- Spark Streaming for real-time processing
- Spark SQL for structured data transformations
- Airflow DAGs for workflow orchestration
- Data quality validation

## Structure

```
data-processing/
├── spark_jobs/          # Spark applications
│   ├── __init__.py
│   ├── streaming_job.py
│   └── batch_job.py
├── dags/                # Airflow DAG definitions
│   ├── __init__.py
│   ├── etl_pipeline.py
│   └── feature_pipeline.py
├── operators/           # Custom Airflow operators
│   └── spark_operator.py
├── transformations/     # Data transformation logic
│   └── transforms.py
├── config.py
├── Dockerfile
└── README.md
```

## Airflow DAGs

| DAG | Schedule | Purpose |
|-----|----------|---------|
| `etl_pipeline` | `0 */2 * * *` | Bronze → Silver ETL |
| `feature_pipeline` | `0 4 * * *` | Feature materialization |
| `data_quality` | `0 6 * * *` | Data quality checks |

## Spark Jobs

| Job | Type | Purpose |
|-----|------|---------|
| `streaming_job` | Streaming | Real-time event processing |
| `batch_job` | Batch | Historical data processing |

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SPARK_MASTER` | Spark master URL | `local[*]` |
| `AIRFLOW_HOME` | Airflow configuration directory | `/opt/airflow` |
| `DELTA_LAKE_PATH` | Delta Lake storage path | `s3://bucket/delta` |

## Development

```bash
# Submit Spark job
spark-submit spark_jobs/streaming_job.py

# Run Airflow locally
airflow standalone
```
