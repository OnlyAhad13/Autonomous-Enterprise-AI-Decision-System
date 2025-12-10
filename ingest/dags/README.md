# Ingest DAGs

Airflow DAGs for data ingestion pipelines.

## DAGs

| DAG ID | Schedule | Description |
|--------|----------|-------------|
| `batch_ingest_daily` | Daily 2:00 AM UTC | Batch ingest from sample data to Delta Lake |

## batch_ingest_daily

Daily batch ingestion pipeline that:

1. **Checks source files** - Verifies `data/sample/events.jsonl` or `events.parquet` exists
2. **Submits Spark job** - Runs `batch_to_delta.py` for processing
3. **Data quality checks** - Validates row counts, null constraints, value ranges
4. **Notifications** - Alerts on success/failure

### Pipeline Diagram

```
start
  │
  ▼
check_source_files ──────────────────┐
  │                                  │
  ▼                                  ▼
submit_spark_job              skip_processing
  │                                  │
  ▼                                  │
data_quality_checks                  │
  │                                  │
  ├───────────┐                      │
  ▼           ▼                      │
notify_success  notify_failure       │
  │                                  │
  └──────────────┬───────────────────┘
                 ▼
                end
```

### Data Quality Checks

| Check | Rule | Action |
|-------|------|--------|
| Row count | > 0 | Fail if no rows |
| Null constraints | Required fields not null | Fail if nulls found |
| Price range | price >= 0 | Fail if negative |
| Quantity range | quantity >= 1 | Fail if < 1 |

### Configuration

Modify paths in `batch_ingest_dag.py`:

```python
PROJECT_ROOT = Path("/opt/airflow/project")
DATA_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample"
DELTA_LAKE_PATH = PROJECT_ROOT / "data" / "lake" / "delta" / "events"
```

## Running Locally with Docker

```bash
# Start Airflow
cd infra
docker compose -f docker-compose.airflow.yml up -d

# Access web UI
open http://localhost:8085
# Login: airflow / airflow

# Trigger DAG manually
docker exec airflow-scheduler airflow dags trigger batch_ingest_daily

# View logs
docker exec airflow-scheduler airflow tasks logs batch_ingest_daily submit_spark_job

# Stop Airflow
docker compose -f docker-compose.airflow.yml down
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIRFLOW_UID` | 50000 | Airflow user ID |
| `_AIRFLOW_WWW_USER_USERNAME` | airflow | Web UI username |
| `_AIRFLOW_WWW_USER_PASSWORD` | airflow | Web UI password |
