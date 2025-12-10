# Spark Jobs

PySpark jobs for data processing and transformation.

## Overview

| Job | Purpose | Input | Output |
|-----|---------|-------|--------|
| `batch_to_delta.py` | Batch processing to Delta Lake | JSONL/Parquet | Delta Lake |

## batch_to_delta.py

Reads events from JSONL or Parquet files, normalizes fields, and writes partitioned Delta files.

### Transformations

1. **Timestamp parsing** - Convert ISO string to TimestampType
2. **Date extraction** - Extract `dt=YYYY-MM-DD` for partitioning
3. **ID normalization** - Ensure `usr_` and `prod_` prefixes
4. **Location parsing** - Split into city and country
5. **Metadata extraction** - Parse JSON and extract common fields
6. **Total calculation** - Compute `total_amount = price * quantity`

### Usage

```bash
# Local mode
spark-submit \
    --packages io.delta:delta-core_2.12:2.4.0 \
    --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
    --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
    spark_jobs/batch_to_delta.py \
    --input-path data/sample \
    --output-path data/lake/delta/events

# With date filter
spark-submit \
    --packages io.delta:delta-core_2.12:2.4.0 \
    spark_jobs/batch_to_delta.py \
    --input-path data/sample \
    --output-path data/lake/delta/events \
    --date 2024-01-15 \
    --mode overwrite
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-path` | Yes | - | Path to input data |
| `--output-path` | Yes | - | Path to Delta output |
| `--date` | No | None | Filter to specific date |
| `--mode` | No | append | Write mode (append/overwrite) |

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Event UUID |
| `event_timestamp` | timestamp | Parsed event time |
| `dt` | string | Partition date (YYYY-MM-DD) |
| `user_id` | string | Normalized user ID |
| `product_id` | string | Normalized product ID |
| `price` | double | Unit price |
| `quantity` | integer | Item count |
| `total_amount` | double | price × quantity |
| `location` | string | Original location |
| `city` | string | Extracted city |
| `country` | string | Extracted country |
| `channel` | string | Sales channel |
| `device_type` | string | Device type |
| `session_id` | string | Session ID |
| `metadata` | string | Raw JSON metadata |
| `processed_at` | timestamp | Processing timestamp |
| `processing_date` | date | Processing date |

### Dependencies

```bash
pip install pyspark==3.5.0 delta-spark==2.4.0
```
