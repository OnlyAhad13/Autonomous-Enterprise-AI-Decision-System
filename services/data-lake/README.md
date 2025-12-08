# Data Lake Service

ACID-compliant data lake powered by Delta Lake.

## Overview

This service manages:
- Medallion architecture (Bronze, Silver, Gold)
- Schema enforcement and evolution
- Time travel and versioning
- Data compaction and optimization

## Structure

```
data-lake/
├── schemas/             # Table schema definitions
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── migrations/          # Schema migration scripts
│   └── v1_initial.py
├── compaction/          # Optimization jobs
│   └── optimize.py
├── retention/           # Data lifecycle policies
│   └── cleanup.py
├── config.py
└── README.md
```

## Medallion Architecture

| Layer | Purpose | Retention |
|-------|---------|-----------|
| **Bronze** | Raw, immutable data | 90 days |
| **Silver** | Cleaned, validated data | 1 year |
| **Gold** | Aggregated, business-ready | 3 years |

## Tables

### Bronze Layer
- `bronze.raw_events` - Raw event data
- `bronze.raw_entities` - Raw entity data

### Silver Layer
- `silver.events` - Validated events
- `silver.entities` - Validated entities
- `silver.relationships` - Entity relationships

### Gold Layer
- `gold.daily_metrics` - Daily aggregations
- `gold.feature_snapshots` - Feature store snapshots

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DELTA_LAKE_PATH` | Base path for Delta tables | `s3://bucket/delta` |
| `CHECKPOINT_PATH` | Streaming checkpoint location | `s3://bucket/checkpoints` |

## Operations

```bash
# Optimize tables
poetry run python compaction/optimize.py --table silver.events

# Run retention cleanup
poetry run python retention/cleanup.py --dry-run
```
