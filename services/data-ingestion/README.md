# Data Ingestion Service

Real-time data ingestion layer powered by Apache Kafka.

## Overview

This service handles:
- Event streaming from external sources
- Message validation and schema enforcement
- Producer and consumer implementations
- Kafka Connect integrations

## Structure

```
data-ingestion/
├── producers/           # Kafka producers
│   └── __init__.py
├── consumers/           # Kafka consumers
│   └── __init__.py
├── schemas/             # Avro/Protobuf schemas
│   └── events.avsc
├── connectors/          # Kafka Connect configs
│   └── source-connector.json
├── config.py            # Configuration
├── main.py              # Entry point
├── Dockerfile
└── README.md
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker addresses | `localhost:9092` |
| `SCHEMA_REGISTRY_URL` | Schema registry URL | `http://localhost:8081` |
| `CONSUMER_GROUP_ID` | Consumer group identifier | `data-ingestion-group` |

## Development

```bash
# Run locally
poetry run python main.py

# Run with Docker
docker build -t data-ingestion .
docker run -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 data-ingestion
```

## Topics

| Topic | Purpose | Partitions |
|-------|---------|------------|
| `raw-events` | Incoming raw events | 12 |
| `validated-events` | Schema-validated events | 12 |
| `dlq-events` | Dead letter queue | 3 |
