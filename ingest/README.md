# Ingest Service

Kafka-based data ingestion for streaming business events.

## Overview

This service provides:
- Kafka producer for publishing events
- Schema Registry integration for JSON schema validation
- Batch processing with configurable throughput

## Quick Start

### 1. Start Kafka Infrastructure

```bash
# Start Zookeeper, Kafka, and Schema Registry
cd infra
chmod +x run.sh
./run.sh start

# Create default topics
./run.sh create-topics

# Check health
./run.sh health
```

### 2. Generate Sample Data

```bash
# Generate sample events
python data/sample_generator.py --count 50000 --seed 42
```

### 3. Produce Events to Kafka

```bash
# Install dependencies
pip install confluent-kafka[json,avro]

# Produce events to Kafka
python ingest/kafka_producer.py --input data/sample/events.jsonl

# With options
python ingest/kafka_producer.py \
    --input data/sample/events.jsonl \
    --topic events.raw.v1 \
    --batch-size 1000 \
    --limit 10000
```

## Testing with kafka-console-consumer

Verify messages are being produced correctly:

```bash
# Consume first 10 messages from beginning
docker exec enterprise-ai-kafka kafka-console-consumer \
    --bootstrap-server localhost:9093 \
    --topic events.raw.v1 \
    --from-beginning \
    --max-messages 10

# Or use the run.sh helper
cd infra
./run.sh consume events.raw.v1 10
```

### Expected Output

```json
{"id": "550e8400-e29b-41d4-a716-446655440000", "timestamp": "2024-11-15T10:30:00.000Z", "user_id": "usr_abc123", "product_id": "prod_xyz789", "price": 99.99, "quantity": 2, "location": "New York, USA", "metadata": {"channel": "web", "device_type": "desktop", "session_id": "sess_123abc"}}
```

## Schema Registry

View registered schemas:

```bash
# List all subjects
curl http://localhost:8081/subjects

# Get schema for topic
curl http://localhost:8081/subjects/events.raw.v1-value/versions/latest
```

## Endpoints

| Service | URL |
|---------|-----|
| Kafka Bootstrap | `localhost:9093` |
| Schema Registry | `http://localhost:8081` |
| Kafka UI (optional) | `http://localhost:8080` |

## Configuration

Environment variables for the producer:

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9093` | Kafka broker addresses |
| `SCHEMA_REGISTRY_URL` | `http://localhost:8081` | Schema Registry URL |

## Cleanup

```bash
# Stop all services
cd infra
./run.sh stop
```
