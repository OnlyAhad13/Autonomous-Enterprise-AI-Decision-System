# E2E and Chaos Testing

End-to-end and chaos engineering tests for the Enterprise AI Platform.

## Quick Start

```bash
# Run E2E tests locally
cd tests/e2e
docker-compose -f docker-compose.e2e.yml up -d
E2E_MODE=true pytest test_pipeline_e2e.py -v

# Run chaos tests (mocked, no Docker needed)
pytest tests/chaos/ -v

# Cleanup
docker-compose -f docker-compose.e2e.yml down -v
```

## E2E Tests

| Test | Description |
|------|-------------|
| `test_pipeline_e2e.py` | Full ingest → stream → feature → train → serve → predict |

### KPI Thresholds

| Metric | Threshold |
|--------|-----------|
| Ingest latency P99 | < 500ms |
| Prediction latency P99 | < 200ms |
| Error rate | < 1% |
| Throughput | > 100 events/sec |

## Chaos Tests

| Test | Description |
|------|-------------|
| `test_kafka_failure.py` | Kafka downtime resilience |
| `test_agent_retry.py` | Retry policy and alerting |

## Running in CI

### GitHub Actions

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  e2e:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Start infrastructure
        run: |
          docker-compose -f tests/e2e/docker-compose.e2e.yml up -d
          sleep 30  # Wait for services
      
      - name: Run E2E tests
        env:
          E2E_MODE: "true"
          KAFKA_BOOTSTRAP_SERVERS: localhost:19092
          MLFLOW_TRACKING_URI: http://localhost:15000
          PREDICT_API_URL: http://localhost:18000
        run: |
          pip install pytest requests kafka-python
          pytest tests/e2e/ -v --tb=short
      
      - name: Cleanup
        if: always()
        run: docker-compose -f tests/e2e/docker-compose.e2e.yml down -v

  chaos:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Run chaos tests (mocked)
        run: |
          pip install pytest
          pytest tests/chaos/ -v --tb=short
```

### Mock Credentials

For CI without real infrastructure:

```bash
# Environment variables for mocked tests
export E2E_MODE=false  # Use mocked fixtures
export MOCK_KAFKA=true
export MOCK_MLFLOW=true
```

### Service Wait Script

```bash
#!/bin/bash
# scripts/wait-for-services.sh

wait_for_service() {
    local url=$1
    local timeout=${2:-60}
    local start=$(date +%s)
    
    while true; do
        if curl -sf "$url/health" > /dev/null 2>&1; then
            echo "✓ $url is ready"
            return 0
        fi
        
        if [ $(($(date +%s) - start)) -gt $timeout ]; then
            echo "✗ Timeout waiting for $url"
            return 1
        fi
        
        sleep 2
    done
}

wait_for_service "http://localhost:18000" 60
wait_for_service "http://localhost:15000" 60
```

## Troubleshooting

### Services not starting
```bash
docker-compose -f tests/e2e/docker-compose.e2e.yml logs
```

### Kafka connection issues
```bash
# Check Kafka is ready
docker exec e2e-kafka kafka-topics --bootstrap-server localhost:9092 --list
```

### Reset environment
```bash
docker-compose -f tests/e2e/docker-compose.e2e.yml down -v
docker system prune -f
```
