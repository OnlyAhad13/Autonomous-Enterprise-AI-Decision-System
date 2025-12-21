# Monitoring and Alerting Guide

Prometheus + Grafana + Alertmanager setup for ML Platform observability.

## Table of Contents

- [Quick Setup](#quick-setup)
- [Instrumenting FastAPI](#instrumenting-fastapi)
- [Instrumenting Spark Jobs](#instrumenting-spark-jobs)
- [Alert Configuration](#alert-configuration)
- [Dashboard Import](#dashboard-import)

---

## Quick Setup

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Verify Prometheus
curl http://localhost:9090/-/healthy

# Verify Alertmanager
curl http://localhost:9093/-/healthy

# Access Grafana
open http://localhost:3000  # admin/admin
```

---

## Instrumenting FastAPI

### 1. Add Metrics Module

```python
# In your FastAPI app
from services.predict.metrics import setup_metrics

app = FastAPI()
setup_metrics(app, service_name="predict")
```

### 2. Track Custom Metrics

```python
from services.predict.metrics import (
    track_prediction,
    update_drift_score,
)

class ModelManager:
    @track_prediction(model_version="1.0.0")
    def predict(self, features):
        return model.predict(features)

# Update drift score
update_drift_score("forecasting-model", 0.08)
```

### 3. Verify Metrics

```bash
curl http://localhost:8000/metrics
```

---

## Instrumenting Spark Jobs

### Option 1: Prometheus Pushgateway

```python
# spark_job.py
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    push_to_gateway,
)

registry = CollectorRegistry()

# Define metrics
records_processed = Counter(
    'spark_records_processed_total',
    'Total records processed',
    ['job_name'],
    registry=registry,
)

job_duration = Gauge(
    'spark_job_duration_seconds',
    'Job duration in seconds',
    ['job_name'],
    registry=registry,
)

# In your Spark job
def run_spark_job():
    start_time = time.time()
    
    # Process data
    df = spark.read.parquet("s3://bucket/data")
    count = df.count()
    
    # Update metrics
    records_processed.labels(job_name='retrain').inc(count)
    job_duration.labels(job_name='retrain').set(time.time() - start_time)
    
    # Push to gateway
    push_to_gateway(
        'pushgateway:9091',
        job='spark_retrain',
        registry=registry,
    )
```

### Option 2: Spark Prometheus Sink (JMX)

```properties
# metrics.properties
*.sink.prometheus.class=org.apache.spark.metrics.sink.PrometheusServletSink
*.sink.prometheus.path=/metrics

master.source.jvm.class=org.apache.spark.metrics.source.JvmSource
driver.source.jvm.class=org.apache.spark.metrics.source.JvmSource
executor.source.jvm.class=org.apache.spark.metrics.source.JvmSource
```

Submit with:
```bash
spark-submit \
  --conf spark.metrics.conf=metrics.properties \
  --conf spark.ui.prometheus.enabled=true \
  your_job.py
```

### Option 3: OpenMetrics (Recommended)

Add to `spark-defaults.conf`:
```properties
spark.metrics.namespace=spark
spark.metrics.staticSources.enabled=true
spark.metrics.appStatusSource.enabled=true
spark.executor.processTreeMetrics.enabled=true
```

---

## Alert Configuration

### Key Alerts

| Alert | Condition | Action |
|-------|-----------|--------|
| HighPredictionLatency | P99 > 500ms | Scale up replicas |
| ModelDriftDetected | drift > 0.1 | Trigger retraining |
| CriticalKafkaLag | lag > 100k | Check consumers |

### Customize Thresholds

Edit `conf/alerting_rules.yml`:

```yaml
- alert: HighPredictionLatency
  expr: |
    histogram_quantile(0.99, ...) > 0.5  # Change threshold here
  for: 5m
  labels:
    severity: warning
```

### Test Alerts

```bash
# Validate rules
promtool check rules conf/alerting_rules.yml

# Test alertmanager routing
amtool check-config conf/alertmanager.yml
```

---

## Dashboard Import

### Via Grafana UI

1. Open Grafana → Dashboards → Import
2. Upload `conf/grafana/dashboards/ml_metrics.json`
3. Select Prometheus datasource

### Via API

```bash
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @conf/grafana/dashboards/ml_metrics.json
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SLACK_WEBHOOK_URL` | Slack webhook | Required |
| `SMTP_PASSWORD` | Email password | Required |
| `PAGERDUTY_SERVICE_KEY` | PagerDuty key | Optional |

---

## Troubleshooting

### No metrics showing

```bash
# Check targets
curl http://localhost:9090/api/v1/targets

# Check specific metric
curl 'http://localhost:9090/api/v1/query?query=up'
```

### Alerts not firing

```bash
# Check pending alerts
curl http://localhost:9090/api/v1/alerts

# Check Alertmanager
curl http://localhost:9093/api/v1/alerts
```
