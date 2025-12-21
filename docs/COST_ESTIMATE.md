# Cost Estimate

Resource sizing and cost estimates for Enterprise AI Decision System.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Resource Sizing](#resource-sizing)
- [AWS Cost Estimate](#aws-cost-estimate)
- [GCP Cost Estimate](#gcp-cost-estimate)
- [Scaling Guidelines](#scaling-guidelines)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                     Kubernetes Cluster                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Predict    │  │    RAG      │  │   Agent     │              │
│  │  (3 pods)   │  │  (2 pods)   │  │  (2 pods)   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└───────────────────────────┼─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Kafka (MSK)  │   │  RDS (MLflow) │   │  S3 Storage   │
│   3 brokers   │   │  db.r5.large  │   │   500 GB      │
└───────────────┘   └───────────────┘   └───────────────┘
```

---

## Resource Sizing

### Kubernetes Nodes

| Environment | Node Type | Count | vCPU | Memory |
|-------------|-----------|-------|------|--------|
| **Staging** | m5.large | 2 | 4 | 16 GB |
| **Production** | m5.xlarge | 3 | 12 | 48 GB |
| **Production (GPU)** | g4dn.xlarge | 1 | 4 | 16 GB |

### Pod Resources

| Service | Replicas | CPU Request | CPU Limit | Memory |
|---------|----------|-------------|-----------|--------|
| Predict | 3 | 500m | 1000m | 1 Gi |
| RAG | 2 | 500m | 2000m | 2 Gi |
| Agent | 2 | 250m | 500m | 512 Mi |
| MLflow | 1 | 250m | 500m | 1 Gi |

### Kafka (MSK)

| Tier | Brokers | Instance | Storage | Throughput |
|------|---------|----------|---------|------------|
| Staging | 2 | kafka.t3.small | 50 GB | 1 MB/s |
| Production | 3 | kafka.m5.large | 500 GB | 10 MB/s |

### Database (RDS)

| Tier | Instance | Storage | IOPS |
|------|----------|---------|------|
| Staging | db.t3.medium | 50 GB | Baseline |
| Production | db.r5.large | 200 GB | 3000 |

### Storage (S3)

| Bucket | Size | Access Pattern |
|--------|------|----------------|
| MLflow Artifacts | 200 GB | Infrequent |
| Training Data | 500 GB | Frequent |
| Model Registry | 50 GB | Infrequent |

---

## AWS Cost Estimate

### Staging (~$500/month)

| Resource | Type | Quantity | Unit Cost | Monthly |
|----------|------|----------|-----------|---------|
| EKS Cluster | - | 1 | $72 | $72 |
| EC2 (m5.large) | On-Demand | 2 | $70 | $140 |
| RDS (db.t3.medium) | - | 1 | $50 | $50 |
| S3 | 100 GB | - | $2.30 | $3 |
| NAT Gateway | - | 1 | $32 | $32 |
| Load Balancer | ALB | 1 | $16 | $16 |
| Data Transfer | 100 GB | - | $9 | $9 |
| **Total** | | | | **~$320** |

### Production (~$2,500/month)

| Resource | Type | Quantity | Unit Cost | Monthly |
|----------|------|----------|-----------|---------|
| EKS Cluster | - | 1 | $72 | $72 |
| EC2 (m5.xlarge) | On-Demand | 3 | $140 | $420 |
| EC2 (g4dn.xlarge) | Training | 1 | $380 | $380 |
| MSK (kafka.m5.large) | - | 3 | $165 | $495 |
| RDS (db.r5.large) | Multi-AZ | 1 | $350 | $350 |
| S3 | 750 GB | - | $17 | $17 |
| NAT Gateway | - | 2 | $32 | $64 |
| Load Balancer | ALB | 1 | $16 | $16 |
| CloudWatch | Logs | - | $50 | $50 |
| Secrets Manager | - | 10 | $0.40 | $4 |
| Data Transfer | 500 GB | - | $45 | $45 |
| **Total** | | | | **~$1,910** |

### Cost Optimization Options

| Strategy | Savings | Description |
|----------|---------|-------------|
| Reserved Instances (1yr) | 30-40% | Commit to 1-year usage |
| Spot Instances (workers) | 60-70% | For batch/training workloads |
| Savings Plans | 20-30% | Flexible compute commitment |
| ARM instances (Graviton) | 20% | Use m6g instead of m5 |

---

## GCP Cost Estimate

### Staging (~$450/month)

| Resource | Type | Quantity | Monthly |
|----------|------|----------|---------|
| GKE Cluster | Autopilot | 1 | $72 |
| GCE (e2-standard-2) | - | 2 | $100 |
| Cloud SQL (db-n1-standard-2) | - | 1 | $80 |
| Cloud Storage | 100 GB | - | $3 |
| NAT | - | 1 | $32 |
| Load Balancer | - | 1 | $18 |
| **Total** | | | **~$305** |

### Production (~$2,200/month)

| Resource | Type | Quantity | Monthly |
|----------|------|----------|---------|
| GKE Cluster | Standard | 1 | $72 |
| GCE (e2-standard-4) | - | 3 | $320 |
| GPU (n1-standard-4 + T4) | - | 1 | $350 |
| Confluent Cloud (Basic) | - | 1 | $400 |
| Cloud SQL (db-n1-standard-4) | HA | 1 | $300 |
| Cloud Storage | 750 GB | - | $17 |
| NAT | - | 2 | $64 |
| Load Balancer | - | 1 | $18 |
| Stackdriver | - | - | $50 |
| **Total** | | | **~$1,590** |

---

## Scaling Guidelines

### Horizontal Scaling

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU > 70% | 5 min | Add pod replica |
| Memory > 80% | 5 min | Add pod replica |
| Request latency P95 > 500ms | 5 min | Add pod replica |
| Kafka lag > 10,000 | 10 min | Add consumer |

### HPA Configuration

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: predict-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: predict
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Vertical Scaling

| Scenario | Current | Upgrade To | When |
|----------|---------|------------|------|
| Memory pressure | m5.large | m5.xlarge | OOM events |
| CPU saturation | m5.xlarge | m5.2xlarge | CPU > 90% sustained |
| Kafka throughput | kafka.m5.large | kafka.m5.2xlarge | Producer lag |

### Throughput Planning

| Metric | Staging | Production | Peak |
|--------|---------|------------|------|
| Predictions/sec | 100 | 1,000 | 5,000 |
| Kafka messages/sec | 1,000 | 10,000 | 50,000 |
| Training jobs/day | 1 | 4 | 12 |

---

## Cost Monitoring

### AWS Cost Alerts

```hcl
resource "aws_budgets_budget" "monthly" {
  name         = "enterprise-ai-monthly"
  budget_type  = "COST"
  limit_amount = "3000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator       = "GREATER_THAN"
    threshold                 = 80
    threshold_type           = "PERCENTAGE"
    notification_type        = "FORECASTED"
    subscriber_email_addresses = ["team@company.com"]
  }
}
```

### Key Cost Drivers

| Component | % of Total | Optimization |
|-----------|------------|--------------|
| Compute (K8s) | 40% | Use spot for workers |
| Kafka (MSK) | 25% | Right-size brokers |
| Database (RDS) | 15% | Reserved instances |
| GPU (Training) | 15% | Use spot, schedule jobs |
| Storage/Network | 5% | Lifecycle policies |

---

## Summary

| Environment | AWS | GCP | Recommendation |
|-------------|-----|-----|----------------|
| Staging | ~$320/mo | ~$305/mo | GCP slightly cheaper |
| Production | ~$1,910/mo | ~$1,590/mo | GCP for cost, AWS for MSK |

**Break-even for Reserved Instances**: 7-8 months
