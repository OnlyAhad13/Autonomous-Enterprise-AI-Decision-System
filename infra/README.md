# Infrastructure Operator Guide

This document provides guidance for deploying and operating the Enterprise AI Decision System infrastructure.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Deployment Steps](#deployment-steps)
- [Secrets Management](#secrets-management)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                            │
│  (Tests → Build → Push to GHCR → Deploy to K8s)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes (EKS/GKE)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Predict   │  │    RAG      │  │   MLflow    │              │
│  │   Service   │  │   Service   │  │   Server    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Amazon MSK   │  │    RDS        │  │      S3       │
│   (Kafka)     │  │  (PostgreSQL) │  │  (Artifacts)  │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.5.0
- kubectl >= 1.28
- Helm >= 3.13
- Docker (for local builds)

---

## Deployment Steps

### 1. Provision Infrastructure with Terraform

```bash
cd infra/terraform

# Initialize Terraform
terraform init

# Review the plan
terraform plan -var="environment=staging"

# Apply
terraform apply -var="environment=staging"

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name enterprise-ai-cluster
```

### 2. Create Kubernetes Secrets

```bash
# Create namespace
kubectl create namespace ml-services

# Create GHCR pull secret
kubectl create secret docker-registry ghcr-secret \
  --namespace ml-services \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN

# Create application secrets
kubectl create secret generic predict-secrets \
  --namespace ml-services \
  --from-literal=MLFLOW_TRACKING_URI="postgresql://..." \
  --from-literal=AWS_ACCESS_KEY_ID="..." \
  --from-literal=AWS_SECRET_ACCESS_KEY="..."
```

### 3. Deploy Services with Helm

```bash
# Deploy predict service
helm upgrade --install predict ./deploy/helm/predict \
  --namespace ml-services \
  --set image.tag=latest

# Deploy RAG service
helm upgrade --install rag ./deploy/helm/rag \
  --namespace ml-services \
  --set image.tag=latest
```

### 4. Verify Deployment

```bash
kubectl get pods -n ml-services
kubectl get svc -n ml-services
kubectl logs -f deployment/predict -n ml-services
```

---

## Secrets Management

### Recommended Approaches

| Approach | Use Case | Pros | Cons |
|----------|----------|------|------|
| **AWS Secrets Manager** | Production | Rotation, audit trail | Cost |
| **External Secrets Operator** | K8s-native | GitOps-friendly | Setup complexity |
| **HashiCorp Vault** | Enterprise | Full-featured | Operational overhead |
| **Sealed Secrets** | Simple GitOps | Easy to use | No rotation |

### AWS Secrets Manager Integration

```yaml
# External Secrets Operator example
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mlflow-credentials
  namespace: ml-services
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: mlflow-credentials
  data:
    - secretKey: db-password
      remoteRef:
        key: enterprise-ai/mlflow-db-password
        property: password
```

### Required Secrets

| Secret | Description | Source |
|--------|-------------|--------|
| `ghcr-secret` | GitHub Container Registry | GitHub PAT |
| `mlflow-credentials` | MLflow DB password | AWS Secrets Manager |
| `kafka-credentials` | MSK authentication | AWS Secrets Manager |
| `slack-webhook` | Slack notifications | Manual |

### GitHub Actions Secrets

Set these in your repository settings:

```
KUBE_CONFIG_STAGING    # Base64 encoded kubeconfig for staging
KUBE_CONFIG_PRODUCTION # Base64 encoded kubeconfig for production
```

Generate with:
```bash
cat ~/.kube/config | base64 -w 0
```

---

## Monitoring & Observability

### Prometheus Metrics

All services expose `/metrics` endpoint. Configure Prometheus to scrape:

```yaml
# conf/prometheus.yml
scrape_configs:
  - job_name: 'predict-service'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `http_request_duration_seconds` | Request latency | p99 > 1s |
| `model_inference_time_ms` | Inference time | p95 > 500ms |
| `model_drift_score` | Model drift | > 0.1 |
| `kafka_consumer_lag` | Consumer lag | > 10000 |

### Logging

Logs are sent to stdout and collected by the cluster's logging solution (e.g., Fluent Bit → CloudWatch/Elasticsearch).

---

## Troubleshooting

### Common Issues

#### 1. Pods not starting

```bash
kubectl describe pod <pod-name> -n ml-services
kubectl logs <pod-name> -n ml-services --previous
```

#### 2. Image pull errors

```bash
# Check secret exists
kubectl get secret ghcr-secret -n ml-services

# Verify secret content
kubectl get secret ghcr-secret -n ml-services -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d
```

#### 3. Database connection issues

```bash
# Test from pod
kubectl exec -it <pod> -n ml-services -- \
  psql "postgresql://user:pass@host:5432/db" -c "SELECT 1"
```

#### 4. Helm upgrade failures

```bash
# Check release status
helm status predict -n ml-services

# Rollback if needed
helm rollback predict 1 -n ml-services
```

---

## Cost Optimization

| Resource | Staging | Production |
|----------|---------|------------|
| EKS nodes | 2x m5.large | 3x m5.xlarge |
| RDS | db.t3.medium | db.r5.large |
| MSK | - | 3x kafka.m5.large |

Estimated monthly cost:
- **Staging**: ~$500/month
- **Production**: ~$2,500/month

---

## Disaster Recovery

### Backup Schedule

- **RDS**: Automated daily snapshots (7-day retention)
- **S3**: Versioning enabled
- **Kafka**: Topic replication factor = 3

### Recovery Procedures

1. **RDS Recovery**: Restore from snapshot via AWS Console or Terraform
2. **EKS Recovery**: Re-apply Terraform + Helm charts
3. **Data Recovery**: S3 versioning allows object recovery

---

## Support

- **Slack**: #ml-platform-support
- **On-call**: PagerDuty rotation
- **Runbooks**: `docs/runbooks/`
